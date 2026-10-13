import ast
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import yaml

from robonex_common.foot_roll import foot_roll
from robonex_common.models import VER2_EDU, action_pose

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
EXPORT = ROOT / "scripts/export_policy_manifest.py"


def contract():
    spec = importlib.util.spec_from_file_location("robot_contract_v2_lean", TASK / "robot_contract_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_module():
    spec = importlib.util.spec_from_file_location("export_policy_manifest_lean", EXPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LeanPoseContractTests(unittest.TestCase):
    def setUp(self):
        self.c = contract()

    def test_home_pose_is_the_existing_contract(self):
        offsets, joint_pos, height = self.c.pose_contract("home")
        self.assertEqual(offsets, self.c.ACTION_OFFSETS)
        self.assertEqual(joint_pos, self.c.CLOSED_LOOP_DEFAULT_JOINT_POS)
        self.assertEqual(height, self.c.BASE_HEIGHT)

    def test_lean25_offsets_come_from_robonex_common_and_keep_the_scales(self):
        offsets, joint_pos, height = self.c.pose_contract("lean25")
        common = action_pose("ver2_edu", "lean25")
        self.assertEqual(offsets, {n: common[n] for n in self.c.LEG_JOINTS})
        self.assertEqual(set(joint_pos), set(self.c.CLOSED_LOOP_DEFAULT_JOINT_POS))
        self.assertEqual(self.c.ACTION_SCALES, contract().ACTION_SCALES)
        self.assertLess(height, self.c.BASE_HEIGHT)
        self.assertGreater(height, self.c.BASE_HEIGHT - 0.005)

    def test_lean25_tilts_only_the_ankle_and_hip_pitch_and_keeps_the_knee(self):
        _, home, _ = self.c.pose_contract("home")
        _, lean, _ = self.c.pose_contract("lean25")
        for s in "lr":
            self.assertAlmostEqual(math.degrees(lean[f"{s}_ankle_pitch_joint"] - home[f"{s}_ankle_pitch_joint"]), 2.5, places=3)
            self.assertAlmostEqual(lean[f"{s}_knee_joint"], home[f"{s}_knee_joint"], places=9)
            self.assertAlmostEqual(lean[f"{s}_knee_pitch_joint"], home[f"{s}_knee_pitch_joint"], places=9)
            self.assertAlmostEqual(lean[f"{s}_knee_coupler_joint_a"], home[f"{s}_knee_coupler_joint_a"], places=9)
            self.assertAlmostEqual(lean[f"{s}_ankle_roll_joint"], home[f"{s}_ankle_roll_joint"], places=4)
        self.assertAlmostEqual(math.degrees(lean["l_hip_pitch_joint"] - home["l_hip_pitch_joint"]), -2.5, places=9)
        self.assertAlmostEqual(math.degrees(lean["r_hip_pitch_joint"] - home["r_hip_pitch_joint"]), 2.5, places=9)

    def test_lean25_stays_inside_the_foot_roll_clip(self):
        offsets, _, _ = self.c.pose_contract("lean25")
        roll = VER2_EDU.foot_roll
        for upper, lower, sign in roll.pairs:
            self.assertLess(abs(foot_roll(sign * offsets[upper], sign * offsets[lower], roll.coeffs)), math.radians(1.0))

    def test_unknown_pose_is_refused(self):
        with self.assertRaises(Exception):
            self.c.pose_contract("lean99")

    def test_lean_pose_is_refused_for_other_variants(self):
        for variant in ("pro", "max"):
            with self.assertRaises(ValueError):
                self.c.pose_contract("lean25", variant)
        self.assertEqual(self.c.pose_contract("home", "pro")[2], self.c.BASE_HEIGHT)


class LeanTaskWiringTests(unittest.TestCase):
    def test_lean_cfg_sets_the_pose_and_the_base_cfg_defaults_to_home(self):
        tree = ast.parse((TASK / "robonex_walking_v2_env_cfg.py").read_text())
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

        def field(cls, name):
            for node in classes[cls].body:
                if isinstance(node, ast.AnnAssign) and node.target.id == name:
                    return ast.literal_eval(node.value)
            return None

        self.assertEqual(field("RoboNexWalkingV2EnvCfg", "action_pose"), "home")
        self.assertEqual(field("RoboNexWalkingV2EduLean25EnvCfg", "action_pose"), "lean25")
        self.assertEqual(field("RoboNexWalkingV2EduLean25EnvCfg", "variant"), "edu")
        self.assertIsNone(field("RoboNexWalkingV2EduEnvCfg", "action_pose"))

    def test_lean_task_is_registered_with_the_edu_runner(self):
        text = (TASK / "__init__.py").read_text()
        self.assertIn('id="RoboNex-Walking-V2-Edu-Lean25-v0"', text)
        self.assertIn("robonex_walking_v2_env_cfg:RoboNexWalkingV2EduLean25EnvCfg", text)
        self.assertIn("rsl_rl_ppo_cfg:V2EduPPORunnerCfg", text)


class LeanExportTests(unittest.TestCase):
    def setUp(self):
        self.export = export_module()
        self.c = contract()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def saved_env(self, pose, offsets_pose=None):
        offsets, _, _ = self.c.pose_contract(offsets_pose or pose)
        roll = VER2_EDU.foot_roll
        data = {
            "actions": {"joint_pos": {
                "offset": offsets,
                "scale": dict(self.c.ACTION_SCALES),
                "clip": {n: list(v) for n, v in self.c.ACTION_CLIPS.items()},
                "foot_roll_limit": roll.limit,
                "foot_roll_coeffs": list(roll.coeffs),
                "foot_roll_pairs": [list(p) for p in roll.pairs],
            }},
        }
        if pose != "home":
            data["action_pose"] = pose
        path = Path(self.tmp.name) / f"env_{pose}_{offsets_pose}.yaml"
        path.write_text(yaml.safe_dump(data))
        return path

    def infer(self, path):
        offsets, scales, clips = self.export.saved_action_normalization(path)
        return self.export.infer_robot_model(path, offsets, scales, clips)

    def test_home_and_lean_checkpoints_infer_their_pose(self):
        home = self.saved_env("home")
        lean = self.saved_env("lean25")
        self.assertEqual(self.infer(home), ("ver2_edu", "home"))
        self.assertEqual(self.export.saved_action_pose(home), "home")
        self.assertEqual(self.infer(lean), ("ver2_edu", "lean25"))
        self.assertEqual(self.export.saved_action_pose(lean), "lean25")
        self.assertEqual(self.export.POSE_TASKS[("ver2_edu", "lean25")], "RoboNex-Walking-V2-Edu-Lean25-v0")

    def test_offsets_that_match_no_pose_are_refused(self):
        path = self.saved_env("home")
        offsets, scales, clips = self.export.saved_action_normalization(path)
        offsets["l_ankle_upper_joint"] += 0.01
        with self.assertRaises(SystemExit):
            self.export.infer_robot_model(path, offsets, scales, clips)


if __name__ == "__main__":
    unittest.main()
