import ast
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1] / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
REWARDS = ROOT / "mdp/rewards.py"
ENV_CFG = ROOT / "robonex_walking_v2_env_cfg.py"

LEG_JOINTS = (
    "l_hip_yaw_joint", "l_hip_pitch_joint", "l_hip_roll_joint", "l_knee_pitch_joint",
    "l_ankle_upper_joint", "l_ankle_lower_joint",
    "r_hip_yaw_joint", "r_hip_pitch_joint", "r_hip_roll_joint", "r_knee_pitch_joint",
    "r_ankle_upper_joint", "r_ankle_lower_joint",
)
RS03 = {"hip_pitch", "hip_roll", "knee_pitch"}
RATED = {name: 13.0 if any(k in name for k in RS03) else 6.0 for name in LEG_JOINTS}


def load():
    node = next(n for n in ast.parse(REWARDS.read_text()).body if getattr(n, "name", None) == "joint_load_rating_l2")
    ns = {"torch": torch, "ManagerBasedRLEnv": object, "SceneEntityCfg": object, "Articulation": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "rewards", "exec"), ns)
    return ns["joint_load_rating_l2"]


def make_env(torque, joint_names=LEG_JOINTS):
    asset = types.SimpleNamespace(joint_names=list(joint_names),
                                  data=types.SimpleNamespace(applied_torque=torch.tensor(torque, dtype=torch.float32)))
    return types.SimpleNamespace(scene={"robot": asset})


def cfg(joint_ids):
    return types.SimpleNamespace(name="robot", joint_ids=joint_ids)


class JointLoadTest(unittest.TestCase):
    def setUp(self):
        self.fn = load()

    def test_numeric_value_on_a_synthetic_torque_vector(self):
        torque = [[6.0, 13.0, -13.0, 6.5, 3.0, -3.0, 0.0, 26.0, 0.0, 0.0, 0.0, 12.0]]
        out = self.fn(make_env(torque), asset_cfg=cfg(list(range(12))), rated=RATED)
        expected = 1.0 + 1.0 + 1.0 + 0.25 + 0.25 + 0.25 + 0.0 + 4.0 + 0.0 + 0.0 + 0.0 + 4.0
        self.assertAlmostEqual(out.item(), expected, places=5)
        self.assertEqual(out.shape, (1,))

    def test_at_rating_every_joint_scores_one(self):
        torque = [[RATED[n] for n in LEG_JOINTS], [-RATED[n] for n in LEG_JOINTS]]
        out = self.fn(make_env(torque), asset_cfg=cfg(list(range(12))), rated=RATED)
        self.assertTrue(torch.allclose(out, torch.full((2,), 12.0)))

    def test_clamp_at_max_ratio(self):
        torque = [[0.0] * 11 + [6.0 * 5.0], [0.0] * 11 + [6.0 * 50.0], [0.0] * 11 + [-6.0 * 50.0]]
        out = self.fn(make_env(torque), asset_cfg=cfg(list(range(12))), rated=RATED)
        self.assertTrue(torch.allclose(out, torch.full((3,), 25.0)))
        out = self.fn(make_env(torque), asset_cfg=cfg(list(range(12))), rated=RATED, max_ratio=2.0)
        self.assertTrue(torch.allclose(out, torch.full((3,), 4.0)))

    def test_nonfinite_torque_counts_as_full_load(self):
        torque = [[0.0] * 11 + [float("nan")], [0.0] * 11 + [float("inf")]]
        out = self.fn(make_env(torque), asset_cfg=cfg(list(range(12))), rated=RATED)
        self.assertTrue(torch.allclose(out, torch.full((2,), 25.0)))

    def test_rating_follows_joint_names_in_policy_order(self):
        names = ("r_ankle_lower_joint", "l_hip_roll_joint", "spare_passive", "r_knee_pitch_joint", "l_hip_yaw_joint")
        torque = [[6.0, 13.0, 99.0, 26.0, 3.0]]
        ids = [1, 3, 0, 4]
        out = self.fn(make_env(torque, names), asset_cfg=cfg(ids), rated=RATED)
        self.assertAlmostEqual(out.item(), 1.0 + 4.0 + 1.0 + 0.25, places=5)
        env = make_env([[6.0, 13.0]], ("r_ankle_lower_joint", "l_hip_roll_joint"))
        out = self.fn(env, asset_cfg=cfg(slice(None)), rated=RATED)
        self.assertAlmostEqual(out.item(), 2.0, places=5)
        with self.assertRaises(KeyError):
            self.fn(make_env(torque, names), asset_cfg=cfg([2]), rated=RATED)

    def test_mirror_invariance(self):
        left = [2.0, 9.0, -11.7, 7.5, 1.0, -1.5]
        right = [-0.5, 4.0, 12.2, -3.0, 5.5, 2.0]
        torque = [left + right, right + left]
        out = self.fn(make_env(torque), asset_cfg=cfg(list(range(12))), rated=RATED)
        self.assertAlmostEqual(out[0].item(), out[1].item(), places=5)

    def test_cfg_term_is_off_by_default_and_uses_common_ratings(self):
        tree = ast.parse(ENV_CFG.read_text())
        consts = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        self.assertEqual(ast.literal_eval(consts["JOINT_LOAD_WEIGHT"]), 0.0)
        self.assertEqual(ast.literal_eval(consts["JOINT_LOAD_MAX_RATIO"]), 5.0)
        rewards = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg")
        term = next(n for n in rewards.body if isinstance(n, ast.Assign) and n.targets[0].id == "joint_load")
        kw = {k.arg: k.value for k in term.value.keywords}
        self.assertEqual(kw["weight"].id, "JOINT_LOAD_WEIGHT")
        self.assertEqual(kw["func"].attr, "joint_load_rating_l2")
        params = {ast.literal_eval(k): v for k, v in zip(kw["params"].keys, kw["params"].values)}
        self.assertEqual(params["rated"].id, "RATED_TORQUE_STANDSTILL")
        self.assertEqual(params["max_ratio"].id, "JOINT_LOAD_MAX_RATIO")
        asset_kw = {k.arg: k.value for k in params["asset_cfg"].keywords}
        self.assertEqual(asset_kw["joint_names"].id, "LEG_JOINTS")
        self.assertTrue(ast.literal_eval(asset_kw["preserve_order"]))


if __name__ == "__main__":
    unittest.main()
