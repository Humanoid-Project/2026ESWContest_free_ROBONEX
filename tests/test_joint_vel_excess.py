import ast
import copy
import importlib.util
import math
import sys
import tempfile
import types
import unittest
from collections.abc import Iterable, Mapping, Sized
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
REWARDS = TASK / "mdp/rewards.py"
JOINT_SPEED = TASK / "mdp/joint_speed.py"
TRACKING = TASK / "mdp/tracking.py"
ENV_CFG = TASK / "robonex_walking_v2_env_cfg.py"
EFFECTS = ROOT / "scripts/rsl_rl/deploy_effects.py"
TRAIN = ROOT / "scripts/rsl_rl/train.py"
ISAAC = Path.home() / "IsaacLab/source/isaaclab/isaaclab"
W104_RUN = Path(
    str(Path(__file__).resolve().parents[1]) + "/logs/rsl_rl/robonex_walking_v2_edu/"
    "2026-10-03_03-42-50_W104s43a_v2_edu_trackexc200_s43"
)
TRACKING_MODULE = "robonex_walking.tasks.manager_based.robonex_walking.mdp.tracking"
JOINT_SPEED_MODULE = "robonex_walking.tasks.manager_based.robonex_walking.mdp.joint_speed"
FUNC = "robonex_walking.tasks.manager_based.robonex_walking.mdp.rewards:joint_vel_excess_l2"

LEG_JOINTS = (
    "l_hip_yaw_joint", "l_hip_pitch_joint", "l_hip_roll_joint", "l_knee_pitch_joint",
    "l_ankle_upper_joint", "l_ankle_lower_joint",
    "r_hip_yaw_joint", "r_hip_pitch_joint", "r_hip_roll_joint", "r_knee_pitch_joint",
    "r_ankle_upper_joint", "r_ankle_lower_joint",
)
WEIGHT = -0.5
FLOOR = 8.0
CAP = 20.0


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_nodes(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def scene_entity(name, joint_ids=slice(None)):
    return types.SimpleNamespace(name=name, joint_ids=joint_ids)


def load_reward():
    speed = load_module("joint_speed_under_test", JOINT_SPEED)
    ns = {"torch": torch, "math": math, "joint_speed_excess": speed.joint_speed_excess,
          "SceneEntityCfg": scene_entity, "Articulation": object, "ManagerBasedRLEnv": object}
    return load_nodes(REWARDS, {"joint_vel_excess_l2"}, ns)["joint_vel_excess_l2"]


def make_env(velocity):
    data = types.SimpleNamespace(joint_vel=torch.as_tensor(velocity, dtype=torch.float32))
    return types.SimpleNamespace(scene={"robot": types.SimpleNamespace(data=data)})


def one_joint(speed, n=12, index=0):
    velocity = [0.0] * n
    velocity[index] = speed
    return [velocity]


def isaac_update_class_from_dict():
    ns = {"Iterable": Iterable, "Mapping": Mapping, "Sized": Sized, "Any": object,
          "string_to_callable": lambda s: s}
    return load_nodes(ISAAC / "utils/dict.py", {"update_class_from_dict"}, ns)["update_class_from_dict"]


class FakeIsaacModules:
    def __enter__(self):
        fake = types.ModuleType("isaaclab.utils.dict")
        fake.update_class_from_dict = isaac_update_class_from_dict()
        self.saved = {k: sys.modules.get(k) for k in ("isaaclab", "isaaclab.utils", "isaaclab.utils.dict")}
        sys.modules["isaaclab"] = types.ModuleType("isaaclab")
        sys.modules["isaaclab.utils"] = types.ModuleType("isaaclab.utils")
        sys.modules["isaaclab.utils.dict"] = fake
        return fake.update_class_from_dict

    def __exit__(self, *exc):
        for key, module in self.saved.items():
            if module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = module


def to_cfg(value, key=None):
    if isinstance(value, dict) and key != "params":
        return types.SimpleNamespace(**{k: to_cfg(v, k) for k, v in value.items()})
    return copy.deepcopy(value)


def new_term(weight=0.0):
    return types.SimpleNamespace(
        func=FUNC,
        weight=weight,
        params={"floor_rad_s": FLOOR, "cap_rad_s": CAP,
                "asset_cfg": {"name": "robot", "joint_names": list(LEG_JOINTS)}},
    )


class KernelTest(unittest.TestCase):
    def setUp(self):
        self.fn = load_reward()
        self.ids = scene_entity("robot", list(range(12)))

    def rate(self, speed, index=0):
        return WEIGHT * self.fn(make_env(one_joint(speed, index=index)), asset_cfg=self.ids).item()

    def test_rates_at_the_design_points(self):
        for speed, expected in ((0.0, 0.0), (8.0, 0.0), (10.0, -2.0), (12.0, -8.0), (16.0, -32.0)):
            self.assertAlmostEqual(self.rate(speed), expected, places=4, msg=speed)
            self.assertAlmostEqual(self.rate(speed), WEIGHT * max(speed - FLOOR, 0.0) ** 2, places=4)
        self.assertEqual(self.rate(7.99), 0.0)
        self.assertAlmostEqual(self.rate(10.0) * 0.02, -0.04, places=6)
        self.assertAlmostEqual(self.rate(16.0) * 0.02, -0.64, places=5)

    def test_floor_and_cap_parameters(self):
        env = make_env(one_joint(12.0))
        self.assertAlmostEqual(self.fn(env, floor_rad_s=9.0, asset_cfg=self.ids).item(), 9.0, places=5)
        self.assertAlmostEqual(self.fn(env, cap_rad_s=11.0, asset_cfg=self.ids).item(), 9.0, places=5)
        far = make_env(one_joint(55.0))
        self.assertAlmostEqual(self.fn(far, asset_cfg=self.ids).item(), (CAP - FLOOR) ** 2, places=4)

    def test_sign_symmetry_and_every_joint_alike(self):
        for index in range(12):
            self.assertAlmostEqual(self.rate(12.0, index), self.rate(-12.0, index), places=6)
            self.assertAlmostEqual(self.rate(12.0, index), self.rate(12.0, 0), places=6)

    def test_sum_over_joints_not_mean(self):
        rows = [[12.0, -12.0] + [0.0] * 10, [12.0] * 12, [7.0] * 12]
        out = self.fn(make_env(rows), asset_cfg=self.ids)
        self.assertEqual(out.shape, (3,))
        self.assertAlmostEqual(out[0].item(), 2.0 * 16.0, places=5)
        self.assertAlmostEqual(out[1].item(), 12.0 * 16.0, places=4)
        self.assertEqual(out[2].item(), 0.0)

    def test_selected_ids_permuted_noncontiguous_and_passive_excluded(self):
        n = 16
        velocity = [0.0] * n
        actuated = [15, 2, 9, 0, 11, 4, 13, 6, 1, 8, 3, 10]
        passive = sorted(set(range(n)) - set(actuated))
        for i in passive:
            velocity[i] = 30.0
        velocity[9] = 12.0
        velocity[15] = -10.0
        env = make_env([velocity])
        expected = 16.0 + 4.0
        for ids in (actuated, list(reversed(actuated)), sorted(actuated)):
            self.assertAlmostEqual(self.fn(env, asset_cfg=scene_entity("robot", ids)).item(), expected, places=5)
        everything = self.fn(env, asset_cfg=scene_entity("robot", slice(None))).item()
        self.assertGreater(everything, expected + 100.0)

    def test_nonfinite_counts_as_the_cap_and_stays_finite(self):
        cap = (CAP - FLOOR) ** 2
        rows = [
            [math.nan] + [0.0] * 11,
            [math.inf] + [0.0] * 11,
            [-math.inf] + [0.0] * 11,
            [math.nan] * 12,
        ]
        out = self.fn(make_env(rows), asset_cfg=self.ids)
        self.assertTrue(bool(torch.isfinite(out).all()))
        for row in range(3):
            self.assertAlmostEqual(out[row].item(), cap, places=4)
        self.assertAlmostEqual(out[3].item(), 12.0 * cap, places=3)
        self.assertTrue(bool((out <= 12.0 * cap + 1e-3).all()))


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(ENV_CFG.read_text())
        self.consts = {n.targets[0].id: n.value for n in self.tree.body
                       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        self.rewards = next(n for n in self.tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg")
        self.terminations = next(
            n for n in self.tree.body if isinstance(n, ast.ClassDef) and n.name == "TerminationsCfg")

    def test_term_present_inert_and_on_the_actuated_joints(self):
        self.assertEqual(ast.literal_eval(self.consts["JOINT_VEL_EXCESS_WEIGHT"]), 0.0)
        self.assertEqual(ast.literal_eval(self.consts["JOINT_VEL_EXCESS_FLOOR_RAD_S"]), FLOOR)
        self.assertEqual(ast.literal_eval(self.consts["JOINT_VEL_EXCESS_CAP_RAD_S"]), CAP)
        term = next(n for n in self.rewards.body if isinstance(n, ast.Assign) and n.targets[0].id == "joint_vel_excess")
        kw = {k.arg: k.value for k in term.value.keywords}
        self.assertEqual(kw["weight"].id, "JOINT_VEL_EXCESS_WEIGHT")
        self.assertEqual(kw["func"].attr, "joint_vel_excess_l2")
        params = {ast.literal_eval(k): v for k, v in zip(kw["params"].keys, kw["params"].values)}
        self.assertEqual(params["floor_rad_s"].id, "JOINT_VEL_EXCESS_FLOOR_RAD_S")
        self.assertEqual(params["cap_rad_s"].id, "JOINT_VEL_EXCESS_CAP_RAD_S")
        asset = params["asset_cfg"]
        self.assertEqual(ast.literal_eval(asset.args[0]), "robot")
        asset_kw = {k.arg: k.value for k in asset.keywords}
        self.assertEqual(asset_kw["joint_names"].id, "LEG_JOINTS")
        self.assertTrue(ast.literal_eval(asset_kw["preserve_order"]))

    def test_new_term_is_last_and_the_existing_joint_vel_term_is_unchanged(self):
        names = [n.targets[0].id for n in self.rewards.body if isinstance(n, ast.Assign)]
        self.assertEqual(names[-1], "joint_vel_excess")
        self.assertEqual(names.count("joint_vel_excess"), 1)
        joint_vel = next(n for n in self.rewards.body if isinstance(n, ast.Assign) and n.targets[0].id == "joint_vel")
        kw = {k.arg: k.value for k in joint_vel.value.keywords}
        self.assertEqual(ast.literal_eval(kw["weight"]), -0.1)
        self.assertEqual(kw["func"].attr, "joint_vel_l2_bounded")

    def test_cap_is_below_the_bad_joint_vel_termination(self):
        term = next(n for n in self.terminations.body
                    if isinstance(n, ast.Assign) and n.targets[0].id == "bad_joint_vel")
        kw = {k.arg: k.value for k in term.value.keywords}
        limit = ast.literal_eval(kw["params"])["limit"]
        self.assertLess(CAP, limit)
        self.assertLess(FLOOR, 10.0)

    def test_reward_manager_skips_a_zero_weight_term(self):
        ns = {"torch": torch}
        tree = ast.parse((ISAAC / "managers/reward_manager.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardManager")
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute")
        exec(compile(ast.Module(body=[node], type_ignores=[]), "reward_manager", "exec"), ns)

        def explode(env, **kwargs):
            raise AssertionError("a zero-weight term was evaluated")

        alive = types.SimpleNamespace(func=lambda env: torch.ones(3), weight=0.5, params={})
        tracking = types.SimpleNamespace(func=lambda env, **kw: torch.full((3,), 2.0), weight=-200.0, params={})
        speed = types.SimpleNamespace(func=explode, weight=0.0, params={"floor_rad_s": FLOOR})
        manager = types.SimpleNamespace(
            _term_names=["alive", "sent_tracking_excess", "joint_vel_excess"],
            _term_cfgs=[alive, tracking, speed], _env=None,
            _reward_buf=torch.zeros(3), _step_reward=torch.full((3, 3), 7.0),
            _episode_sums={"alive": torch.zeros(3), "sent_tracking_excess": torch.zeros(3),
                           "joint_vel_excess": torch.zeros(3)},
        )
        out = ns["compute"](manager, dt=0.02)
        torch.testing.assert_close(out, torch.full((3,), 0.01 - 8.0))
        self.assertTrue(bool((manager._episode_sums["joint_vel_excess"] == 0.0).all()))
        self.assertTrue(bool((manager._step_reward[:, 2] == 0.0).all()))


class OverrideRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def cfg(self, weight=0.0):
        return types.SimpleNamespace(
            rewards=types.SimpleNamespace(
                sent_tracking_excess=types.SimpleNamespace(
                    func="rewards:sent_tracking_excess_l2", weight=0.0, params={"floor_deg": 20.0}),
                joint_vel_excess=new_term(weight),
            ),
            events=types.SimpleNamespace(), observations=types.SimpleNamespace(),
            scene=types.SimpleNamespace(robot=types.SimpleNamespace(spawn=types.SimpleNamespace(usd_path=None))),
        )

    def test_hydra_override_survives_the_saved_yaml_and_restore(self):
        try:
            from hydra import compose, initialize
            from hydra.core.config_store import ConfigStore
            from hydra.core.global_hydra import GlobalHydra
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("hydra not installed")
        live = self.cfg()
        node = {"env": {"rewards": {
            "sent_tracking_excess": {"func": "rewards:sent_tracking_excess_l2", "weight": 0.0,
                                     "params": {"floor_deg": 20.0}},
            "joint_vel_excess": {"func": FUNC, "weight": 0.0,
                                 "params": copy.deepcopy(live.rewards.joint_vel_excess.params)},
        }}}
        ConfigStore.instance().store(name="w107_vel_override_test", node=node)
        GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path=None):
            composed = compose(config_name="w107_vel_override_test",
                               overrides=[f"env.rewards.joint_vel_excess.weight={WEIGHT}",
                                          "env.rewards.sent_tracking_excess.weight=-200.0"])
        GlobalHydra.instance().clear()
        overridden = OmegaConf.to_container(composed, resolve=True)["env"]
        with FakeIsaacModules() as update:
            update(live, overridden)
            self.assertEqual(live.rewards.joint_vel_excess.weight, WEIGHT)
            self.assertIsInstance(live.rewards.joint_vel_excess.weight, float)
            saved = {"rewards": {name: {"func": term.func, "weight": term.weight, "params": term.params}
                                 for name, term in vars(live.rewards).items()}}
            run = Path(self.tmp.name)
            (run / "params").mkdir()
            (run / "params" / "env.yaml").write_text(yaml.dump(saved))
            restored = self.cfg()
            load_module("deploy_effects_vel_roundtrip", EFFECTS).apply_training_env_cfg(
                restored, str(run / "model_999.pt"))
        term = restored.rewards.joint_vel_excess
        self.assertEqual(term.weight, WEIGHT)
        self.assertEqual(term.params["floor_rad_s"], FLOOR)
        self.assertEqual(term.params["cap_rad_s"], CAP)
        self.assertEqual(term.params["asset_cfg"]["joint_names"], list(LEG_JOINTS))
        self.assertEqual(restored.rewards.sent_tracking_excess.weight, -200.0)


class OldRunRestoreTest(unittest.TestCase):
    def test_w104_env_yaml_restores_with_the_new_term_absent(self):
        path = W104_RUN / "params" / "env.yaml"
        if not path.is_file():
            self.skipTest("W104s43a run directory not found in the main worktree")
        effects = load_module("deploy_effects_vel_old_run", EFFECTS)
        with path.open() as handle:
            saved = yaml.load(handle, Loader=effects._SavedCfgLoader)
        self.assertNotIn("joint_vel_excess", saved["rewards"])
        self.assertEqual(saved["rewards"]["sent_tracking_excess"]["weight"], -200.0)
        live = to_cfg({k: v for k, v in saved.items() if k not in ("viewer", "log_dir", "seed")})
        live.rewards.joint_vel_excess = new_term()
        with FakeIsaacModules():
            restored_from = effects.apply_training_env_cfg(live, str(W104_RUN / "model_999.pt"))
        self.assertEqual(restored_from, path.resolve())
        self.assertIsNone(live.rewards.joint_vel_excess)
        present = sorted(name for name, term in vars(live.rewards).items() if term is not None)
        self.assertEqual(present, sorted(saved["rewards"]))
        for name in present:
            self.assertEqual(getattr(live.rewards, name).weight, saved["rewards"][name]["weight"], name)
        self.assertEqual(live.rewards.sent_tracking_excess.weight, -200.0)


class JointSpeedMetricsTest(unittest.TestCase):
    def setUp(self):
        self.speed = load_module("joint_speed_metrics_under_test", JOINT_SPEED)
        self.tracking = load_module("tracking_metrics_for_speed", TRACKING)
        self.names = ["a", "b", "c"]
        self.data = types.SimpleNamespace(joint_pos=torch.zeros(4, 5), joint_pos_target=torch.zeros(4, 5),
                                          joint_vel=torch.zeros(4, 5))
        self.robot = types.SimpleNamespace(data=self.data)

    def test_fractions_peaks_and_keys(self):
        metrics = self.speed.JointSpeedMetrics(self.robot, [4, 0, 2], self.names)
        self.data.joint_vel[:, 1] = 50.0
        self.data.joint_vel[0, 4] = 9.0
        self.data.joint_vel[1, 0] = -11.0
        self.data.joint_vel[2, 2] = 8.0
        self.data.joint_vel[3, 2] = 4.0
        metrics.sample()
        self.data.joint_vel.zero_()
        self.data.joint_vel[3, 2] = math.nan
        metrics.sample()
        log = metrics.take_log()
        self.assertAlmostEqual(log["Policy/joint_vel_excess_any_gt8_fraction"], 3 / 8)
        self.assertAlmostEqual(log["Policy/joint_vel_excess_any_gt10_fraction"], 2 / 8)
        self.assertAlmostEqual(log["Policy/joint_vel_excess_gt8_fraction/a"], 1 / 8)
        self.assertAlmostEqual(log["Policy/joint_vel_excess_gt10_fraction/a"], 0.0)
        self.assertAlmostEqual(log["Policy/joint_vel_excess_gt10_fraction/b"], 1 / 8)
        self.assertAlmostEqual(log["Policy/joint_vel_excess_gt10_fraction/c"], 1 / 8)
        self.assertAlmostEqual(log["Policy/joint_vel_excess_peak_rad_s_max"], 11.0, places=5)
        self.assertAlmostEqual(log["Policy/joint_vel_excess_peak_rad_s_mean"], (9.0 + 11.0 + 8.0 + 4.0) / 8, places=5)
        self.assertAlmostEqual(log["Policy/joint_vel_excess_nonfinite_fraction"], 1 / 24)
        self.assertEqual(metrics.take_log()["Policy/joint_vel_excess_any_gt8_fraction"], 0.0)

    def test_training_wrapper_samples_speed_at_the_pre_reset_boundary(self):
        data = self.data

        class BaseWrapper:
            def __init__(self, env, clip_actions=None):
                self.unwrapped = env
                self.device = "cpu"
                self.num_envs = 4
                self.num_actions = 3
                self.clip_actions = clip_actions

            def step(self, actions):
                self.unwrapped.termination_manager.compute()
                data.joint_vel.zero_()
                return actions

        diagnostics = load_nodes(TRAIN, {"DiagnosticVecEnvWrapper"},
                                 {"torch": torch, "math": math, "RslRlVecEnvWrapper": BaseWrapper})
        term = types.SimpleNamespace(_joint_names=self.names, _joint_ids=[4, 0, 2],
                                     _scale=torch.ones(4, 3), _offset=torch.zeros(4, 3), _clip=None)
        terminations = types.SimpleNamespace(compute=lambda: torch.zeros(4, dtype=torch.bool))

        class Scene(dict):
            sensors = {}

        env = types.SimpleNamespace(action_manager=types.SimpleNamespace(get_term=lambda _: term),
                                    scene=Scene(robot=self.robot), termination_manager=terminations)
        modules = {TRACKING_MODULE: self.tracking, JOINT_SPEED_MODULE: self.speed}
        saved = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)
        try:
            wrapper = diagnostics["DiagnosticVecEnvWrapper"](env, clip_actions=None)
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        self.data.joint_vel[0, 4] = 12.0
        wrapper.step(torch.zeros(4, 3))
        self.data.joint_vel[1, 0] = 9.0
        wrapper.step(torch.zeros(4, 3))
        log = wrapper.take_action_diagnostics()
        self.assertAlmostEqual(log["Policy/joint_vel_excess_any_gt8_fraction"], 2 / 8)
        self.assertAlmostEqual(log["Policy/joint_vel_excess_any_gt10_fraction"], 1 / 8)
        self.assertAlmostEqual(log["Policy/joint_vel_excess_gt10_fraction/a"], 1 / 8)
        self.assertAlmostEqual(log["Policy/joint_vel_excess_gt8_fraction/b"], 1 / 8)
        self.assertAlmostEqual(log["Policy/joint_vel_excess_peak_rad_s_max"], 12.0, places=5)
        self.assertIn("Policy/sent_tracking_any_gt25_fraction", log)


if __name__ == "__main__":
    unittest.main()
