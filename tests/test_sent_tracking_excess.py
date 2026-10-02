import ast
import copy
import importlib.util
import math
import sys
import tempfile
import types
import unittest
from collections.abc import Iterable, Mapping, Sequence, Sized
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
REWARDS = TASK / "mdp/rewards.py"
TRACKING = TASK / "mdp/tracking.py"
ACTIONS = TASK / "mdp/actions.py"
ENV_CFG = TASK / "robonex_walking_v2_env_cfg.py"
EFFECTS = ROOT / "scripts/rsl_rl/deploy_effects.py"
EVAL = ROOT / "scripts/rsl_rl/eval_commands.py"
TRAIN = ROOT / "scripts/rsl_rl/train.py"
ISAAC = Path("/home/polygon/IsaacLab/source/isaaclab/isaaclab")
W102_RUN = Path(
    "/home/polygon/humanoid_project/robonex-walking/logs/rsl_rl/robonex_walking_v2_edu/"
    "2026-10-02_10-26-01_W102s44_v2_edu_jointdelay_s44"
)
TRACKING_MODULE = "robonex_walking.tasks.manager_based.robonex_walking.mdp.tracking"

LEG_JOINTS = (
    "l_hip_yaw_joint", "l_hip_pitch_joint", "l_hip_roll_joint", "l_knee_pitch_joint",
    "l_ankle_upper_joint", "l_ankle_lower_joint",
    "r_hip_yaw_joint", "r_hip_pitch_joint", "r_hip_roll_joint", "r_knee_pitch_joint",
    "r_ankle_upper_joint", "r_ankle_lower_joint",
)
W102_CLIP = (
    (-0.827758, 0.827758), (-1.648063, 1.648063), (-2.084395, 0.164533), (-1.21173, 0.164533),
    (-0.269253, 0.862665), (-0.862665, 0.513599), (-0.827758, 0.827758), (-1.648063, 1.648063),
    (-0.164533, 2.084395), (-0.164533, 1.21173), (-0.862665, 0.269253), (-0.513599, 0.862665),
)
WEIGHT = -20.0


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
    tracking = load_module("tracking_under_test", TRACKING)
    ns = {"torch": torch, "math": math, "sent_tracking_error": tracking.sent_tracking_error,
          "SceneEntityCfg": scene_entity, "Articulation": object, "ManagerBasedRLEnv": object}
    return load_nodes(REWARDS, {"sent_tracking_excess_l2"}, ns)["sent_tracking_excess_l2"]


def make_env(target, position):
    data = types.SimpleNamespace(joint_pos_target=torch.as_tensor(target, dtype=torch.float32),
                                 joint_pos=torch.as_tensor(position, dtype=torch.float32))
    return types.SimpleNamespace(scene={"robot": types.SimpleNamespace(data=data)})


def one_joint(error_deg, n=12, index=0):
    target = [0.0] * n
    target[index] = math.radians(error_deg)
    return [target], [[0.0] * n]


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
        func="robonex_walking.tasks.manager_based.robonex_walking.mdp.rewards:sent_tracking_excess_l2",
        weight=weight,
        params={"floor_deg": 20.0, "asset_cfg": {"name": "robot", "joint_names": list(LEG_JOINTS)}},
    )


class KernelTest(unittest.TestCase):
    def setUp(self):
        self.fn = load_reward()
        self.ids = scene_entity("robot", list(range(12)))

    def rate(self, error_deg, index=0):
        return WEIGHT * self.fn(make_env(*one_joint(error_deg, index=index)), asset_cfg=self.ids).item()

    def test_rates_at_the_design_points(self):
        for error, expected in ((0.0, 0.0), (20.0, 0.0), (25.0, -0.15231), (30.0, -0.60923)):
            self.assertAlmostEqual(self.rate(error), expected, places=4, msg=error)
        self.assertAlmostEqual(self.rate(25.0), WEIGHT * math.radians(5.0) ** 2, places=5)
        self.assertAlmostEqual(self.rate(30.0), WEIGHT * math.radians(10.0) ** 2, places=5)
        self.assertEqual(self.rate(19.9), 0.0)

    def test_floor_parameter(self):
        env = make_env(*one_joint(25.0))
        self.assertAlmostEqual(self.fn(env, floor_deg=15.0, asset_cfg=self.ids).item(),
                               math.radians(10.0) ** 2, places=6)

    def test_sign_symmetry_and_every_joint_alike(self):
        for index in range(12):
            self.assertAlmostEqual(self.rate(30.0, index), self.rate(-30.0, index), places=6)
            self.assertAlmostEqual(self.rate(30.0, index), self.rate(30.0, 0), places=6)

    def test_error_is_target_minus_position_wrapped_to_pi(self):
        cases = (
            (2.0 * math.pi - math.radians(30.0), 0.0, 30.0),
            (math.radians(10.0), 2.0 * math.pi + math.radians(40.0), 30.0),
            (math.pi, 0.0, 180.0),
            (-math.pi, 0.0, 180.0),
            (0.0, math.pi, 180.0),
            (math.pi + math.radians(10.0), 0.0, 170.0),
            (-math.pi - math.radians(10.0), 0.0, 170.0),
            (4.0 * math.pi + math.radians(25.0), 0.0, 25.0),
        )
        for target, position, error_deg in cases:
            out = self.fn(make_env([[target] + [0.0] * 11], [[position] + [0.0] * 11]), asset_cfg=self.ids)
            self.assertAlmostEqual(out.item(), math.radians(error_deg - 20.0) ** 2, places=4,
                                   msg=(target, position))

    def test_sum_over_joints_not_mean(self):
        target = [[math.radians(30.0)] * 2 + [0.0] * 10, [math.radians(30.0)] * 12]
        out = self.fn(make_env(target, [[0.0] * 12] * 2), asset_cfg=self.ids)
        single = math.radians(10.0) ** 2
        self.assertAlmostEqual(out[0].item(), 2.0 * single, places=6)
        self.assertAlmostEqual(out[1].item(), 12.0 * single, places=5)
        self.assertEqual(out.shape, (2,))

    def test_selected_ids_permuted_noncontiguous_and_passive_excluded(self):
        n = 16
        target = [0.0] * n
        position = [0.0] * n
        actuated = [15, 2, 9, 0, 11, 4, 13, 6, 1, 8, 3, 10]
        passive = sorted(set(range(n)) - set(actuated))
        for i in passive:
            target[i] = math.radians(90.0)
        target[9] = math.radians(30.0)
        target[15] = math.radians(-25.0)
        env = make_env([target], [position])
        expected = math.radians(10.0) ** 2 + math.radians(5.0) ** 2
        for ids in (actuated, list(reversed(actuated)), sorted(actuated)):
            self.assertAlmostEqual(self.fn(env, asset_cfg=scene_entity("robot", ids)).item(), expected, places=6)
        everything = self.fn(env, asset_cfg=scene_entity("robot", slice(None))).item()
        self.assertGreater(everything, expected + 1.0)

    def test_nonfinite_counts_as_the_largest_wrapped_error_and_stays_finite(self):
        cap = (math.pi - math.radians(20.0)) ** 2
        rows_target = [
            [math.nan] + [0.0] * 11,
            [math.inf] + [0.0] * 11,
            [0.0] * 12,
            [-math.inf] + [0.0] * 11,
            [math.nan] * 12,
        ]
        rows_position = [
            [0.0] * 12,
            [0.0] * 12,
            [math.nan] + [0.0] * 11,
            [-math.inf] + [0.0] * 11,
            [0.0] * 12,
        ]
        out = self.fn(make_env(rows_target, rows_position), asset_cfg=self.ids)
        self.assertTrue(bool(torch.isfinite(out).all()))
        for row in range(4):
            self.assertAlmostEqual(out[row].item(), cap, places=4)
        self.assertAlmostEqual(out[4].item(), 12.0 * cap, places=3)
        self.assertTrue(bool((out <= 12.0 * cap + 1e-3).all()))


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(ENV_CFG.read_text())
        self.consts = {n.targets[0].id: n.value for n in self.tree.body
                       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        self.rewards = next(n for n in self.tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg")

    def test_term_present_inert_and_on_the_actuated_joints(self):
        self.assertEqual(ast.literal_eval(self.consts["SENT_TRACKING_EXCESS_WEIGHT"]), 0.0)
        self.assertEqual(ast.literal_eval(self.consts["SENT_TRACKING_EXCESS_FLOOR_DEG"]), 20.0)
        term = next(n for n in self.rewards.body if isinstance(n, ast.Assign) and n.targets[0].id == "sent_tracking_excess")
        kw = {k.arg: k.value for k in term.value.keywords}
        self.assertEqual(kw["weight"].id, "SENT_TRACKING_EXCESS_WEIGHT")
        self.assertEqual(kw["func"].attr, "sent_tracking_excess_l2")
        params = {ast.literal_eval(k): v for k, v in zip(kw["params"].keys, kw["params"].values)}
        self.assertEqual(params["floor_deg"].id, "SENT_TRACKING_EXCESS_FLOOR_DEG")
        asset = params["asset_cfg"]
        self.assertEqual(ast.literal_eval(asset.args[0]), "robot")
        asset_kw = {k.arg: k.value for k in asset.keywords}
        self.assertEqual(asset_kw["joint_names"].id, "LEG_JOINTS")
        self.assertTrue(ast.literal_eval(asset_kw["preserve_order"]))

    def test_new_term_is_last_so_existing_term_order_is_unchanged(self):
        names = [n.targets[0].id for n in self.rewards.body if isinstance(n, ast.Assign)]
        self.assertEqual(names[-1], "sent_tracking_excess")
        self.assertEqual(names.count("sent_tracking_excess"), 1)

    def test_reward_manager_skips_a_zero_weight_term(self):
        ns = {"torch": torch}
        tree = ast.parse((ISAAC / "managers/reward_manager.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardManager")
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute")
        exec(compile(ast.Module(body=[node], type_ignores=[]), "reward_manager", "exec"), ns)

        def explode(env, **kwargs):
            raise AssertionError("a zero-weight term was evaluated")

        alive = types.SimpleNamespace(func=lambda env: torch.ones(3), weight=0.5, params={})
        tracking = types.SimpleNamespace(func=explode, weight=0.0, params={"floor_deg": 20.0})
        manager = types.SimpleNamespace(
            _term_names=["alive", "sent_tracking_excess"], _term_cfgs=[alive, tracking], _env=None,
            _reward_buf=torch.zeros(3), _step_reward=torch.full((3, 2), 7.0),
            _episode_sums={"alive": torch.zeros(3), "sent_tracking_excess": torch.zeros(3)},
        )
        out = ns["compute"](manager, dt=0.02)
        torch.testing.assert_close(out, torch.full((3,), 0.01))
        self.assertTrue(bool((manager._episode_sums["sent_tracking_excess"] == 0.0).all()))
        self.assertTrue(bool((manager._step_reward[:, 1] == 0.0).all()))


class OverrideRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def cfg(self, weight=0.0):
        return types.SimpleNamespace(
            rewards=types.SimpleNamespace(
                joint_load=types.SimpleNamespace(func="rewards:joint_load_rating_l2", weight=0.0,
                                                 params={"max_ratio": 5.0}),
                sent_tracking_excess=new_term(weight),
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
            "joint_load": {"func": "rewards:joint_load_rating_l2", "weight": 0.0, "params": {"max_ratio": 5.0}},
            "sent_tracking_excess": {"func": live.rewards.sent_tracking_excess.func, "weight": 0.0,
                                     "params": copy.deepcopy(live.rewards.sent_tracking_excess.params)},
        }}}
        ConfigStore.instance().store(name="w104_tracking_override_test", node=node)
        GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path=None):
            composed = compose(config_name="w104_tracking_override_test",
                               overrides=["env.rewards.sent_tracking_excess.weight=-20.0"])
        GlobalHydra.instance().clear()
        overridden = OmegaConf.to_container(composed, resolve=True)["env"]
        with FakeIsaacModules() as update:
            update(live, overridden)
            self.assertEqual(live.rewards.sent_tracking_excess.weight, -20.0)
            self.assertIsInstance(live.rewards.sent_tracking_excess.weight, float)
            saved = {"rewards": {name: {"func": term.func, "weight": term.weight,
                                        "params": term.params}
                                 for name, term in vars(live.rewards).items()}}
            run = Path(self.tmp.name)
            (run / "params").mkdir()
            (run / "params" / "env.yaml").write_text(yaml.dump(saved))
            restored = self.cfg()
            load_module("deploy_effects_tracking_roundtrip", EFFECTS).apply_training_env_cfg(
                restored, str(run / "model_999.pt"))
        term = restored.rewards.sent_tracking_excess
        self.assertEqual(term.weight, -20.0)
        self.assertEqual(term.params["floor_deg"], 20.0)
        self.assertEqual(term.params["asset_cfg"]["joint_names"], list(LEG_JOINTS))


class OldRunRestoreTest(unittest.TestCase):
    def test_w102_env_yaml_restores_with_the_new_term_absent(self):
        path = W102_RUN / "params" / "env.yaml"
        if not path.is_file():
            self.skipTest("W102 run directory not found in the main worktree")
        effects = load_module("deploy_effects_tracking_old_run", EFFECTS)
        with path.open() as handle:
            saved = yaml.load(handle, Loader=effects._SavedCfgLoader)
        self.assertNotIn("sent_tracking_excess", saved["rewards"])
        live = to_cfg({k: v for k, v in saved.items() if k not in ("viewer", "log_dir", "seed")})
        live.rewards.sent_tracking_excess = new_term()
        with FakeIsaacModules():
            restored_from = effects.apply_training_env_cfg(live, str(W102_RUN / "model_999.pt"))
        self.assertEqual(restored_from, path.resolve())
        self.assertIsNone(live.rewards.sent_tracking_excess)
        present = sorted(name for name, term in vars(live.rewards).items() if term is not None)
        self.assertEqual(present, sorted(saved["rewards"]))
        for name in present:
            self.assertEqual(getattr(live.rewards, name).weight, saved["rewards"][name]["weight"], name)
        self.assertTrue(live.actions.joint_pos.slew_enabled)


class EffectiveActionTest(unittest.TestCase):
    def setUp(self):
        self.effects = load_module("deploy_effects_tracking_action", EFFECTS)

    def term(self, slew_enabled=True, stage=True, roll=math.radians(12.0)):
        cfg = types.SimpleNamespace(slew_enabled=slew_enabled, max_speed=6.0, max_accel=120.0, foot_roll_limit=roll)
        term = type("Ver2JointPositionAction", (), {})()
        term.cfg = cfg
        term._roll_pairs = [(4, 5, 1.0, (0, 1), (0, 1))]
        if stage:
            term._slew_position = torch.zeros(1, 12)
        return term

    def test_trained_slew_stage_is_reported_on_without_the_cli_flag(self):
        record = self.effects.effective_action_config(self.term(), cli_slew_limit=False)
        self.assertEqual(record, {
            "action_class": "Ver2JointPositionAction", "slew_enabled": True, "slew_source": "action_term",
            "slew_max_speed": 6.0, "slew_max_accel": 120.0, "foot_roll_limit": math.radians(12.0),
            "foot_roll_projection": True, "cli_slew_limit": False,
        })

    def test_disabled_stage_and_eval_limiter(self):
        off = self.effects.effective_action_config(self.term(slew_enabled=False, roll=0.0))
        self.assertEqual((off["slew_enabled"], off["slew_source"], off["slew_max_speed"]), (False, None, None))
        self.assertFalse(off["foot_roll_projection"])
        plain = self.term(stage=False)
        plain.cfg = types.SimpleNamespace()
        extra = self.effects.effective_action_config(plain, cli_slew_limit=True)
        self.assertEqual((extra["slew_enabled"], extra["slew_source"], extra["slew_max_speed"], extra["slew_max_accel"]),
                         (True, "eval_limiter", 6.0, 120.0))
        self.assertIsNone(extra["foot_roll_limit"])

    def test_eval_commands_records_it_after_the_limiter_and_uses_the_wrapped_error(self):
        tree = ast.parse(EVAL.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        calls = {}
        for node in ast.walk(main):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.setdefault(node.func.id, []).append(node)
        record = calls["effective_action_config"][0]
        self.assertEqual([ast.unparse(a) for a in record.args], ["action_term", "args_cli.slew_limit"])
        self.assertLess(calls["install_slew_limiter"][0].lineno, record.lineno)
        error = next(n for n in ast.walk(main) if isinstance(n, ast.Assign)
                     and getattr(n.targets[0], "id", None) == "error")
        self.assertIn("sent_tracking_error(robot.data.joint_pos_target[:, joint_ids], q)", ast.unparse(error.value))
        text = EVAL.read_text()
        for field in ('"slew_limit": bool(args_cli.slew_limit)', '"tracking_error": error > max_error_rad',
                      '"tracking_error_deg": peak_error.item()', '"action_effective":', '"tracking_error_kernel":'):
            self.assertIn(field, text)


class FakeJointPositionAction:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env
        self._asset = env.scene["robot"]
        self._joint_ids = env.joint_ids
        self._joint_names = list(LEG_JOINTS)
        self.num_envs = env.num_envs
        self.device = "cpu"
        self._scale = torch.full((env.num_envs, 12), 0.25)
        self._offset = torch.zeros(env.num_envs, 12)
        self._clip = torch.tensor(W102_CLIP).unsqueeze(0).repeat(env.num_envs, 1, 1)
        self._raw_actions = torch.zeros(env.num_envs, 12)
        self._processed_actions = torch.zeros(env.num_envs, 12)

    @property
    def processed_actions(self):
        return self._processed_actions

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale + self._offset
        self._processed_actions = torch.clamp(self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

    def apply_actions(self):
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        self._raw_actions[env_ids] = 0.0


class FakeAsset:
    def __init__(self, num_envs, num_joints):
        self.data = types.SimpleNamespace(
            joint_pos=torch.zeros(num_envs, num_joints),
            joint_pos_target=torch.zeros(num_envs, num_joints),
            default_joint_pos=torch.zeros(num_envs, num_joints),
        )

    def set_joint_position_target(self, target, joint_ids):
        self.data.joint_pos_target[:, joint_ids] = target


class StepOrderTest(unittest.TestCase):
    def test_reward_sees_the_final_held_command_against_fresh_position(self):
        ns = {"torch": torch, "JointPositionAction": FakeJointPositionAction, "ManagerBasedEnv": object,
              "Sequence": Sequence, "SlewLimitedJointPositionActionCfg": object, "Ver2JointPositionActionCfg": object}
        load_nodes(ACTIONS, {"slew_limit_step", "foot_roll", "clip_foot_roll",
                             "SlewLimitedJointPositionAction", "Ver2JointPositionAction"}, ns)
        num_envs, num_joints = 64, 16
        joint_ids = [15, 2, 9, 0, 11, 4, 13, 6, 1, 8, 3, 10]
        asset = FakeAsset(num_envs, num_joints)
        env = types.SimpleNamespace(scene={"robot": asset}, joint_ids=joint_ids, num_envs=num_envs, step_dt=0.02)
        cfg = types.SimpleNamespace(
            max_speed=6.0, max_accel=120.0, slew_enabled=True, foot_roll_limit=math.radians(12.0),
            foot_roll_coeffs=(-0.001195, 0.498159, 0.491024, -0.105182, -0.031960, 0.006356),
            foot_roll_pairs=(("l_ankle_upper_joint", "l_ankle_lower_joint", 1.0),
                             ("r_ankle_upper_joint", "r_ankle_lower_joint", -1.0)),
        )
        term = ns["Ver2JointPositionAction"](cfg, env)
        reward = load_reward()
        generator = torch.Generator().manual_seed(3)
        decimation = 8
        for step in range(40):
            actions = 6.0 * torch.randn(num_envs, 12, generator=generator)
            requested = torch.clamp(actions * 0.25, torch.tensor(W102_CLIP)[:, 0], torch.tensor(W102_CLIP)[:, 1])
            term.process_actions(actions)
            held = term.processed_actions.clone()
            for _ in range(decimation):
                term.apply_actions()
                torch.testing.assert_close(asset.data.joint_pos_target[:, joint_ids], held, rtol=0, atol=0)
                q = asset.data.joint_pos[:, joint_ids]
                asset.data.joint_pos[:, joint_ids] = q + 0.2 * (held - q)
            q = asset.data.joint_pos[:, joint_ids].clone()
            torch.testing.assert_close(held, term._slew_position, rtol=0, atol=0)
            error = torch.remainder(held - q + math.pi, 2.0 * math.pi) - math.pi
            expected = torch.square(torch.clamp(error.abs() - math.radians(20.0), min=0.0)).sum(dim=1)
            out = reward(env, asset_cfg=scene_entity("robot", joint_ids))
            torch.testing.assert_close(out, expected)
            if step == 0:
                self.assertGreater(float((requested - held).abs().max()), 0.05)


class SentTrackingMetricsTest(unittest.TestCase):
    def setUp(self):
        self.tracking = load_module("tracking_metrics_under_test", TRACKING)
        self.names = ["a", "b", "c"]
        self.data = types.SimpleNamespace(joint_pos=torch.zeros(4, 5), joint_pos_target=torch.zeros(4, 5))
        self.robot = types.SimpleNamespace(data=self.data)

    def test_fractions_peaks_and_keys(self):
        metrics = self.tracking.SentTrackingMetrics(self.robot, [4, 0, 2], self.names)
        self.data.joint_pos_target[:, 1] = math.radians(90.0)
        self.data.joint_pos_target[0, 4] = math.radians(22.0)
        self.data.joint_pos_target[1, 0] = -math.radians(26.0)
        self.data.joint_pos_target[2, 2] = math.radians(10.0)
        metrics.sample()
        self.data.joint_pos_target.zero_()
        self.data.joint_pos[3, 2] = math.nan
        metrics.sample()
        log = metrics.take_log()
        self.assertAlmostEqual(log["Policy/sent_tracking_any_gt20_fraction"], 3 / 8)
        self.assertAlmostEqual(log["Policy/sent_tracking_any_gt25_fraction"], 2 / 8)
        self.assertAlmostEqual(log["Policy/sent_tracking_gt20_fraction/a"], 1 / 8)
        self.assertAlmostEqual(log["Policy/sent_tracking_gt25_fraction/a"], 0.0)
        self.assertAlmostEqual(log["Policy/sent_tracking_gt25_fraction/b"], 1 / 8)
        self.assertAlmostEqual(log["Policy/sent_tracking_gt25_fraction/c"], 1 / 8)
        self.assertAlmostEqual(log["Policy/sent_tracking_peak_deg_max"], 180.0, places=3)
        self.assertAlmostEqual(log["Policy/sent_tracking_peak_deg_mean"], (22.0 + 26.0 + 10.0 + 180.0) / 8, places=3)
        self.assertAlmostEqual(log["Policy/sent_tracking_nonfinite_fraction"], 1 / 24)
        self.assertEqual(metrics.take_log()["Policy/sent_tracking_any_gt20_fraction"], 0.0)

    def test_training_wrapper_samples_at_the_pre_reset_boundary(self):
        class BaseWrapper:
            def __init__(self, env, clip_actions=None):
                self.unwrapped = env
                self.device = "cpu"
                self.num_envs = 4
                self.num_actions = 3
                self.clip_actions = clip_actions

            def step(self, actions):
                env = self.unwrapped
                env.termination_manager.compute()
                self.data.joint_pos_target.zero_()
                return actions

        BaseWrapper.data = self.data
        diagnostics = load_nodes(TRAIN, {"DiagnosticVecEnvWrapper"},
                                 {"torch": torch, "math": math, "RslRlVecEnvWrapper": BaseWrapper})
        term = types.SimpleNamespace(_joint_names=self.names, _joint_ids=[4, 0, 2],
                                     _scale=torch.ones(4, 3), _offset=torch.zeros(4, 3), _clip=None)
        compute_calls = []
        terminations = types.SimpleNamespace(
            compute=lambda: compute_calls.append(1) or torch.zeros(4, dtype=torch.bool))

        class Scene(dict):
            sensors = {}

        env = types.SimpleNamespace(action_manager=types.SimpleNamespace(get_term=lambda _: term),
                                    scene=Scene(robot=self.robot), termination_manager=terminations)
        saved = sys.modules.get(TRACKING_MODULE)
        sys.modules[TRACKING_MODULE] = self.tracking
        try:
            wrapper = diagnostics["DiagnosticVecEnvWrapper"](env, clip_actions=None)
        finally:
            if saved is None:
                sys.modules.pop(TRACKING_MODULE, None)
            else:
                sys.modules[TRACKING_MODULE] = saved
        self.data.joint_pos_target[0, 4] = math.radians(30.0)
        result = env.termination_manager.compute()
        self.assertEqual(result.shape, (4,))
        self.assertEqual(len(compute_calls), 1)
        self.data.joint_pos_target[0, 4] = math.radians(30.0)
        wrapper.step(torch.zeros(4, 3))
        self.assertEqual(len(compute_calls), 2)
        log = wrapper.take_action_diagnostics()
        self.assertAlmostEqual(log["Policy/sent_tracking_any_gt25_fraction"], 2 / 8)
        self.assertAlmostEqual(log["Policy/sent_tracking_gt25_fraction/a"], 2 / 8)

    def test_wrapper_without_termination_manager_has_no_tracking_metrics(self):
        source = TRAIN.read_text()
        self.assertIn("if scene is not None and terminations is not None:", source)


if __name__ == "__main__":
    unittest.main()
