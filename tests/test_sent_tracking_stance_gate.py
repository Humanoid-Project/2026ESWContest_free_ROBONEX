import ast
import copy
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
import yaml

from test_sent_tracking_excess import (
    EFFECTS,
    ENV_CFG,
    JOINT_SPEED,
    JOINT_SPEED_MODULE,
    LEG_JOINTS,
    REWARDS,
    TRACKING,
    TRACKING_MODULE,
    TRAIN,
    FakeIsaacModules,
    load_module,
    load_nodes,
    to_cfg,
)

W104_RUN = Path(
    str(Path(__file__).resolve().parents[1]) + "/logs/rsl_rl/robonex_walking_v2_edu/"
    "2026-10-03_11-56-34_W104s44a_v2_edu_trackexc200_s44"
)
FUNC = "robonex_walking.tasks.manager_based.robonex_walking.mdp.rewards:sent_tracking_excess_l2"
ORIGINAL = '''
def original(env, floor_deg=20.0, asset_cfg=None):
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    error = sent_tracking_error(asset.data.joint_pos_target[:, joint_ids], asset.data.joint_pos[:, joint_ids])
    excess = torch.clamp(error - math.radians(floor_deg), min=0.0)
    return torch.sum(torch.square(excess), dim=1)
'''
FEET = ("l_foot", "r_foot")
HISTORY = 8


class entity(types.SimpleNamespace):
    def __init__(self, name, joint_ids=slice(None), body_ids=slice(None)):
        super().__init__(name=name, joint_ids=joint_ids, body_ids=body_ids)


def load_kernel():
    tracking = load_module("tracking_gate_under_test", TRACKING)
    ns = {"torch": torch, "math": math, "sent_tracking_error": tracking.sent_tracking_error,
          "SentTrackingGateStats": tracking.SentTrackingGateStats, "SceneEntityCfg": entity,
          "Articulation": object, "ContactSensor": object, "ManagerBasedRLEnv": object}
    load_nodes(REWARDS, {"_contact_mask", "_selected", "_leg_side", "_stance_gate_layout",
                         "sent_tracking_excess_l2"}, ns)
    exec(ORIGINAL, ns)
    return ns


class Scene(dict):
    pass


def make_env(target, position, forces, joint_names=LEG_JOINTS, feet=FEET):
    target = torch.as_tensor(target, dtype=torch.float32)
    position = torch.as_tensor(position, dtype=torch.float32)
    robot = types.SimpleNamespace(joint_names=list(joint_names),
                                  data=types.SimpleNamespace(joint_pos_target=target, joint_pos=position))
    forces = torch.as_tensor(forces, dtype=torch.float32)
    if forces.dim() == 2:
        history = torch.zeros(forces.shape[0], HISTORY, forces.shape[1], 3)
        history[:, :, :, 2] = forces.unsqueeze(1)
        forces = history
    sensor = types.SimpleNamespace(body_names=list(feet), data=types.SimpleNamespace(net_forces_w_history=forces))
    scene = Scene(robot=robot)
    scene.sensors = {"contact_forces": sensor}
    return types.SimpleNamespace(scene=scene, device="cpu", num_envs=target.shape[0])


def random_errors(rows, n=12, seed=0, scale=0.9):
    generator = torch.Generator().manual_seed(seed)
    return (scale * torch.randn(rows, n, generator=generator)).tolist(), [[0.0] * n] * rows


class GateKernelTest(unittest.TestCase):
    def setUp(self):
        self.ns = load_kernel()
        self.fn = self.ns["sent_tracking_excess_l2"]
        self.asset = entity("robot", list(range(12)))
        self.sensor = entity("contact_forces", body_ids=[0, 1])

    def gated(self, env, **kw):
        return self.fn(env, asset_cfg=self.asset, stance_gate=True, sensor_cfg=self.sensor, **kw)

    def test_gate_off_is_bit_identical_to_the_original_term(self):
        target, position = random_errors(256)
        target[3][0] = math.nan
        target[4][7] = math.inf
        for forces in ([[0.0, 0.0]] * 256, [[500.0, 500.0]] * 256, [[0.0, 500.0]] * 256):
            env = make_env(target, position, forces)
            reference = self.ns["original"](env, asset_cfg=self.asset)
            for kwargs in ({}, {"stance_gate": False}, {"stance_gate": False, "sensor_cfg": self.sensor,
                                                         "contact_threshold_n": 1e9}):
                out = self.fn(env, asset_cfg=self.asset, **kwargs)
                self.assertTrue(torch.equal(out, reference), kwargs)
            self.assertFalse(hasattr(env, "sent_tracking_gate_stats"))
            self.assertFalse(hasattr(env, "_sent_tracking_gate_layout"))
            for floor in (10.0, 30.0):
                self.assertTrue(torch.equal(self.fn(env, floor_deg=floor, asset_cfg=self.asset),
                                            self.ns["original"](env, floor_deg=floor, asset_cfg=self.asset)))

    def test_both_feet_loaded_equals_ungated(self):
        target, position = random_errors(128, seed=1)
        env = make_env(target, position, [[300.0, 25.0]] * 128)
        self.assertTrue(torch.equal(self.gated(env), self.fn(env, asset_cfg=self.asset)))

    def test_unloaded_foot_zeroes_its_own_leg_only(self):
        target = [[math.radians(30.0)] * 6 + [math.radians(-40.0)] * 6] * 4
        position = [[0.0] * 12] * 4
        forces = [[0.0, 300.0], [300.0, 0.0], [0.0, 0.0], [300.0, 300.0]]
        out = self.gated(make_env(target, position, forces))
        left, right = 6 * math.radians(10.0) ** 2, 6 * math.radians(20.0) ** 2
        for row, expected in enumerate((right, left, 0.0, left + right)):
            self.assertAlmostEqual(out[row].item(), expected, places=5, msg=row)
        self.assertEqual(out[2].item(), 0.0)

    def test_every_single_joint_follows_its_foot(self):
        for index in range(12):
            target = [[0.0] * 12 for _ in range(2)]
            target[0][index] = target[1][index] = math.radians(30.0)
            left_only = [[300.0, 0.0], [0.0, 300.0]]
            out = self.gated(make_env(target, [[0.0] * 12] * 2, left_only))
            on_left = LEG_JOINTS[index].startswith("l_")
            self.assertEqual(out[0].item() > 0.0, on_left, LEG_JOINTS[index])
            self.assertEqual(out[1].item() > 0.0, not on_left, LEG_JOINTS[index])

    def test_threshold_is_strict_and_a_parameter(self):
        target = [[math.radians(30.0)] * 12] * 3
        forces = [[20.0, 20.0], [20.01, 19.99], [21.0, 21.0]]
        env = make_env(target, [[0.0] * 12] * 3, forces)
        single = math.radians(10.0) ** 2
        out = self.gated(env)
        self.assertEqual(out[0].item(), 0.0)
        self.assertAlmostEqual(out[1].item(), 6 * single, places=5)
        self.assertAlmostEqual(out[2].item(), 12 * single, places=5)
        raised = self.gated(env, contact_threshold_n=20.5)
        self.assertEqual(raised[:2].tolist(), [0.0, 0.0])
        self.assertAlmostEqual(raised[2].item(), 12 * single, places=5)

    def test_force_norm_not_vertical_component(self):
        history = torch.zeros(1, HISTORY, 2, 3)
        history[0, 0, 0] = torch.tensor([12.0, 12.0, 12.0])
        history[0, 0, 1] = torch.tensor([0.0, 0.0, 19.0])
        target = [[math.radians(30.0)] * 12]
        out = self.gated(make_env(target, [[0.0] * 12], history))
        self.assertAlmostEqual(out.item(), 6 * math.radians(10.0) ** 2, places=5)

    def test_load_in_any_substep_of_the_step_counts(self):
        target = [[math.radians(30.0)] * 12] * 3
        history = torch.zeros(3, HISTORY, 2, 3)
        history[0, 0, :, 2] = 300.0
        history[1, HISTORY - 1, :, 2] = 300.0
        history[2, HISTORY // 2, 0, 2] = 300.0
        out = self.gated(make_env(target, [[0.0] * 12] * 3, history))
        single = math.radians(10.0) ** 2
        self.assertAlmostEqual(out[0].item(), 12 * single, places=5)
        self.assertAlmostEqual(out[1].item(), 12 * single, places=5)
        self.assertAlmostEqual(out[2].item(), 6 * single, places=5)

    def test_hang_both_feet_unloaded_gives_exactly_zero_even_for_nonfinite_errors(self):
        target = [[math.radians(170.0)] * 12, [math.nan] * 12, [math.inf] + [0.0] * 11]
        position = [[0.0] * 12, [0.0] * 12, [math.nan] * 12]
        history = torch.zeros(3, HISTORY, 2, 3)
        history[:, :, :, 2] = 19.99
        out = self.gated(make_env(target, position, history))
        self.assertTrue(torch.equal(out, torch.zeros(3)))

    def test_nonfinite_force_counts_as_loaded(self):
        target = [[math.radians(30.0)] * 12] * 2
        history = torch.zeros(2, HISTORY, 2, 3)
        history[0, 3, 0, 2] = math.nan
        history[1, 3, 1, 0] = -math.inf
        out = self.gated(make_env(target, [[0.0] * 12] * 2, history))
        self.assertAlmostEqual(out[0].item(), 6 * math.radians(10.0) ** 2, places=5)
        self.assertAlmostEqual(out[1].item(), 6 * math.radians(10.0) ** 2, places=5)

    def test_leg_mapping_from_names_with_permuted_joints_and_feet(self):
        names = ["base_spring", "r_knee_pitch_joint", "l_hip_yaw_joint", "r_hip_yaw_joint", "l_knee_coupler_joint_a",
                 "l_ankle_lower_joint", "r_hip_roll_joint", "l_hip_pitch_joint", "r_ankle_upper_joint",
                 "l_hip_roll_joint", "r_hip_pitch_joint", "l_knee_pitch_joint", "r_ankle_lower_joint",
                 "l_ankle_upper_joint", "r_knee_coupler_joint_b"]
        ids = [names.index(j) for j in reversed(LEG_JOINTS)]
        target = [[math.radians(30.0)] * len(names)]
        sensor = entity("contact_forces", body_ids=[2, 0])
        feet = ("r_foot", "torso", "l_foot")
        history = torch.zeros(1, HISTORY, 3, 3)
        history[0, :, 2, 2] = 300.0
        history[0, :, 1, 2] = 300.0
        env = make_env(target, [[0.0] * len(names)], history, joint_names=names, feet=feet)
        out = self.fn(env, asset_cfg=entity("robot", ids), stance_gate=True, sensor_cfg=sensor)
        self.assertAlmostEqual(out.item(), 6 * math.radians(10.0) ** 2, places=5)
        target = [[0.0] * len(names)]
        target[0][names.index("r_ankle_upper_joint")] = math.radians(30.0)
        env = make_env(target, [[0.0] * len(names)], history, joint_names=names, feet=feet)
        out = self.fn(env, asset_cfg=entity("robot", ids), stance_gate=True, sensor_cfg=sensor)
        self.assertEqual(out.item(), 0.0)
        history[0, :, 0, 2] = 300.0
        env = make_env(target, [[0.0] * len(names)], history, joint_names=names, feet=feet)
        out = self.fn(env, asset_cfg=entity("robot", ids), stance_gate=True, sensor_cfg=sensor)
        self.assertAlmostEqual(out.item(), math.radians(10.0) ** 2, places=5)
        self.assertEqual(env._sent_tracking_gate_layout[1].tolist(), [0 if j.startswith("l_") else 1
                                                                       for j in reversed(LEG_JOINTS)])

    def test_layout_is_cached_per_env_and_rebuilt_when_ids_change(self):
        target, position = random_errors(4, seed=2)
        env = make_env(target, position, [[300.0, 0.0]] * 4)
        self.gated(env)
        cached = env._sent_tracking_gate_layout
        self.gated(env)
        self.assertIs(env._sent_tracking_gate_layout, cached)
        env.scene["robot"].joint_names = list(reversed(LEG_JOINTS))
        self.fn(env, asset_cfg=entity("robot", list(range(11, -1, -1))), stance_gate=True, sensor_cfg=self.sensor)
        self.assertIsNot(env._sent_tracking_gate_layout, cached)

    def test_bad_names_and_missing_sensor_fail_loudly(self):
        target, position = random_errors(1)
        with self.assertRaises(ValueError):
            self.fn(make_env(target, position, [[0.0, 0.0]]), asset_cfg=self.asset, stance_gate=True)
        names = list(LEG_JOINTS)
        names[3] = "knee_pitch_joint"
        with self.assertRaises(ValueError):
            self.gated(make_env(target, position, [[0.0, 0.0]], joint_names=names))
        with self.assertRaises(ValueError):
            self.gated(make_env(target, position, [[0.0, 0.0]], feet=("l_foot", "l_toe")))
        with self.assertRaises(ValueError):
            self.fn(make_env(target, position, [[0.0, 0.0]]), asset_cfg=self.asset, stance_gate=True,
                    sensor_cfg=entity("contact_forces", body_ids=[0]))


class GateStatsTest(unittest.TestCase):
    def setUp(self):
        self.ns = load_kernel()
        self.fn = self.ns["sent_tracking_excess_l2"]

    def test_gated_fraction_and_loaded_fractions(self):
        target = [[math.radians(30.0)] * 6 + [math.radians(40.0)] * 6] * 4
        forces = [[300.0, 0.0], [300.0, 0.0], [0.0, 0.0], [300.0, 300.0]]
        env = make_env(target, [[0.0] * 12] * 4, forces)
        kwargs = {"asset_cfg": entity("robot", list(range(12))), "stance_gate": True,
                  "sensor_cfg": entity("contact_forces", body_ids=[0, 1])}
        self.fn(env, **kwargs)
        self.fn(env, **kwargs)
        left, right = 6 * math.radians(10.0) ** 2, 6 * math.radians(20.0) ** 2
        log = env.sent_tracking_gate_stats.take_log()
        self.assertAlmostEqual(log["Policy/sent_tracking_gated_fraction"], (3 * left + right) / (4 * (left + right)),
                               places=5)
        self.assertAlmostEqual(log["Policy/sent_tracking_gate_loaded_fraction/l_foot"], 3 / 4)
        self.assertAlmostEqual(log["Policy/sent_tracking_gate_loaded_fraction/r_foot"], 1 / 4)
        self.assertEqual(set(log), {"Policy/sent_tracking_gated_fraction",
                                    "Policy/sent_tracking_gate_loaded_fraction/l_foot",
                                    "Policy/sent_tracking_gate_loaded_fraction/r_foot"})
        empty = env.sent_tracking_gate_stats.take_log()
        self.assertEqual(empty["Policy/sent_tracking_gated_fraction"], 1.0)
        self.assertEqual(empty["Policy/sent_tracking_gate_loaded_fraction/l_foot"], 0.0)

    def test_training_wrapper_logs_the_gate_stats_and_omits_them_without_a_gate(self):
        tracking = load_module("tracking_gate_wrapper", TRACKING)
        data = types.SimpleNamespace(joint_pos=torch.zeros(4, 3), joint_pos_target=torch.zeros(4, 3),
                                     joint_vel=torch.zeros(4, 3))
        robot = types.SimpleNamespace(data=data)

        class BaseWrapper:
            def __init__(self, env, clip_actions=None):
                self.unwrapped = env
                self.device = "cpu"
                self.num_envs = 4
                self.num_actions = 3
                self.clip_actions = clip_actions

        diagnostics = load_nodes(TRAIN, {"DiagnosticVecEnvWrapper"},
                                 {"torch": torch, "math": math, "RslRlVecEnvWrapper": BaseWrapper})
        term = types.SimpleNamespace(_joint_names=["a", "b", "c"], _joint_ids=[0, 1, 2],
                                     _scale=torch.ones(4, 3), _offset=torch.zeros(4, 3), _clip=None)
        scene = Scene(robot=robot)
        scene.sensors = {}
        env = types.SimpleNamespace(action_manager=types.SimpleNamespace(get_term=lambda _: term), scene=scene,
                                    termination_manager=types.SimpleNamespace(compute=lambda: torch.zeros(4)))
        modules = {TRACKING_MODULE: tracking, JOINT_SPEED_MODULE: load_module("joint_speed_gate", JOINT_SPEED)}
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
        log = wrapper.take_action_diagnostics()
        self.assertNotIn("Policy/sent_tracking_gated_fraction", log)
        self.assertIn("Policy/sent_tracking_any_gt20_fraction", log)
        env.sent_tracking_gate_stats = tracking.SentTrackingGateStats(FEET)
        env.sent_tracking_gate_stats.add(torch.ones(2, 12), torch.cat([torch.ones(2, 6), torch.zeros(2, 6)], 1),
                                         torch.tensor([[True, False], [True, False]]))
        log = wrapper.take_action_diagnostics()
        self.assertAlmostEqual(log["Policy/sent_tracking_gated_fraction"], 0.5)
        self.assertAlmostEqual(log["Policy/sent_tracking_gate_loaded_fraction/l_foot"], 1.0)
        self.assertAlmostEqual(log["Policy/sent_tracking_gate_loaded_fraction/r_foot"], 0.0)
        self.assertIn("Policy/sent_tracking_any_gt20_fraction", log)


class GateConfigTest(unittest.TestCase):
    def test_one_term_with_the_gate_off_by_default_and_the_foot_sensor(self):
        tree = ast.parse(ENV_CFG.read_text())
        consts = {n.targets[0].id: n.value for n in tree.body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        self.assertIs(ast.literal_eval(consts["SENT_TRACKING_EXCESS_STANCE_GATE"]), False)
        self.assertEqual(ast.literal_eval(consts["SENT_TRACKING_EXCESS_CONTACT_N"]), 20.0)
        self.assertIsInstance(ast.literal_eval(consts["SENT_TRACKING_EXCESS_CONTACT_N"]), float)
        rewards = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg")
        names = [n.targets[0].id for n in rewards.body if isinstance(n, ast.Assign)]
        self.assertEqual(names.count("sent_tracking_excess"), 1)
        term = next(n for n in rewards.body if isinstance(n, ast.Assign) and n.targets[0].id == "sent_tracking_excess")
        kw = {k.arg: k.value for k in term.value.keywords}
        params = {ast.literal_eval(k): v for k, v in zip(kw["params"].keys, kw["params"].values)}
        self.assertEqual(list(params), ["floor_deg", "asset_cfg", "stance_gate", "contact_threshold_n", "sensor_cfg"])
        self.assertEqual(params["stance_gate"].id, "SENT_TRACKING_EXCESS_STANCE_GATE")
        self.assertEqual(params["contact_threshold_n"].id, "SENT_TRACKING_EXCESS_CONTACT_N")
        sensor = params["sensor_cfg"]
        self.assertEqual(ast.literal_eval(sensor.args[0]), "contact_forces")
        sensor_kw = {k.arg: ast.literal_eval(k.value) for k in sensor.keywords}
        self.assertEqual(sensor_kw, {"body_names": ["l_foot", "r_foot"], "preserve_order": True})

    def test_signature_defaults_keep_the_gate_off(self):
        tree = ast.parse(REWARDS.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sent_tracking_excess_l2")
        args = [a.arg for a in fn.args.args]
        defaults = dict(zip(args[len(args) - len(fn.args.defaults):], fn.args.defaults))
        self.assertIs(ast.literal_eval(defaults["stance_gate"]), False)
        self.assertEqual(ast.literal_eval(defaults["contact_threshold_n"]), 20.0)
        self.assertIsNone(ast.literal_eval(defaults["sensor_cfg"]))


def gate_term(weight=-200.0):
    return types.SimpleNamespace(
        func=FUNC, weight=weight,
        params={"floor_deg": 20.0, "asset_cfg": {"name": "robot", "joint_names": list(LEG_JOINTS)},
                "stance_gate": False, "contact_threshold_n": 20.0,
                "sensor_cfg": {"name": "contact_forces", "body_names": list(FEET), "preserve_order": True}},
    )


class GateOverrideTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def cfg(self):
        return types.SimpleNamespace(
            rewards=types.SimpleNamespace(sent_tracking_excess=gate_term()),
            events=types.SimpleNamespace(), observations=types.SimpleNamespace(),
            scene=types.SimpleNamespace(robot=types.SimpleNamespace(spawn=types.SimpleNamespace(usd_path=None))),
        )

    def compose(self, overrides):
        try:
            from hydra import compose, initialize
            from hydra.core.config_store import ConfigStore
            from hydra.core.global_hydra import GlobalHydra
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("hydra not installed")
        live = self.cfg()
        term = live.rewards.sent_tracking_excess
        node = {"env": {"rewards": {"sent_tracking_excess": {"func": term.func, "weight": term.weight,
                                                             "params": copy.deepcopy(term.params)}}}}
        ConfigStore.instance().store(name="w104_stance_gate_override_test", node=node)
        GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path=None):
            composed = compose(config_name="w104_stance_gate_override_test", overrides=overrides)
        GlobalHydra.instance().clear()
        return live, OmegaConf.to_container(composed, resolve=True)["env"]

    def test_override_true_is_a_bool_and_survives_the_saved_yaml(self):
        live, overridden = self.compose(["env.rewards.sent_tracking_excess.params.stance_gate=true",
                                         "env.rewards.sent_tracking_excess.params.contact_threshold_n=30.0"])
        params = overridden["rewards"]["sent_tracking_excess"]["params"]
        self.assertIs(params["stance_gate"], True)
        with FakeIsaacModules() as update:
            update(live, overridden)
            term = live.rewards.sent_tracking_excess
            self.assertIs(term.params["stance_gate"], True)
            self.assertEqual(term.params["contact_threshold_n"], 30.0)
            saved = {"rewards": {"sent_tracking_excess": {"func": term.func, "weight": term.weight,
                                                          "params": term.params}}}
            run = Path(self.tmp.name)
            (run / "params").mkdir()
            (run / "params" / "env.yaml").write_text(yaml.dump(saved))
            restored = self.cfg()
            load_module("deploy_effects_gate_roundtrip", EFFECTS).apply_training_env_cfg(
                restored, str(run / "model_999.pt"))
        params = restored.rewards.sent_tracking_excess.params
        self.assertIs(params["stance_gate"], True)
        self.assertEqual(params["contact_threshold_n"], 30.0)
        self.assertEqual(params["sensor_cfg"]["body_names"], list(FEET))
        self.assertEqual(restored.rewards.sent_tracking_excess.weight, -200.0)

    def test_without_the_override_the_gate_stays_off(self):
        _, overridden = self.compose([])
        self.assertIs(overridden["rewards"]["sent_tracking_excess"]["params"]["stance_gate"], False)

    def test_integer_threshold_override_is_rejected_by_the_isaac_merge(self):
        live, overridden = self.compose(["env.rewards.sent_tracking_excess.params.contact_threshold_n=30"])
        with FakeIsaacModules() as update, self.assertRaises(ValueError):
            update(live, overridden)


class OldRunGateTest(unittest.TestCase):
    def test_w104_env_yaml_without_the_gate_params_restores_with_the_gate_off(self):
        path = W104_RUN / "params" / "env.yaml"
        if not path.is_file():
            self.skipTest("W104s44a run directory not found in the main worktree")
        effects = load_module("deploy_effects_gate_old_run", EFFECTS)
        with path.open() as handle:
            saved = yaml.load(handle, Loader=effects._SavedCfgLoader)
        self.assertNotIn("stance_gate", saved["rewards"]["sent_tracking_excess"]["params"])
        live = to_cfg({k: v for k, v in saved.items() if k not in ("viewer", "log_dir", "seed")})
        live.rewards.sent_tracking_excess = gate_term(weight=0.0)
        live.rewards.sent_tracking_excess.params["asset_cfg"] = copy.deepcopy(
            saved["rewards"]["sent_tracking_excess"]["params"]["asset_cfg"])
        live.rewards.sent_tracking_excess.params["stance_gate"] = True
        with FakeIsaacModules():
            effects.apply_training_env_cfg(live, str(W104_RUN / "model_999.pt"))
        term = live.rewards.sent_tracking_excess
        self.assertEqual(term.weight, -200.0)
        self.assertEqual(sorted(term.params), ["asset_cfg", "floor_deg"])
        ns = load_kernel()
        target, position = random_errors(16, seed=4)
        env = make_env(target, position, [[0.0, 0.0]] * 16)
        asset = entity("robot", list(range(12)))
        out = ns["sent_tracking_excess_l2"](env, floor_deg=term.params["floor_deg"], asset_cfg=asset)
        self.assertTrue(torch.equal(out, ns["original"](env, asset_cfg=asset)))
        self.assertGreater(out.sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
