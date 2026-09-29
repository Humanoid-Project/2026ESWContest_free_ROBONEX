import ast
import importlib.util
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
EVENTS = TASK / "mdp/events.py"
CFG = TASK / "robonex_walking_v2_env_cfg.py"
CONTRACT = TASK / "robot_contract_v2.py"
EFFECTS = ROOT / "scripts/rsl_rl/deploy_effects.py"

MEASURED_KINETIC_NM = {"rs02": 0.14, "rs03": 0.47}


def load_event(version_major=5):
    node = next(n for n in ast.parse(EVENTS.read_text()).body
                if getattr(n, "name", None) == "randomize_joint_coulomb_friction")
    version = types.SimpleNamespace(major=version_major)
    ns = {"torch": torch, "SceneEntityCfg": object, "Articulation": object,
          "get_isaac_sim_version": lambda: version}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "events", "exec"), ns)
    return ns["randomize_joint_coulomb_friction"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cfg_tree():
    return ast.parse(CFG.read_text())


def constant(name):
    for node in cfg_tree().body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise KeyError(name)


def event_term(name):
    events = next(n for n in cfg_tree().body if isinstance(n, ast.ClassDef) and n.name == "EventCfg")
    for node in events.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == name:
            return {k.arg: k.value for k in node.value.keywords}
    raise KeyError(name)


class FakeAsset:
    def __init__(self, joint_names, num_envs):
        self.joint_names = list(joint_names)
        self.num_joints = len(joint_names)
        self.device = "cpu"
        self.writes = []
        self.num_envs = num_envs

    def write_joint_friction_coefficient_to_sim(self, **kwargs):
        self.writes.append(kwargs)


class Scene(dict):
    def __init__(self, asset):
        super().__init__(robot=asset)
        self.num_envs = asset.num_envs


class CoulombEventTest(unittest.TestCase):
    JOINTS = ["passive_a", "l_hip_yaw_joint", "l_hip_pitch_joint", "passive_b", "l_knee_pitch_joint"]
    RANGE = {"l_hip_yaw_joint": (0.05, 0.25), "l_hip_pitch_joint": (0.2, 0.8), "l_knee_pitch_joint": (0.2, 0.8)}

    def call(self, env_ids, static_ratio=1.0, viscous=0.0, version_major=5):
        asset = FakeAsset(self.JOINTS, num_envs=64)
        env = types.SimpleNamespace(scene=Scene(asset))
        asset_cfg = types.SimpleNamespace(name="robot", joint_ids=[1, 2, 4])
        load_event(version_major)(env, env_ids, asset_cfg, self.RANGE, static_ratio=static_ratio, viscous=viscous)
        return asset

    def test_draws_per_env_per_joint_in_newton_metres_by_joint_name(self):
        torch.manual_seed(0)
        asset = self.call(torch.arange(64))
        self.assertEqual(len(asset.writes), 1)
        write = asset.writes[0]
        dynamic = write["joint_dynamic_friction_coeff"]
        self.assertEqual(tuple(dynamic.shape), (64, 3))
        self.assertEqual(write["joint_ids"].tolist(), [1, 2, 4])
        self.assertEqual(write["env_ids"].tolist(), list(range(64)))
        for column, name in enumerate(["l_hip_yaw_joint", "l_hip_pitch_joint", "l_knee_pitch_joint"]):
            low, high = self.RANGE[name]
            self.assertTrue(bool(torch.all(dynamic[:, column] >= low)) and bool(torch.all(dynamic[:, column] <= high)))
            self.assertGreater(float(dynamic[:, column].std()), 0.0)
        self.assertGreater(float(dynamic[:, 1].max()), 0.25)
        self.assertNotEqual(dynamic[:, 1].tolist(), dynamic[:, 2].tolist())
        self.assertTrue(torch.equal(write["joint_friction_coeff"], dynamic))
        self.assertTrue(torch.equal(write["joint_viscous_friction_coeff"], torch.zeros_like(dynamic)))

    def test_static_ratio_and_viscous_are_applied(self):
        asset = self.call(torch.tensor([3, 7]), static_ratio=2.0, viscous=0.001)
        write = asset.writes[0]
        self.assertEqual(write["env_ids"].tolist(), [3, 7])
        self.assertTrue(torch.allclose(write["joint_friction_coeff"], 2.0 * write["joint_dynamic_friction_coeff"]))
        self.assertTrue(torch.allclose(write["joint_viscous_friction_coeff"], torch.full((2, 3), 0.001)))

    def test_none_env_ids_means_every_env(self):
        asset = self.call(None)
        self.assertEqual(asset.writes[0]["env_ids"].tolist(), list(range(64)))

    def test_refuses_isaac_sim_4_where_the_value_is_a_load_coefficient(self):
        with self.assertRaises(RuntimeError):
            self.call(None, version_major=4)

    def test_refuses_static_below_dynamic(self):
        with self.assertRaises(ValueError):
            self.call(None, static_ratio=0.5)


class WiringTest(unittest.TestCase):
    def test_the_joint_friction_event_is_the_coulomb_term(self):
        term = event_term("randomize_joint_friction")
        self.assertEqual(ast.unparse(term["func"]), "mdp.randomize_joint_coulomb_friction")
        self.assertEqual(ast.literal_eval(term["mode"]), "reset")
        params = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(term["params"].keys, term["params"].values)}
        self.assertEqual(set(params), {"asset_cfg", "friction_range", "static_ratio", "viscous"})
        self.assertIn("LEG_JOINTS", params["asset_cfg"])
        self.assertIn("MOTOR_MODEL_BY_JOINT", params["friction_range"])
        self.assertEqual(params["static_ratio"], "JOINT_STATIC_FRICTION_RATIO")
        self.assertEqual(params["viscous"], "0.0")
        self.assertNotIn("randomize_joint_parameters", CFG.read_text())

    def test_ranges_cover_the_measured_friction(self):
        ranges = constant("JOINT_FRICTION_RANGE")
        self.assertEqual(set(ranges), {"rs02", "rs03"})
        for model, measured in MEASURED_KINETIC_NM.items():
            low, high = ranges[model]
            self.assertTrue(0.0 <= low < measured < high, (model, ranges[model]))
        self.assertEqual(constant("JOINT_STATIC_FRICTION_RATIO"), 1.0)

    def test_every_leg_joint_has_a_motor_model(self):
        contract = load_module("robot_contract_under_test", CONTRACT)
        self.assertEqual(set(contract.MOTOR_MODEL_BY_JOINT), set(contract.LEG_JOINTS))
        models = list(contract.MOTOR_MODEL_BY_JOINT.values())
        self.assertEqual((models.count("rs02"), models.count("rs03")), (6, 6))
        for name, model in contract.MOTOR_MODEL_BY_JOINT.items():
            expected = "rs03" if any(k in name for k in ("hip_pitch", "hip_roll", "knee_pitch")) else "rs02"
            self.assertEqual(model, expected, name)


class RestoreTest(unittest.TestCase):
    STOCK = {
        "func": "isaaclab.envs.mdp.events:randomize_joint_parameters",
        "mode": "reset",
        "params": {
            "asset_cfg": {"name": "robot", "joint_names": ["l_hip_yaw_joint"]},
            "friction_distribution_params": (0.0, 0.02),
            "operation": "add",
            "distribution": "uniform",
        },
    }

    def new_term(self):
        return types.SimpleNamespace(func="coulomb", params={
            "asset_cfg": object(), "friction_range": {"l_hip_yaw_joint": (0.05, 0.25)},
            "static_ratio": 1.0, "viscous": 0.0})

    def test_an_old_run_restores_its_stock_friction_params(self):
        effects = load_module("deploy_effects_under_test", EFFECTS)
        term = self.new_term()
        env_cfg = types.SimpleNamespace(events=types.SimpleNamespace(randomize_joint_friction=term))
        saved = {"events": {"randomize_joint_friction": self.STOCK}}
        effects._add_trained_params(env_cfg, saved)
        for key in ("friction_distribution_params", "operation", "distribution"):
            self.assertEqual(term.params[key], self.STOCK["params"][key])
        effects._drop_untrained_params(term, self.STOCK)
        self.assertEqual(set(term.params), set(self.STOCK["params"]))

    def test_a_new_run_restores_unchanged(self):
        effects = load_module("deploy_effects_under_test", EFFECTS)
        term = self.new_term()
        env_cfg = types.SimpleNamespace(events=types.SimpleNamespace(randomize_joint_friction=term))
        trained = {"params": {"asset_cfg": {"name": "robot"}, "friction_range": {"l_hip_yaw_joint": (0.1, 0.1)},
                              "static_ratio": 1.0, "viscous": 0.0}}
        effects._add_trained_params(env_cfg, {"events": {"randomize_joint_friction": trained}})
        effects._drop_untrained_params(term, trained)
        self.assertEqual(set(term.params), {"asset_cfg", "friction_range", "static_ratio", "viscous"})


class PinTest(unittest.TestCase):
    def env_cfg(self):
        asset_cfg = object()
        term = types.SimpleNamespace(func="stock", params={"asset_cfg": asset_cfg, "operation": "add"})
        return types.SimpleNamespace(events=types.SimpleNamespace(randomize_joint_friction=term)), asset_cfg

    def test_pin_sets_one_level_per_motor_model(self):
        effects = load_module("deploy_effects_under_test", EFFECTS)
        contract = load_module("robot_contract_under_test", CONTRACT)
        env_cfg, asset_cfg = self.env_cfg()
        coulomb = object()
        recorded = effects.pin_joint_friction(env_cfg, {"rs02": 0.14, "rs03": 0.47}, coulomb)
        self.assertEqual(recorded, {"rs02": 0.14, "rs03": 0.47})
        term = env_cfg.events.randomize_joint_friction
        self.assertIs(term.func, coulomb)
        self.assertIs(term.params["asset_cfg"], asset_cfg)
        self.assertEqual(term.params["static_ratio"], 1.0)
        self.assertEqual(term.params["viscous"], 0.0)
        self.assertEqual(set(term.params["friction_range"]), set(contract.LEG_JOINTS))
        for name, (low, high) in term.params["friction_range"].items():
            level = 0.47 if contract.MOTOR_MODEL_BY_JOINT[name] == "rs03" else 0.14
            self.assertEqual((low, high), (level, level))

    def test_pin_rejects_bad_levels_and_a_missing_term(self):
        effects = load_module("deploy_effects_under_test", EFFECTS)
        with self.assertRaises(ValueError):
            effects.pin_joint_friction(self.env_cfg()[0], {"rs02": 0.14}, object())
        with self.assertRaises(ValueError):
            effects.pin_joint_friction(self.env_cfg()[0], {"rs02": -0.1, "rs03": 0.47}, object())
        with self.assertRaises(KeyError):
            effects.pin_joint_friction(types.SimpleNamespace(events=types.SimpleNamespace()), {"rs02": 0.1, "rs03": 0.4},
                                       object())


if __name__ == "__main__":
    unittest.main()
