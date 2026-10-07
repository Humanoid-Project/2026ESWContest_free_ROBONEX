import ast
import importlib.util
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EFFECTS = ROOT / "scripts/rsl_rl/deploy_effects.py"
EVAL = ROOT / "scripts/rsl_rl/eval_commands.py"
ENV_CFG = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking/robonex_walking_v2_env_cfg.py"


def effects():
    spec = importlib.util.spec_from_file_location("deploy_effects_hip_yaw_kp", EFFECTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def env(present=True, operation="scale", distribution="uniform"):
    term = None
    if present:
        term = types.SimpleNamespace(params={
            "asset_cfg": object(),
            "stiffness_distribution_params": (0.25, 1.0),
            "operation": operation,
            "distribution": distribution,
        })
    return types.SimpleNamespace(events=types.SimpleNamespace(randomize_hip_yaw_stiffness=term))


class HipYawKpNominalTest(unittest.TestCase):
    def test_default_keeps_training_range(self):
        cfg = env()
        record = effects().apply_hip_yaw_kp_nominal(cfg, False)
        self.assertEqual(cfg.events.randomize_hip_yaw_stiffness.params["stiffness_distribution_params"], (0.25, 1.0))
        self.assertEqual(record, {"source": "training", "event_present": True, "stiffness_scale": [0.25, 1.0]})

    def test_pin_sets_nominal_scale(self):
        cfg = env()
        record = effects().apply_hip_yaw_kp_nominal(cfg, True)
        self.assertEqual(cfg.events.randomize_hip_yaw_stiffness.params["stiffness_distribution_params"], (1.0, 1.0))
        self.assertEqual(record, {"source": "pin", "event_present": True, "stiffness_scale": [1.0, 1.0]})

    def test_pin_refused_when_one_is_not_nominal(self):
        for operation, distribution in (("add", "uniform"), ("abs", "uniform"), ("scale", "gaussian")):
            with self.assertRaises(ValueError):
                effects().apply_hip_yaw_kp_nominal(env(operation=operation, distribution=distribution), True)
        record = effects().apply_hip_yaw_kp_nominal(env(distribution="log_uniform"), True)
        self.assertEqual(record["stiffness_scale"], [1.0, 1.0])

    def test_pin_without_event_refused(self):
        with self.assertRaises(KeyError):
            effects().apply_hip_yaw_kp_nominal(env(present=False), True)
        record = effects().apply_hip_yaw_kp_nominal(env(present=False), False)
        self.assertEqual(record, {"source": "training", "event_present": False, "stiffness_scale": None})

    def test_event_name_matches_the_env_cfg(self):
        tree = ast.parse(ENV_CFG.read_text())
        events = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EventCfg")
        names = [n.targets[0].id for n in events.body if isinstance(n, ast.Assign)]
        self.assertIn(effects().HIP_YAW_STIFFNESS_EVENT, names)

    def test_eval_records_and_applies_after_config_load(self):
        text = EVAL.read_text()
        self.assertIn('"--hip_yaw_kp_nominal"', text)
        self.assertIn('"hip_yaw_kp": hip_yaw_kp', text)
        pin = text.index("apply_hip_yaw_kp_nominal(env_cfg, args_cli.hip_yaw_kp_nominal)")
        self.assertLess(text.index("apply_training_env_cfg(env_cfg, args_cli.checkpoint)"), pin)
        self.assertLess(pin, text.index("gym.make(args_cli.task, cfg=env_cfg)"))


if __name__ == "__main__":
    unittest.main()
