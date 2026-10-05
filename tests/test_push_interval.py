import importlib.util
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EFFECTS = ROOT / "scripts/rsl_rl/deploy_effects.py"
EVAL = ROOT / "scripts/rsl_rl/eval_commands.py"


def effects():
    spec = importlib.util.spec_from_file_location("deploy_effects_push_interval", EFFECTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def env(push=True):
    term = types.SimpleNamespace(interval_range_s=(2.0, 4.0), mode="interval") if push else None
    return types.SimpleNamespace(events=types.SimpleNamespace(push_robot=term))


class PushIntervalTest(unittest.TestCase):
    def test_default_keeps_training_interval(self):
        cfg = env()
        record = effects().apply_push_interval(cfg, None)
        self.assertEqual(cfg.events.push_robot.interval_range_s, (2.0, 4.0))
        self.assertEqual(record, {"source": "training", "event_present": True, "interval_s": [2.0, 4.0]})

    def test_pin_sets_interval(self):
        cfg = env()
        record = effects().apply_push_interval(cfg, [10.0, 15.0])
        self.assertEqual(cfg.events.push_robot.interval_range_s, (10.0, 15.0))
        self.assertEqual(record["source"], "pin")
        self.assertEqual(record["interval_s"], [10.0, 15.0])

    def test_invalid_intervals_refused(self):
        for bad in ([0.0, 1.0], [5.0, 4.0], [float("nan"), 2.0], [-1.0, 2.0]):
            with self.assertRaises(ValueError):
                effects().apply_push_interval(env(), bad)

    def test_pin_without_push_event_refused(self):
        with self.assertRaises(KeyError):
            effects().apply_push_interval(env(push=False), [10.0, 15.0])
        record = effects().apply_push_interval(env(push=False), None)
        self.assertEqual(record, {"source": "training", "event_present": False, "interval_s": None})

    def test_eval_records_and_applies_after_push_disabling(self):
        text = EVAL.read_text()
        self.assertIn('"push_interval": push_interval', text)
        self.assertLess(text.index("if args_cli.no_push:"), text.index("apply_push_interval(env_cfg, args_cli.push_interval)"))
        self.assertIn('"--push_interval"', text)


if __name__ == "__main__":
    unittest.main()
