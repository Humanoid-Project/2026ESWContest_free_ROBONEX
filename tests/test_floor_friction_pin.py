import ast
import importlib.util
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EFFECTS = ROOT / "scripts/rsl_rl/deploy_effects.py"
EVAL = ROOT / "scripts/rsl_rl/eval_commands.py"


def effects():
    spec = importlib.util.spec_from_file_location("deploy_effects_floor_mu", EFFECTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cfg(combine="average"):
    material = types.SimpleNamespace(static_friction=0.6, dynamic_friction=0.6, friction_combine_mode=combine)
    ground = types.SimpleNamespace(spawn=types.SimpleNamespace(physics_material=material))
    term = types.SimpleNamespace(params={"static_friction_range": (0.3, 1.0), "dynamic_friction_range": (0.3, 1.0),
                                         "restitution_range": (0.0, 0.0), "num_buckets": 64})
    return types.SimpleNamespace(events=types.SimpleNamespace(randomize_friction=term),
                                 scene=types.SimpleNamespace(ground=ground))


class FloorFrictionPin(unittest.TestCase):
    def test_pins_robot_range_and_ground_material(self):
        env_cfg = cfg("min")
        out = effects().pin_floor_friction(env_cfg, 0.3)
        params = env_cfg.events.randomize_friction.params
        self.assertEqual(params["static_friction_range"], (0.3, 0.3))
        self.assertEqual(params["dynamic_friction_range"], (0.3, 0.3))
        self.assertEqual(params["num_buckets"], 64)
        material = env_cfg.scene.ground.spawn.physics_material
        self.assertEqual((material.static_friction, material.dynamic_friction), (0.3, 0.3))
        self.assertEqual(out, {"mu": 0.3, "ground_combine": "min"})

    def test_combined_value_is_mu_under_every_combine_mode(self):
        rules = {"average": lambda a, b: (a + b) / 2, "min": min, "max": max, "multiply": lambda a, b: a * b}
        for mode in ("average", "min", "max"):
            env_cfg = cfg(mode)
            effects().pin_floor_friction(env_cfg, 0.5)
            robot = env_cfg.events.randomize_friction.params["static_friction_range"][0]
            ground = env_cfg.scene.ground.spawn.physics_material.static_friction
            self.assertAlmostEqual(rules[mode](robot, ground), 0.5, msg=mode)

    def test_none_leaves_the_config_alone(self):
        env_cfg = cfg()
        self.assertIsNone(effects().pin_floor_friction(env_cfg, None))
        self.assertEqual(env_cfg.events.randomize_friction.params["static_friction_range"], (0.3, 1.0))
        self.assertEqual(env_cfg.scene.ground.spawn.physics_material.static_friction, 0.6)

    def test_rejects_non_positive_and_missing_event(self):
        with self.assertRaises(ValueError):
            effects().pin_floor_friction(cfg(), 0.0)
        env_cfg = cfg()
        env_cfg.events.randomize_friction = None
        with self.assertRaises(KeyError):
            effects().pin_floor_friction(env_cfg, 0.5)

    def test_eval_pins_after_restore_before_make_and_records_it(self):
        tree = ast.parse(EVAL.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        lines = {}
        for node in ast.walk(main):
            if isinstance(node, ast.Call):
                lines.setdefault(ast.unparse(node.func), node.lineno)
        self.assertLess(lines["apply_training_env_cfg"], lines["pin_floor_friction"])
        self.assertLess(lines["pin_floor_friction"], lines["gym.make"])
        call = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
                    and ast.unparse(node.func) == "pin_floor_friction")
        self.assertEqual([ast.unparse(a) for a in call.args], ["env_cfg", "args_cli.floor_friction"])
        self.assertIn('"floor_friction": floor_friction,', EVAL.read_text())


if __name__ == "__main__":
    unittest.main()
