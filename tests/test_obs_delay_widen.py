import ast
import importlib.util
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EFFECTS = ROOT / "scripts/rsl_rl/deploy_effects.py"
SCRIPTS = [ROOT / "scripts/rsl_rl/eval_commands.py", ROOT / "scripts/rsl_rl/sim_joint_trace.py"]


def effects():
    spec = importlib.util.spec_from_file_location("deploy_effects_widen", EFFECTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cfg(max_steps):
    term = lambda: types.SimpleNamespace(params={"max_delay_steps": max_steps, "min_delay_steps": 0, "field": "pos"})
    policy = types.SimpleNamespace(joint_pos_rel=term(), joint_vel_rel=term(), base_ang_vel=types.SimpleNamespace(params={}))
    return types.SimpleNamespace(observations=types.SimpleNamespace(policy=policy))


class WidenJointObsDelay(unittest.TestCase):
    def test_widens_both_terms_below_the_pin(self):
        env_cfg = cfg(0)
        out = effects().widen_joint_obs_delay(env_cfg, 1)
        self.assertEqual(out, [{"term": "joint_pos_rel", "trained_max": 0, "max": 1},
                               {"term": "joint_vel_rel", "trained_max": 0, "max": 1}])
        policy = env_cfg.observations.policy
        self.assertEqual(policy.joint_pos_rel.params["max_delay_steps"], 1)
        self.assertEqual(policy.joint_vel_rel.params["max_delay_steps"], 1)
        self.assertEqual(policy.joint_pos_rel.params["min_delay_steps"], 0)

    def test_leaves_wide_enough_terms_and_no_pin_alone(self):
        env_cfg = cfg(1)
        self.assertEqual(effects().widen_joint_obs_delay(env_cfg, 1), [])
        self.assertEqual(effects().widen_joint_obs_delay(env_cfg, 0), [])
        self.assertEqual(effects().widen_joint_obs_delay(cfg(0), -1), [])
        self.assertEqual(env_cfg.observations.policy.joint_pos_rel.params["max_delay_steps"], 1)

    def test_scripts_widen_before_make_and_record_it(self):
        for path in SCRIPTS:
            tree = ast.parse(path.read_text())
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
            lines = {}
            for node in ast.walk(main):
                if isinstance(node, ast.Call):
                    lines.setdefault(ast.unparse(node.func), node.lineno)
            self.assertLess(lines["widen_joint_obs_delay"], lines["gym.make"], path.name)
            self.assertLess(lines["gym.make"], lines["pin_joint_obs_delay"], path.name)
            self.assertIn('"obs_delay_widened": obs_widened,', path.read_text(), path.name)


if __name__ == "__main__":
    unittest.main()
