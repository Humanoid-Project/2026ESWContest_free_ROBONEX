import ast
import importlib.util
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "scripts/rsl_rl/sim_joint_trace.py"
EVAL = ROOT / "scripts/rsl_rl/eval_commands.py"
EFFECTS = ROOT / "scripts/rsl_rl/deploy_effects.py"


def load_effects():
    spec = importlib.util.spec_from_file_location("deploy_effects_under_test", EFFECTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parser_args(path):
    args = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)):
            args[node.args[0].value] = {k.arg: k.value for k in node.keywords}
    return args


def pin_assignment(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.IfExp)
                and "pin_joint_obs_delay" in ast.unparse(node.value)):
            return node
    raise AssertionError(f"no pin_joint_obs_delay assignment in {path.name}")


def line_of(path, text):
    lines = [i for i, line in enumerate(path.read_text().splitlines()) if text in line]
    if len(lines) != 1:
        raise AssertionError(f"{text!r} found {len(lines)} times in {path.name}")
    return lines[0]


class FakeDelayedTerm:
    def __init__(self, delay, max_delay=1):
        self._delay = delay
        self._max_delay = max_delay

    def reset(self, env_ids=None):
        self._delay[:] = torch.randint(0, self._max_delay + 1, self._delay.shape)


def fake_env(terms, shared=None, imu=None):
    manager = types.SimpleNamespace(
        _group_obs_term_names={"policy": [name for name, _ in terms]},
        _group_obs_term_cfgs={"policy": [types.SimpleNamespace(func=func) for _, func in terms]},
    )
    env = types.SimpleNamespace(observation_manager=manager,
                                scene=types.SimpleNamespace(sensors={} if imu is None else {"imu": imu}))
    if shared is not None:
        env._joint_observation_delay = shared
    return env


class TraceArgumentTest(unittest.TestCase):
    def test_pin_argument_matches_eval(self):
        trace, evaluate = parser_args(TRACE), parser_args(EVAL)
        self.assertIn("--obs_delay_pin", trace)
        self.assertEqual(ast.literal_eval(trace["--obs_delay_pin"]["default"]), -1)
        self.assertEqual(ast.unparse(trace["--obs_delay_pin"]["type"]), "int")
        self.assertEqual(ast.unparse(trace["--obs_delay_pin"]["type"]), ast.unparse(evaluate["--obs_delay_pin"]["type"]))
        self.assertEqual(ast.literal_eval(trace["--obs_delay"]["default"]), 0)

    def test_pin_call_is_the_eval_call(self):
        trace, evaluate = pin_assignment(TRACE), pin_assignment(EVAL)
        self.assertEqual(ast.unparse(trace.value), ast.unparse(evaluate.value))
        self.assertEqual(ast.unparse(trace.value.test), "args_cli.obs_delay_pin >= 0")
        self.assertEqual(ast.unparse(trace.value.orelse), "[]")

    def test_pin_after_policy_load_and_before_rollout(self):
        load = line_of(TRACE, "runner.load(")
        pin = line_of(TRACE, "pin_joint_obs_delay(unwrapped, args_cli.obs_delay_pin)")
        loop = line_of(TRACE, "for step in range(args_cli.warmup_steps + args_cli.measure_steps)")
        self.assertLess(load, pin)
        self.assertLess(pin, loop)

    def test_sidecar_records_delays_and_lands_before_the_csv(self):
        text = TRACE.read_text()
        for key in ('"obs_delay_pin"', '"obs_delay_pinned_terms"', '"obs_delay_steps"', '"delays"',
                    '"checkpoint"', '"seed"', '"env_index"', '"command"', '"rows"'):
            self.assertIn(key, text)
        self.assertIn('out + ".meta.json"', text)
        self.assertLess(line_of(TRACE, "os.replace(meta_path"), line_of(TRACE, 'with open(out, "w"'))

    def test_trace_imports_the_shared_helpers(self):
        tree = ast.parse(TRACE.read_text())
        imported = {alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
                    and node.module == "deploy_effects" for alias in node.names}
        self.assertTrue({"pin_joint_obs_delay", "observation_delay_state", "summarize_delay_states"} <= imported)


class DelayStateTest(unittest.TestCase):
    def test_pinned_joint_delay_and_imu_delay_are_reported(self):
        effects = load_effects()
        shared = torch.tensor([0, 1, 1, 0])
        imu = types.SimpleNamespace(_delay=torch.tensor([3, 6, 1, 2]),
                                    cfg=types.SimpleNamespace(min_delay_steps=1, max_delay_steps=6))
        pos, vel = FakeDelayedTerm(shared), FakeDelayedTerm(shared)
        env = fake_env([("joint_pos_rel", pos), ("joint_vel_rel", vel), ("imu_ang_vel", len)], shared, imu)
        self.assertEqual(effects.observation_delay_state(env, 0)["joint_obs_delay"], 0)
        effects.pin_joint_obs_delay(env, 1)
        pos.reset()
        state = effects.observation_delay_state(env, 1)
        self.assertEqual(state, {"joint_obs_delay": 1, "joint_obs_max_delay": 1,
                                 "imu_delay_physics_steps": 6, "imu_delay_range_physics_steps": [1, 6]})

    def test_undelayed_policy_and_plain_imu_report_none(self):
        effects = load_effects()
        env = fake_env([("joint_pos_rel", len)], imu=types.SimpleNamespace(cfg=types.SimpleNamespace()))
        self.assertEqual(effects.observation_delay_state(env, 0),
                         {"joint_obs_delay": None, "joint_obs_max_delay": None,
                          "imu_delay_physics_steps": None, "imu_delay_range_physics_steps": None})

    def test_summary_counts_changes_across_resets(self):
        effects = load_effects()
        states = [{"joint_obs_delay": d, "joint_obs_max_delay": 1,
                   "imu_delay_physics_steps": i, "imu_delay_range_physics_steps": [1, 6]}
                  for d, i in ((0, 4), (0, 4), (1, 2), (1, 2), (0, 2))]
        summary = effects.summarize_delay_states(states)
        self.assertEqual(summary["joint_obs_delay"], {"first": 0, "last": 0, "values": [0, 1], "changes": 2})
        self.assertEqual(summary["imu_delay_physics_steps"], {"first": 4, "last": 2, "values": [2, 4], "changes": 1})
        self.assertEqual(summary["joint_obs_max_delay"], 1)
        self.assertEqual(summary["imu_delay_range_physics_steps"], [1, 6])

    def test_summary_of_nothing(self):
        effects = load_effects()
        summary = effects.summarize_delay_states([])
        self.assertIsNone(summary["joint_obs_delay"])
        self.assertIsNone(summary["imu_delay_physics_steps"])


if __name__ == "__main__":
    unittest.main()
