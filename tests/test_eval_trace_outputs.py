import ast
import json
import os
import tempfile
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts/rsl_rl/eval_commands.py"
TRACE = ROOT / "scripts/rsl_rl/sim_joint_trace.py"


def main_of(path):
    tree = ast.parse(path.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def clip_count_statement():
    for node in ast.walk(main_of(EVAL)):
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "row['clip_hit_count']":
            return ast.Module(body=[node], type_ignores=[])
    raise AssertionError("eval_commands.main never sets clip_hit_count")


def out_block():
    for node in ast.walk(main_of(EVAL)):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "args_cli.out":
            return ast.Module(body=[node], type_ignores=[])
    raise AssertionError("eval_commands.main has no `if args_cli.out:` block")


def accumulated_hits(num_envs, hits, steps=880, joints=12):
    clip_hits = torch.zeros(joints, dtype=torch.float32)
    for step in range(steps):
        over = torch.zeros(num_envs, joints)
        if step == 10:
            over[:hits, 0] = 1.0
        clip_hits += (over > 0.0).float().mean(dim=0)
    return clip_hits


class ClipHitCountTests(unittest.TestCase):
    def count(self, num_envs, hits):
        row = {}
        namespace = {"row": row, "clip_hits": accumulated_hits(num_envs, hits),
                     "unwrapped": types.SimpleNamespace(num_envs=num_envs)}
        exec(compile(clip_count_statement(), "clip_count", "exec"), namespace)
        return row["clip_hit_count"]

    def test_a_single_hit_at_100_envs_counts_one(self):
        self.assertEqual(self.count(100, 1), 1)

    def test_counts_are_exact_for_non_power_of_two_env_counts(self):
        for num_envs in (100, 300, 500, 600, 1000):
            for hits in (1, 3, 5, 6, 9, 17, 39):
                self.assertEqual(self.count(num_envs, hits), hits, (num_envs, hits))


class EvalOutTests(unittest.TestCase):
    def test_out_into_a_missing_directory_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "new", "nested", "eval.json")
            namespace = {"args_cli": types.SimpleNamespace(out=out), "os": os, "json": json,
                         "meta": {"checkpoint": "x"}, "results": {"cell": {}}, "print": lambda *a, **k: None}
            exec(compile(out_block(), "out_block", "exec"), namespace)
            with open(out, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"_meta": {"checkpoint": "x"}, "cell": {}})


def sample_before_reset():
    node = next(n for n in ast.parse(TRACE.read_text()).body
                if getattr(n, "name", None) == "sample_before_reset")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "trace", "exec"), namespace)
    return namespace["sample_before_reset"]


class TraceTerminalRowTests(unittest.TestCase):
    def test_the_terminal_state_is_sampled_before_the_reset(self):
        dones = iter([torch.tensor([False, False]), torch.tensor([False, True]), torch.tensor([True, False])])
        state = {"root_z": 0.6}
        unwrapped = types.SimpleNamespace(termination_manager=types.SimpleNamespace(compute=lambda: next(dones)))
        terminal = sample_before_reset()(unwrapped, 1, lambda: dict(state))

        def step(fall_height):
            state["root_z"] = fall_height
            done = unwrapped.termination_manager.compute()
            if bool(done[1]):
                state["root_z"] = 0.6
            return done

        step(0.55)
        self.assertEqual(terminal, [])
        step(0.2)
        self.assertEqual(terminal, [{"root_z": 0.2}])
        self.assertEqual(state["root_z"], 0.6)
        terminal.pop()
        step(0.58)
        self.assertEqual(terminal, [])

    def test_the_trace_loop_reads_state_only_through_the_sample(self):
        loop = next(n for n in ast.walk(main_of(TRACE))
                    if isinstance(n, ast.For) and "warmup_steps" in ast.unparse(n.iter))
        source = ast.unparse(loop)
        for name in ("asset.data.", "contact.data.", "unwrapped.episode_length_buf"):
            self.assertNotIn(name, source)
        body = [ast.unparse(stmt) for stmt in loop.body]
        pop = next(i for i, line in enumerate(body) if "terminal.pop()" in line)
        warmup = next(i for i, line in enumerate(body) if line.startswith("if step < args_cli.warmup_steps"))
        self.assertLess(pop, warmup)


if __name__ == "__main__":
    unittest.main()
