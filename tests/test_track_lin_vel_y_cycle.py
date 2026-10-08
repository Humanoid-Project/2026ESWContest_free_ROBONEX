import ast
import math
import types
import unittest
from pathlib import Path

import torch

REWARDS = Path(__file__).resolve().parents[1] / (
    "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking/mdp/rewards.py"
)
ENV_CFG = REWARDS.parents[1] / "robonex_walking_v2_env_cfg.py"


class Base:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


def load():
    tree = ast.parse(REWARDS.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) == "track_lin_vel_y_cycle_exp"]
    ns = {"torch": torch, "ManagerTermBase": Base, "ManagerBasedRLEnv": object, "SceneEntityCfg": lambda *a, **k: None}
    ns["_root_lin_vel_yaw_frame"] = lambda env, cfg: env.vel
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "rewards", "exec"), ns)
    return ns["track_lin_vel_y_cycle_exp"]


class Env:
    def __init__(self, n):
        self.common_step_counter = 0
        self.command = torch.zeros(n, 3)
        self.vel = torch.zeros(n, 3)
        self.command_manager = types.SimpleNamespace(get_command=lambda name: self.command)


class TrackLinVelYCycleTest(unittest.TestCase):
    def run_steps(self, term, env, vy, steps, start=0):
        out = None
        for k in range(steps):
            env.common_step_counter = start + k
            env.vel[:, 1] = vy(k) if callable(vy) else vy
            out = term(env, std=0.15, window_steps=40)
        return out

    def test_sway_around_the_command_is_not_penalised(self):
        term_cls = load()
        env = Env(2)
        env.command[:, 1] = torch.tensor([0.2, 0.2])
        term = term_cls(types.SimpleNamespace(params={}), env)
        out = None
        for k in range(80):
            env.common_step_counter = k
            sway = 0.25 * math.sin(2 * math.pi * k / 40)
            env.vel[:, 1] = torch.tensor([0.2 + sway, 0.0 + sway])
            out = term(env, std=0.15, window_steps=40)
        self.assertGreater(float(out[0]), 0.99)
        self.assertAlmostEqual(float(out[1]), math.exp(-(0.2 / 0.15) ** 2), places=3)
        instantaneous = math.exp(-(0.25 / 0.15) ** 2)
        self.assertLess(instantaneous, 0.1)

    def test_command_change_and_reset_clear_the_window(self):
        term_cls = load()
        env = Env(2)
        term = term_cls(types.SimpleNamespace(params={}), env)
        self.run_steps(term, env, 0.0, 40)
        env.command[0, 1] = 0.2
        out = self.run_steps(term, env, 0.2, 1, start=40)
        self.assertAlmostEqual(float(out[0]), 1.0, places=6)
        self.assertAlmostEqual(float(out[1]), math.exp(-(0.2 / 40 / 0.15) ** 2), places=6)
        self.run_steps(term, env, 0.0, 39, start=41)
        term.reset(torch.tensor([1]))
        env.vel[:, 1] = 0.0
        env.common_step_counter = 80
        out = term(env, std=0.15, window_steps=40)
        self.assertAlmostEqual(float(out[1]), 1.0, places=6)

    def test_same_step_call_does_not_advance_the_window(self):
        term_cls = load()
        env = Env(1)
        term = term_cls(types.SimpleNamespace(params={}), env)
        env.vel[:, 1] = 0.1
        a = term(env, std=0.15, window_steps=40)
        b = term(env, std=0.15, window_steps=40)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(int(term._count[0]), 1)

    def test_cfg_term_is_inert_by_default(self):
        tree = ast.parse(ENV_CFG.read_text())
        found = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "track_lin_vel_y_cycle" for t in node.targets):
                found = node
        self.assertIsNotNone(found)
        kw = {k.arg: k.value for k in found.value.keywords}
        self.assertEqual(ast.literal_eval(kw["weight"]), 0.0)
        self.assertEqual(kw["func"].attr, "track_lin_vel_y_cycle_exp")


if __name__ == "__main__":
    unittest.main()
