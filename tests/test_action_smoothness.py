import ast
import types
import unittest
from pathlib import Path

import torch

REWARDS = Path(__file__).resolve().parents[1] / (
    "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking/mdp/rewards.py"
)


class Base:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


def load():
    tree = ast.parse(REWARDS.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in {"_bounded_square", "action_smoothness_l2"}]
    ns = {"torch": torch, "ManagerTermBase": Base, "ManagerBasedRLEnv": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "rewards", "exec"), ns)
    return ns["action_smoothness_l2"]


class Env:
    def __init__(self, scale):
        self.device = "cpu"
        self.common_step_counter = 0
        term = types.SimpleNamespace(_scale=torch.tensor(scale))
        self.action_manager = types.SimpleNamespace(action=torch.zeros(1, len(scale)), get_term=lambda name: term)


class ActionSmoothnessTest(unittest.TestCase):
    def test_constant_velocity_costs_only_first_order_and_chatter_costs_more(self):
        cls = load()
        env = Env([0.25, 0.25])
        term = cls(types.SimpleNamespace(params={}), env)
        values = []
        for k in range(6):
            env.common_step_counter = k
            env.action_manager.action = torch.tensor([[0.4 * k, 0.0]])
            values.append(float(term(env)))
        self.assertAlmostEqual(values[-1], (0.4 * 0.25) ** 2, places=6)
        chatter = []
        for k in range(6, 12):
            env.common_step_counter = k
            env.action_manager.action = torch.tensor([[2.0 * (-1) ** k, 0.0]])
            chatter.append(float(term(env)))
        self.assertGreater(min(chatter[2:]), 10 * values[-1])

    def test_value_is_computed_once_per_step(self):
        cls = load()
        env = Env([0.25])
        term = cls(types.SimpleNamespace(params={}), env)
        env.common_step_counter = 1
        env.action_manager.action = torch.tensor([[1.0]])
        a = float(term(env)); b = float(term(env))
        self.assertEqual(a, b)

    def test_reset_zeroes_history_of_those_envs(self):
        cls = load()
        env = Env([0.25])
        env.action_manager.action = torch.tensor([[1.0], [1.0]])
        term = cls(types.SimpleNamespace(params={}), env)
        env.common_step_counter = 1
        term(env)
        term.reset(torch.tensor([0]))
        self.assertEqual(term._prev[:, 0].tolist(), [0.0, 0.25])


if __name__ == "__main__":
    unittest.main()
