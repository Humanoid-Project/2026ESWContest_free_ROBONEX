import ast
import math
import random
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking/mdp/actions.py"
DEPLOY_SAFETY = ROOT.parent / "robonex-deploy/scripts/sim_to_real/safety.py"


def load_function(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def clamp(value, low, high):
    return max(low, min(high, value))


class SlewLimitStepTest(unittest.TestCase):
    def setUp(self):
        self.step = load_function(ACTIONS, "slew_limit_step", {"torch": torch})
        if not DEPLOY_SAFETY.exists():
            self.skipTest("robonex-deploy checkout not found next to robonex-walking")
        namespace = {"math": math, "clamp": clamp, "dataclass": dataclass}
        self.axis_limiter = dataclass(load_function(DEPLOY_SAFETY, "AxisLimiter", namespace))

    def test_matches_deploy_axis_limiter(self):
        rng = random.Random(0)
        dt, max_speed, max_accel = 0.02, 6.0, 120.0
        for _ in range(20):
            start = rng.uniform(-1.0, 1.0)
            deploy = self.axis_limiter(start)
            position = torch.tensor([start], dtype=torch.float64)
            velocity = torch.zeros(1, dtype=torch.float64)
            for _ in range(200):
                target = rng.uniform(-1.5, 1.5) if rng.random() < 0.2 else deploy.position + rng.uniform(-0.05, 0.05)
                expected_position, expected_velocity = deploy.step(target, dt, max_speed, max_accel)
                position, velocity = self.step(
                    position, velocity, torch.tensor([target], dtype=torch.float64), dt, max_speed, max_accel
                )
                self.assertAlmostEqual(position.item(), expected_position, places=9)
                self.assertAlmostEqual(velocity.item(), expected_velocity, places=9)

    def test_speed_and_acceleration_bounds(self):
        dt, max_speed, max_accel = 0.02, 6.0, 120.0
        position = torch.zeros(4)
        velocity = torch.zeros(4)
        target = torch.tensor([2.0, -2.0, 0.01, 0.0])
        previous = velocity.clone()
        for _ in range(50):
            position, velocity = self.step(position, velocity, target, dt, max_speed, max_accel)
            self.assertTrue(torch.all(velocity.abs() <= max_speed + 1e-6))
            self.assertTrue(torch.all((velocity - previous).abs() <= max_accel * dt + 1e-5))
            previous = velocity.clone()
        self.assertTrue(torch.allclose(position, target))


if __name__ == "__main__":
    unittest.main()


class SlewLagRewardTest(unittest.TestCase):
    def test_lag_penalty_is_zero_without_lag_and_saturates(self):
        rewards = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking/mdp/rewards.py"
        ns = {"torch": torch, "ManagerBasedRLEnv": object}
        tree = ast.parse(rewards.read_text())
        nodes = [n for n in tree.body if getattr(n, "name", None) in ("_bounded_square", "_saturate", "slew_lag_l2")]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "rewards", "exec"), ns)
        term = type("T", (), {})()
        term.slew_lag = torch.tensor([[0.0, 0.0], [0.05, 0.0], [1.0, 1.0]])
        env = type("E", (), {})()
        env.num_envs = 3
        env.device = "cpu"
        env.action_manager = type("M", (), {"get_term": staticmethod(lambda name: term)})()
        out = ns["slew_lag_l2"](env, scale=0.0025)
        self.assertEqual(out[0].item(), 0.0)
        self.assertAlmostEqual(out[1].item(), 1 - math.exp(-1.0), places=5)
        self.assertGreater(out[2].item(), 0.999)
