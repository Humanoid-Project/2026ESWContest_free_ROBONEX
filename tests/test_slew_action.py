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
