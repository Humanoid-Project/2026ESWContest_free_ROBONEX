import ast
import math
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1] / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
REWARDS = ROOT / "mdp/rewards.py"
CFG = ROOT / "robonex_walking_v2_env_cfg.py"
DT = 0.02


class Base:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


def load():
    tree = ast.parse(REWARDS.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in {"_saturate", "feet_touchdown_descent_l2"}]
    ns = {"torch": torch, "math": math, "ManagerTermBase": Base, "ManagerBasedRLEnv": object, "SceneEntityCfg": object,
          "Articulation": object, "ContactSensor": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "rewards", "exec"), ns)
    return ns["feet_touchdown_descent_l2"], ns["_saturate"]


class Sensor:
    def __init__(self, n):
        self.data = types.SimpleNamespace(current_contact_time=torch.zeros(n, 2))

    def compute_first_contact(self, dt, abs_tol=1.0e-8):
        t = self.data.current_contact_time
        return (t > 0.0) & (t < dt + abs_tol)


class Env:
    def __init__(self, n=1):
        self.step_dt = DT
        self.common_step_counter = 0
        self.vel = torch.zeros(n, 2, 3)
        self.robot = types.SimpleNamespace(data=types.SimpleNamespace(body_lin_vel_w=self.vel))
        self.sensor = Sensor(n)
        self.scene = types.SimpleNamespace(sensors={"contact_forces": self.sensor})
        self.scene_items = {"robot": self.robot}

    def step(self, vz, contact_time):
        self.common_step_counter += 1
        self.vel[:, :, 2] = torch.as_tensor(vz)
        self.sensor.data.current_contact_time[:] = torch.as_tensor(contact_time)


class SceneDict(types.SimpleNamespace):
    def __getitem__(self, key):
        return self.items[key]


def make():
    env = Env()
    env.scene = SceneDict(sensors={"contact_forces": env.sensor}, items={"robot": env.robot})
    return env


SENSOR = types.SimpleNamespace(name="contact_forces", body_ids=[0, 1])
ASSET = types.SimpleNamespace(name="robot", body_ids=[0, 1])


class TouchdownDescentTest(unittest.TestCase):
    def setUp(self):
        self.cls, self.sat = load()

    def call(self, term, env):
        return term(env, sensor_cfg=SENSOR, asset_cfg=ASSET, deadband=0.25, scale=0.16)

    def test_extrapolates_to_the_contact_instant_and_penalises_only_first_contact(self):
        env = make()
        term = self.cls(types.SimpleNamespace(params={}), env)
        env.step([[-0.4, 0.0]], [[0.0, 0.0]])
        self.assertEqual(self.call(term, env).item(), 0.0)
        env.step([[-0.6, 0.0]], [[0.0, 0.0]])
        self.assertEqual(self.call(term, env).item(), 0.0)
        env.step([[-0.1, 0.0]], [[0.005, 0.0]])
        lag = DT - 0.005
        speed = 0.6 + (0.6 - 0.4) * lag / DT
        expected = 1.0 - math.exp(-((speed - 0.25) ** 2) / 0.16)
        self.assertAlmostEqual(self.call(term, env).item(), expected, places=5)
        env.step([[0.0, 0.0]], [[0.025, 0.0]])
        self.assertEqual(self.call(term, env).item(), 0.0)

    def test_slow_landing_inside_the_deadband_is_free(self):
        env = make()
        term = self.cls(types.SimpleNamespace(params={}), env)
        env.step([[-0.1, 0.0]], [[0.0, 0.0]])
        self.call(term, env)
        env.step([[-0.2, 0.0]], [[0.0, 0.0]])
        self.call(term, env)
        env.step([[0.0, 0.0]], [[0.015, 0.0]])
        self.assertEqual(self.call(term, env).item(), 0.0)

    def test_first_step_after_reset_is_masked_and_history_reseeded(self):
        env = make()
        term = self.cls(types.SimpleNamespace(params={}), env)
        env.step([[-0.9, 0.0]], [[0.0, 0.0]])
        self.call(term, env)
        term.reset([0])
        env.step([[-0.9, 0.0]], [[0.01, 0.0]])
        self.assertEqual(self.call(term, env).item(), 0.0)

    def test_value_is_computed_once_per_step(self):
        env = make()
        term = self.cls(types.SimpleNamespace(params={}), env)
        env.step([[-0.4, 0.0]], [[0.0, 0.0]])
        self.call(term, env)
        env.step([[-0.6, 0.0]], [[0.0, 0.0]])
        self.call(term, env)
        env.step([[-0.1, 0.0]], [[0.005, 0.0]])
        first = self.call(term, env).item()
        self.assertEqual(self.call(term, env).item(), first)
        self.assertGreater(first, 0.0)

    def test_cfg_term_is_inert_with_a_float_weight(self):
        text = CFG.read_text()
        block = text[text.index("feet_touchdown_descent = RewTerm("):]
        block = block[: block.index("    )\n") + 6]
        self.assertIn("weight=0.0,", block)
        self.assertIn('"deadband": 0.25', block)
        self.assertIn('"scale": 0.16', block)


if __name__ == "__main__":
    unittest.main()
