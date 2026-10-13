import ast
import math
import types
import unittest
from pathlib import Path

import torch

OBS = Path(__file__).resolve().parents[1] / (
    "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking/mdp/observations.py"
)


class Base:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env
        self.num_envs = env.num_envs
        self.device = "cpu"


def load():
    node = next(n for n in ast.parse(OBS.read_text()).body if getattr(n, "name", None) == "delayed_imu_projected_gravity_biased")
    ns = {"torch": torch, "ManagerTermBase": Base, "ManagerBasedRLEnv": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "observations", "exec"), ns)
    return ns["delayed_imu_projected_gravity_biased"]


def make(n, gravity, **params):
    imu = types.SimpleNamespace(delayed=lambda: (torch.zeros(n, 3), gravity))
    env = types.SimpleNamespace(num_envs=n, device="cpu", scene={"imu": imu}, common_step_counter=0, step_dt=0.02)
    asset_cfg = types.SimpleNamespace(name="imu")
    cfg = types.SimpleNamespace(params={"asset_cfg": asset_cfg, **params})
    return load()(cfg, env), env, asset_cfg


class GravityBiasTest(unittest.TestCase):
    def test_inert_defaults_return_the_same_tensor(self):
        g = torch.tensor([[0.1, -0.2, -0.97]] * 4)
        term, env, asset_cfg = make(4, g)
        out = term(env, asset_cfg)
        self.assertIs(out, g)

    def test_offset_rotates_by_the_drawn_angles_and_keeps_unit_length(self):
        g = torch.tensor([[0.0, 0.0, -1.0]] * 3)
        term, env, asset_cfg = make(3, g, offset_deg=3.0)
        term._offset = torch.tensor([[math.radians(2.0), 0.0], [0.0, math.radians(-3.0)], [0.0, 0.0]])
        out = term(env, asset_cfg, offset_deg=3.0)
        self.assertTrue(torch.allclose(out.norm(dim=1), torch.ones(3), atol=1e-6))
        self.assertAlmostEqual(math.degrees(math.asin(out[0, 1])), 2.0, places=4)
        self.assertAlmostEqual(math.degrees(math.asin(out[1, 0])), 3.0, places=4)
        self.assertTrue(torch.allclose(out[2], g[2]))

    def test_reset_draws_offsets_inside_the_range(self):
        g = torch.tensor([[0.0, 0.0, -1.0]] * 2000)
        term, env, asset_cfg = make(2000, g, offset_deg=3.0)
        term.reset()
        self.assertLessEqual(term._offset.abs().max().item(), math.radians(3.0) + 1e-9)
        self.assertGreater(term._offset.abs().max().item(), math.radians(2.5))
        term.reset(torch.tensor([0, 1]))
        self.assertEqual(term._offset.shape, (2000, 2))

    def test_ou_keeps_its_stationary_spread_and_moves_once_per_step(self):
        torch.manual_seed(0)
        g = torch.tensor([[0.0, 0.0, -1.0]] * 4000)
        term, env, asset_cfg = make(4000, g, ou_sigma_deg=0.5, ou_tau_s=2.0)
        before = term._ou.clone()
        term(env, asset_cfg)
        moved = term._ou.clone()
        term(env, asset_cfg)
        self.assertTrue(torch.equal(moved, term._ou))
        self.assertFalse(torch.equal(before, moved))
        for step in range(1, 300):
            env.common_step_counter = step
            term(env, asset_cfg)
        self.assertAlmostEqual(math.degrees(term._ou.std().item()), 0.5, delta=0.03)
        lag = torch.exp(torch.tensor(-0.02 / 2.0)).item()
        self.assertGreater(lag, 0.98)

    def test_clip_bounds_the_total_bias(self):
        g = torch.tensor([[0.0, 0.0, -1.0]])
        term, env, asset_cfg = make(1, g, offset_deg=3.0, clip_deg=1.0)
        term._offset = torch.tensor([[math.radians(3.0), math.radians(-3.0)]])
        self.assertTrue(torch.allclose(term.bias(), torch.tensor([[math.radians(1.0), math.radians(-1.0)]])))

    def test_invalid_parameters_are_refused(self):
        g = torch.tensor([[0.0, 0.0, -1.0]])
        with self.assertRaises(ValueError):
            make(1, g, ou_tau_s=0.0)
        with self.assertRaises(ValueError):
            make(1, g, offset_deg=-1.0)


if __name__ == "__main__":
    unittest.main()
