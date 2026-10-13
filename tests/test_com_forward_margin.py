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


def quat_yaw(yaw):
    return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def apply_inverse(q, v):
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    yaw = 2 * torch.atan2(z, w).squeeze(-1)
    c, s = torch.cos(yaw), torch.sin(yaw)
    out = v.clone()
    out[:, 0] = c * v[:, 0] + s * v[:, 1]
    out[:, 1] = -s * v[:, 0] + c * v[:, 1]
    return out


def load(gate=None):
    tree = ast.parse(REWARDS.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in ("com_forward_margin_l2", "_saturate")]
    ns = {"torch": torch, "math": math, "ManagerTermBase": Base, "ManagerBasedRLEnv": object, "Articulation": object,
          "SceneEntityCfg": lambda *a, **k: types.SimpleNamespace(name="robot", body_ids=[1, 2]),
          "quat_apply_inverse": apply_inverse, "yaw_quat": lambda q: q,
          "_recovery_quiet": (lambda env, *a: gate) if gate is not None else (lambda env, *a: torch.ones(env.num_envs))}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "rewards", "exec"), ns)
    return ns["com_forward_margin_l2"]


class Env:
    def __init__(self, n, com_x, yaw=0.0):
        self.num_envs = n
        self.device = "cpu"
        self.common_step_counter = 0
        self.step_dt = 0.02
        self.episode_length_buf = torch.full((n,), 10)
        mass = torch.tensor([[2.0, 1.0, 1.0]]).repeat(n, 1)
        com = torch.zeros(n, 3, 3)
        feet = torch.zeros(n, 3, 3)
        feet[:, 1, :] = torch.tensor([0.0, 0.1, 0.0])
        feet[:, 2, :] = torch.tensor([0.0, -0.1, 0.0])
        self.data = types.SimpleNamespace(default_mass=mass, body_com_pos_w=com, body_pos_w=feet,
                                          root_quat_w=quat_yaw(yaw).repeat(n, 1))
        self.scene = {"robot": types.SimpleNamespace(data=self.data)}
        self.set(com_x, yaw)

    def set(self, com_x, yaw=0.0):
        c, s = math.cos(yaw), math.sin(yaw)
        self.data.body_com_pos_w[:, :, 0] = c * com_x
        self.data.body_com_pos_w[:, :, 1] = s * com_x
        self.data.body_com_pos_w[:, :, 2] = 0.6
        self.data.body_pos_w[:, 1:, 0] = 0.0
        self.data.root_quat_w = quat_yaw(yaw).repeat(self.num_envs, 1)


def cfg():
    return types.SimpleNamespace(params={})


class ComForwardMarginTest(unittest.TestCase):
    def test_margin_is_the_yaw_frame_com_ahead_of_the_foot_midpoint(self):
        cls = load()
        for yaw in (0.0, 1.2, -2.5):
            env = Env(2, 0.05, yaw)
            term = cls(cfg(), env)
            dx = term.margin(env, env.scene["robot"] and types.SimpleNamespace(name="robot"),
                             types.SimpleNamespace(body_ids=[1, 2]))
            self.assertTrue(torch.allclose(dx, torch.full((2,), 0.05), atol=1e-6), (yaw, dx))

    def test_no_penalty_above_target_and_saturating_below(self):
        cls = load()
        env = Env(1, 0.08)
        term = cls(cfg(), env)
        self.assertEqual(term(env).item(), 0.0)
        env = Env(1, 0.045)
        term = cls(cfg(), env)
        self.assertAlmostEqual(term(env).item(), 1 - math.exp(-1.0), places=5)
        env = Env(1, -0.2)
        term = cls(cfg(), env)
        self.assertGreater(term(env).item(), 0.99)

    def test_ema_follows_with_the_gait_period_and_resets_on_fresh_episodes(self):
        cls = load()
        env = Env(1, 0.10)
        term = cls(cfg(), env)
        term(env)
        env.set(0.0)
        for k in range(1, 41):
            env.common_step_counter = k
            term(env)
        expected = 0.10 * (1 - 0.02 / 0.82) ** 40
        self.assertAlmostEqual(term._ema.item(), expected, places=5)
        env.common_step_counter = 41
        env.episode_length_buf = torch.tensor([1])
        term(env)
        self.assertAlmostEqual(term._ema.item(), 0.0, places=6)

    def test_recovery_gate_switches_the_penalty_off(self):
        cls = load(gate=torch.zeros(1))
        env = Env(1, 0.0)
        term = cls(cfg(), env)
        self.assertEqual(term(env, recovery_gate=(0.2, 0.08, 0.2, 1.5)).item(), 0.0)

    def test_cfg_term_is_inert_by_default(self):
        tree = ast.parse(ENV_CFG.read_text())
        rewards = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg")
        term = next(n for n in rewards.body if isinstance(n, ast.Assign) and n.targets[0].id == "com_forward_margin")
        kw = {k.arg: k.value for k in term.value.keywords}
        self.assertEqual(ast.unparse(kw["func"]), "mdp.com_forward_margin_l2")
        self.assertEqual(ast.literal_eval(kw["weight"]), 0.0)


if __name__ == "__main__":
    unittest.main()
