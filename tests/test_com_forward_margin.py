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
        self.command = torch.zeros(n, 3)
        self.command_manager = types.SimpleNamespace(get_command=lambda name: self.command)
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

    def advance(self, term, env, steps, **kw):
        out = None
        for _ in range(steps):
            env.common_step_counter += 1
            out = term(env, **kw)
        return out

    def test_cap_mode_keeps_the_floor_while_moving_and_through_the_settle_window(self):
        cls = load()
        env = Env(1, 0.040)
        env.command = torch.tensor([[0.3, 0.0, 0.0]])
        term = cls(cfg(), env)
        kw = dict(target=0.055, standing="cap", standing_cap=0.060, settle_s=2.5)
        floor = self.advance(term, env, 5, **kw).item()
        self.assertAlmostEqual(floor, 1 - math.exp(-(0.015 ** 2) / 0.02 ** 2), places=5)
        env.command = torch.zeros(1, 3)
        self.assertAlmostEqual(self.advance(term, env, 124, **kw).item(), floor, places=5)
        self.assertEqual(self.advance(term, env, 1, **kw).item(), 0.0)

    def test_cap_mode_penalises_standing_above_the_cap_only(self):
        cls = load()
        env = Env(1, 0.080)
        term = cls(cfg(), env)
        kw = dict(target=0.055, standing="cap", standing_cap=0.060, settle_s=2.5)
        out = self.advance(term, env, 130, **kw).item()
        self.assertAlmostEqual(out, 1 - math.exp(-(0.020 ** 2) / 0.02 ** 2), places=4)
        env.set(0.058)
        term = cls(cfg(), env)
        self.assertEqual(self.advance(term, env, 130, **kw).item(), 0.0)

    def test_off_mode_drops_the_standing_penalty_and_a_command_restarts_the_floor(self):
        cls = load()
        env = Env(1, 0.020)
        term = cls(cfg(), env)
        kw = dict(target=0.055, standing="off", settle_s=2.5)
        self.assertEqual(self.advance(term, env, 130, **kw).item(), 0.0)
        env.command = torch.tensor([[0.0, 0.2, 0.0]])
        self.assertGreater(self.advance(term, env, 1, **kw).item(), 0.9)

    def test_fresh_episode_restarts_the_settle_timer(self):
        cls = load()
        env = Env(1, 0.080)
        term = cls(cfg(), env)
        kw = dict(target=0.055, standing="cap", standing_cap=0.060, settle_s=2.5)
        self.assertGreater(self.advance(term, env, 130, **kw).item(), 0.0)
        env.episode_length_buf = torch.tensor([1])
        self.assertEqual(self.advance(term, env, 1, **kw).item(), 0.0)

    def test_the_settle_timer_runs_in_every_mode(self):
        cls = load()
        env = Env(1, 0.080)
        term = cls(cfg(), env)
        kw = dict(target=0.055, standing_cap=0.060, settle_s=2.5)
        self.advance(term, env, 130, standing="cap", **kw)
        env.command = torch.tensor([[0.3, 0.0, 0.0]])
        self.advance(term, env, 3, standing="floor", **kw)
        env.command = torch.zeros(1, 3)
        self.assertEqual(self.advance(term, env, 1, standing="cap", **kw).item(), 0.0)

    def test_floor_mode_is_the_w132_term_and_unknown_modes_are_refused(self):
        cls = load()
        env = Env(1, 0.045)
        term = cls(cfg(), env)
        self.assertAlmostEqual(self.advance(term, env, 130).item(), 1 - math.exp(-1.0), places=5)
        with self.assertRaises(ValueError):
            term(env, standing="hold")

    def test_push_grace_zeroes_the_penalty_for_its_window_after_a_push(self):
        cls = load()
        env = Env(1, 0.020)
        env.command = torch.tensor([[0.3, 0.0, 0.0]])
        term = cls(cfg(), env)
        kw = dict(target=0.055, standing="cap", standing_cap=0.060, settle_s=2.5, push_grace_s=1.7)
        floor = self.advance(term, env, 5, **kw).item()
        self.assertGreater(floor, 0.9)
        env._sway_last_push_step = torch.tensor([env.common_step_counter])
        self.assertEqual(self.advance(term, env, 85, **kw).item(), 0.0)
        self.assertAlmostEqual(self.advance(term, env, 1, **kw).item(), floor, places=5)

    def test_push_grace_is_inert_at_zero_and_without_a_marker(self):
        cls = load()
        env = Env(1, 0.020)
        env.command = torch.tensor([[0.3, 0.0, 0.0]])
        term = cls(cfg(), env)
        kw = dict(target=0.055, push_grace_s=1.7)
        self.assertGreater(self.advance(term, env, 3, **kw).item(), 0.9)
        env._sway_last_push_step = torch.tensor([env.common_step_counter])
        self.assertGreater(self.advance(term, env, 1, target=0.055, push_grace_s=0.0).item(), 0.9)
        with self.assertRaises(ValueError):
            term(env, push_grace_s=-1.0)

    def test_standing_floor_default_none_matches_cap_mode(self):
        cls = load()
        kw = dict(target=0.055, standing="cap", standing_cap=0.060, settle_s=2.5)
        for x in (0.030, 0.050, 0.070):
            env = Env(1, x)
            a = self.advance(cls(cfg(), env), env, 130, **kw).item()
            env = Env(1, x)
            b = self.advance(cls(cfg(), env), env, 130, standing_floor=None, **kw).item()
            self.assertEqual(a, b)
        env = Env(1, 0.030)
        self.assertEqual(self.advance(cls(cfg(), env), env, 130, **kw).item(), 0.0)

    def test_standing_floor_zero_matches_none(self):
        cls = load()
        kw = dict(target=0.055, standing="cap", standing_cap=0.060, settle_s=2.5)
        for x in (-0.020, 0.030, 0.070):
            env = Env(1, x)
            a = self.advance(cls(cfg(), env), env, 130, standing_floor=None, **kw).item()
            env = Env(1, x)
            b = self.advance(cls(cfg(), env), env, 130, standing_floor=0.0, **kw).item()
            self.assertEqual(a, b, x)
        env = Env(1, 0.050)
        cls(cfg(), env)(env, standing="cap", standing_cap=0.0, standing_floor=0.0)

    def test_standing_floor_prices_settled_standing_below_the_floor(self):
        cls = load()
        kw = dict(target=0.055, standing="cap", standing_cap=0.060, settle_s=2.5, standing_floor=0.045)
        env = Env(1, 0.030)
        self.assertAlmostEqual(self.advance(cls(cfg(), env), env, 130, **kw).item(),
                               1 - math.exp(-(0.015 ** 2) / 0.02 ** 2), places=4)
        env = Env(1, 0.050)
        self.assertEqual(self.advance(cls(cfg(), env), env, 130, **kw).item(), 0.0)
        env = Env(1, 0.070)
        self.assertAlmostEqual(self.advance(cls(cfg(), env), env, 130, **kw).item(),
                               1 - math.exp(-(0.010 ** 2) / 0.02 ** 2), places=4)

    def test_standing_floor_leaves_the_settle_window_on_the_walking_floor(self):
        cls = load()
        env = Env(1, 0.030)
        env.command = torch.tensor([[0.3, 0.0, 0.0]])
        term = cls(cfg(), env)
        kw = dict(target=0.055, standing="cap", standing_cap=0.060, settle_s=2.5, standing_floor=0.045)
        walking = 1 - math.exp(-(0.025 ** 2) / 0.02 ** 2)
        self.assertAlmostEqual(self.advance(term, env, 5, **kw).item(), walking, places=5)
        env.command = torch.zeros(1, 3)
        self.assertAlmostEqual(self.advance(term, env, 124, **kw).item(), walking, places=5)
        self.assertAlmostEqual(self.advance(term, env, 1, **kw).item(), 1 - math.exp(-(0.015 ** 2) / 0.02 ** 2), places=5)

    def test_standing_floor_with_off_mode_and_ignored_in_floor_mode(self):
        cls = load()
        env = Env(1, 0.030)
        out = self.advance(cls(cfg(), env), env, 130, target=0.055, standing="off", settle_s=2.5, standing_floor=0.045).item()
        self.assertAlmostEqual(out, 1 - math.exp(-(0.015 ** 2) / 0.02 ** 2), places=4)
        env = Env(1, 0.080)
        self.assertEqual(self.advance(cls(cfg(), env), env, 130, target=0.055, standing="off", settle_s=2.5,
                                      standing_floor=0.045).item(), 0.0)
        env = Env(1, 0.045)
        a = self.advance(cls(cfg(), env), env, 130).item()
        env = Env(1, 0.045)
        self.assertEqual(self.advance(cls(cfg(), env), env, 130, standing_floor=0.030).item(), a)

    def test_standing_floor_is_validated(self):
        cls = load()
        env = Env(1, 0.050)
        term = cls(cfg(), env)
        for bad in (float("nan"), float("inf"), -0.01):
            with self.assertRaises(ValueError):
                term(env, standing="cap", standing_floor=bad)
        with self.assertRaises(ValueError):
            term(env, standing="cap", standing_cap=0.060, standing_floor=0.060)
        term(env, standing="off", standing_cap=0.060, standing_floor=0.070)

    def test_standing_floor_is_gated_like_the_other_branches(self):
        cls = load(gate=torch.zeros(1))
        env = Env(1, 0.030)
        kw = dict(target=0.055, standing="cap", standing_cap=0.060, settle_s=2.5, standing_floor=0.045)
        self.assertEqual(self.advance(cls(cfg(), env), env, 130, recovery_gate=(0.2, 0.08, 0.2, 1.5), **kw).item(), 0.0)
        cls = load()
        env = Env(1, 0.030)
        term = cls(cfg(), env)
        self.assertGreater(self.advance(term, env, 130, push_grace_s=1.7, **kw).item(), 0.4)
        env._sway_last_push_step = torch.tensor([env.common_step_counter])
        self.assertEqual(self.advance(term, env, 1, push_grace_s=1.7, **kw).item(), 0.0)

    def test_cfg_term_is_inert_by_default(self):
        tree = ast.parse(ENV_CFG.read_text())
        rewards = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg")
        term = next(n for n in rewards.body if isinstance(n, ast.Assign) and n.targets[0].id == "com_forward_margin")
        kw = {k.arg: k.value for k in term.value.keywords}
        self.assertEqual(ast.unparse(kw["func"]), "mdp.com_forward_margin_l2")
        self.assertEqual(ast.literal_eval(kw["weight"]), 0.0)
        params = {ast.literal_eval(k): v for k, v in zip(kw["params"].keys, kw["params"].values)}
        self.assertNotIn("recovery_gate", params)
        literal = {k: ast.literal_eval(v) for k, v in params.items() if k not in ("asset_cfg", "foot_cfg")}
        self.assertEqual(literal, {"target": 0.065, "scale": 0.02, "tau_s": 0.8, "standing": "floor", "standing_cap": 0.060,
                                   "standing_floor": 0.0, "settle_s": 2.5, "command_name": "base_velocity", "command_deadband": 0.05,
                                   "push_grace_s": 0.0})


if __name__ == "__main__":
    unittest.main()
