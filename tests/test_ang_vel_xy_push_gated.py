import ast
import math
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REWARDS = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking/mdp/rewards.py"
CFG = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking/robonex_walking_v2_env_cfg.py"


class Base:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


def load():
    tree = ast.parse(REWARDS.read_text())
    names = {"_bounded_square", "_saturate", "_mark_sway_push", "push_robot_with_sway_marker",
             "push_standing_with_sway_marker", "ang_vel_xy_push_gated"}
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    ns = {"torch": torch, "math": math, "ManagerTermBase": Base,
          "SceneEntityCfg": lambda name: types.SimpleNamespace(name=name)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "rewards", "exec"), ns)
    return ns


class Env:
    def __init__(self):
        self.device = "cpu"
        self.num_envs = 3
        self.step_dt = 0.02
        self.common_step_counter = 10
        self.commands = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.command_manager = types.SimpleNamespace(get_command=lambda name: self.commands)
        self.rates = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 4.0, 3.0]])
        self.scene = {"robot": types.SimpleNamespace(data=types.SimpleNamespace(root_ang_vel_b=self.rates))}


class AngVelXYPushGatedTest(unittest.TestCase):
    def setUp(self):
        self.ns = load()
        self.env = Env()
        self.term = self.ns["ang_vel_xy_push_gated"](types.SimpleNamespace(params={}), self.env)

    def test_both_axes_and_repeated_reads(self):
        expected = 1 - torch.exp(-(self.env.rates[:, 0]**2 + self.env.rates[:, 1]**2) / 0.5)
        torch.testing.assert_close(self.term(self.env), expected)
        torch.testing.assert_close(self.term(self.env), expected)
        self.assertGreater(float(self.term(self.env)[2]), 0.99)

    def test_push_masks_only_selected_envs_for_exactly_75_future_rewards(self):
        before = self.term(self.env).clone()
        self.ns["_mark_sway_push"](self.env, torch.tensor([0]))
        for step in range(11, 86):
            self.env.common_step_counter = step
            value = self.term(self.env)
            self.assertEqual(float(value[0]), 0.0)
            self.assertEqual(float(value[1]), float(before[1]))
        self.env.common_step_counter = 86
        torch.testing.assert_close(self.term(self.env), before)

    def test_repeat_push_extends_grace_and_reset_clears_only_selected_ids(self):
        self.ns["_mark_sway_push"](self.env, torch.tensor([0, 1]))
        self.env.common_step_counter = 60
        self.ns["_mark_sway_push"](self.env, torch.tensor([0]))
        self.env.common_step_counter = 86
        self.assertEqual(float(self.term(self.env)[0]), 0.0)
        self.assertGreater(float(self.term(self.env)[1]), 0.0)
        self.term.reset(torch.tensor([0]))
        self.assertGreater(float(self.term(self.env)[0]), 0.0)
        self.ns["_mark_sway_push"](self.env, torch.tensor([0, 1]))
        self.term.reset()
        self.assertTrue(torch.all(self.env._sway_last_push_step == -1))

    def test_zero_grace_and_nonfinite_input(self):
        self.ns["_mark_sway_push"](self.env, torch.tensor([0]))
        self.assertGreater(float(self.term(self.env, recovery_s=0)[0]), 0.0)
        self.term.reset()
        self.env.rates[:, 0] = torch.tensor([float('nan'), float('inf'), -float('inf')])
        self.env.rates[:, 1] = 0.0
        self.assertTrue(torch.all(self.term(self.env) == 1.0))

    def test_invalid_parameters(self):
        for scale in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.term(self.env, scale=scale)
        for recovery_s in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.term(self.env, recovery_s=recovery_s)

    def test_wrappers_forward_original_calls_and_mark_actual_standing_ids(self):
        calls = []
        self.ns["_push_by_setting_velocity"] = lambda *args: calls.append(args)
        self.ns["_push_standing_by_setting_velocity"] = lambda *args: calls.append(args)
        ids = torch.tensor([0, 1])
        ranges = {"x": (-0.5, 0.5)}
        self.ns["push_robot_with_sway_marker"](self.env, ids, ranges)
        self.assertIs(calls[-1][1], ids)
        self.assertIs(calls[-1][2], ranges)
        self.assertEqual(self.env._sway_last_push_step.tolist(), [10, 10, -1])
        self.term.reset()
        self.ns["push_standing_with_sway_marker"](self.env, ids, ranges)
        self.assertIs(calls[-1][1], ids)
        self.assertEqual(self.env._sway_last_push_step.tolist(), [10, -1, -1])
        self.term.reset()
        self.ns["push_standing_with_sway_marker"](self.env, torch.tensor([1]), ranges)
        self.assertEqual(self.env._sway_last_push_step.tolist(), [-1, -1, -1])

    def test_event_before_reward_construction_retains_marker(self):
        del self.env._sway_last_push_step
        self.ns["_mark_sway_push"](self.env, torch.tensor([1]))
        term = self.ns["ang_vel_xy_push_gated"](types.SimpleNamespace(params={}), self.env)
        self.env.common_step_counter += 1
        self.assertEqual(float(term(self.env)[1]), 0.0)

    def test_cfg_is_default_off_with_both_push_hooks_and_original_baseline(self):
        tree = ast.parse(CFG.read_text())
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        def assigned(cls, name):
            return next(n.value for n in classes[cls].body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        def kw(node, name):
            return next(k.value for k in node.keywords if k.arg == name)
        self.assertEqual(ast.literal_eval(kw(assigned('RewardsCfg', 'ang_vel_xy_push_gated'), 'weight')), 0.0)
        self.assertEqual(ast.literal_eval(kw(assigned('RewardsCfg', 'ang_vel_xy'), 'weight')), -0.1)
        self.assertEqual(kw(assigned('EventCfg', 'push_robot'), 'func').attr, 'push_robot_with_sway_marker')
        self.assertEqual(kw(assigned('EventCfg', 'push_standing'), 'func').attr, 'push_standing_with_sway_marker')


if __name__ == "__main__":
    unittest.main()
