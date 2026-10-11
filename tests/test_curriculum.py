import ast
import types
import unittest
from pathlib import Path

import torch

TASK = Path(__file__).resolve().parents[1] / (
    "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
)


def load(name):
    src = (TASK / "mdp/curriculums.py").read_text()
    node = next(n for n in ast.parse(src).body if getattr(n, "name", None) == name)
    namespace = {"torch": torch, "ManagerBasedRLEnv": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "curriculums", "exec"), namespace)
    return namespace[name]


def make_env(reward, step, ranges, limits=(0.0, 0.3), weight=3.0, axis="lin_vel_x",
             term="track_lin_vel_x"):
    span = {"lin_vel_x": ranges, "lin_vel_y": ranges, "ang_vel_z": ranges, axis: ranges}
    bound = {"lin_vel_x": limits, "lin_vel_y": limits, "ang_vel_z": limits, axis: limits}
    cfg = types.SimpleNamespace(
        ranges=types.SimpleNamespace(**span),
        limit_ranges=types.SimpleNamespace(**bound),
    )
    env = types.SimpleNamespace()
    env.command_manager = types.SimpleNamespace(get_term=lambda _: types.SimpleNamespace(cfg=cfg))
    env.reward_manager = types.SimpleNamespace(
        get_term_cfg=lambda _: types.SimpleNamespace(weight=weight),
        _episode_sums={term: torch.full((4,), reward * 20.0)},
    )
    env.episode_length_buf = torch.full((4,), 1000)
    env.max_episode_length_s = 20.0
    env.max_episode_length = 1000
    env.common_step_counter = step
    return env, cfg


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.levels = load("lin_vel_cmd_levels")

    def test_a_weak_tracker_does_not_widen_the_command(self):
        env, cfg = make_env(2.0, 1000, (0.1, 0.1))
        self.levels(env, torch.arange(4))
        self.assertEqual(cfg.ranges.lin_vel_x, (0.1, 0.1))

    def test_a_good_tracker_widens_once_per_episode_length(self):
        env, cfg = make_env(2.6, 1000, (0.1, 0.1))
        self.levels(env, torch.arange(4))
        self.assertEqual(cfg.ranges.lin_vel_x, (0.0, 0.2))
        # too soon: the step budget since the last widening has not elapsed
        env.common_step_counter = 1500
        self.levels(env, torch.arange(4))
        self.assertEqual(cfg.ranges.lin_vel_x, (0.0, 0.2))
        env.common_step_counter = 2000
        self.levels(env, torch.arange(4))
        self.assertEqual(cfg.ranges.lin_vel_x, (0.0, 0.3))

    def test_widening_does_not_need_an_exact_episode_boundary(self):
        # the curriculum manager only runs on reset batches, so a modulo gate
        # silently never fires once resets drift off the boundary
        env, cfg = make_env(2.6, 4321, (0.1, 0.1))
        self.levels(env, torch.arange(4))
        self.assertEqual(cfg.ranges.lin_vel_x, (0.0, 0.2))

    def test_the_command_never_passes_its_limit(self):
        env, cfg = make_env(2.6, 1000, (0.0, 0.3))
        for step in (1000, 2000, 3000, 4000):
            env.common_step_counter = step
            self.levels(env, torch.arange(4))
        self.assertEqual(cfg.ranges.lin_vel_x, (0.0, 0.3))

    def test_reverse_walking_is_reachable(self):
        env, cfg = make_env(2.6, 1000, (0.1, 0.1), limits=(-0.2, 0.4))
        for step in (1000, 2000, 3000, 4000, 5000):
            env.common_step_counter = step
            self.levels(env, torch.arange(4))
        self.assertEqual(cfg.ranges.lin_vel_x, (-0.2, 0.4))


class WindowGateTests(unittest.TestCase):
    def setUp(self):
        self.levels = load("lin_vel_cmd_levels")

    @staticmethod
    def batch(env, rewards, lengths=None):
        rewards = torch.tensor(rewards, dtype=torch.float32)
        env.reward_manager._episode_sums["track_lin_vel_x"] = rewards * env.max_episode_length_s
        env.episode_length_buf = torch.full((len(rewards),), 1000) if lengths is None else torch.tensor(lengths)
        return torch.arange(len(rewards))

    def run_batches(self, gate, batches, start=900):
        env, cfg = make_env(0.0, start, (0.1, 0.1))
        for step, rewards in enumerate(batches, start=start):
            env.common_step_counter = step
            self.levels(env, self.batch(env, rewards), gate=gate)
        return env, cfg

    def test_legacy_widens_on_the_first_lucky_batch_once_due(self):
        batches = [[2.0, 2.1]] * 150 + [[2.5, 2.6]] + [[2.0, 2.1]] * 10
        _, cfg = self.run_batches("legacy", batches)
        self.assertEqual(cfg.ranges.lin_vel_x, (0.0, 0.2))

    def test_window_ignores_one_lucky_batch(self):
        batches = [[2.0, 2.1]] * 150 + [[2.5, 2.6]] + [[2.0, 2.1]] * 10
        _, cfg = self.run_batches("window", batches)
        self.assertEqual(cfg.ranges.lin_vel_x, (0.1, 0.1))

    def test_window_widens_on_a_good_window_mean(self):
        batches = [[2.3, 2.6]] * 150
        _, cfg = self.run_batches("window", batches)
        self.assertEqual(cfg.ranges.lin_vel_x, (0.0, 0.2))

    def test_window_decides_once_and_restarts_after_a_failed_test(self):
        env, cfg = make_env(0.0, 1000, (0.1, 0.1))
        self.levels(env, self.batch(env, [2.0, 2.0]), gate="window")
        self.assertEqual(env._lin_vel_cmd_last_step, 1000)
        self.assertEqual(env._lin_vel_cmd_window.tolist(), [0.0, 0.0])
        env.common_step_counter = 1001
        self.levels(env, self.batch(env, [3.0, 3.0]), gate="window")
        self.assertEqual(cfg.ranges.lin_vel_x, (0.1, 0.1))
        env.common_step_counter = 2000
        self.levels(env, self.batch(env, [2.0, 2.0]), gate="window")
        self.assertEqual(cfg.ranges.lin_vel_x, (0.0, 0.2))
        self.assertEqual(env._lin_vel_cmd_last_step, 2000)

    def test_window_counts_falls_and_skips_zero_length_resets(self):
        env, cfg = make_env(0.0, 0, (0.1, 0.1))
        self.levels(env, self.batch(env, [0.0] * 8, lengths=[0] * 8), gate="window")
        self.assertEqual(env._lin_vel_cmd_window.tolist(), [0.0, 0.0])
        env.common_step_counter = 1000
        self.levels(env, self.batch(env, [3.0, 0.5], lengths=[1000, 120]), gate="window")
        self.assertEqual(cfg.ranges.lin_vel_x, (0.1, 0.1))

    def test_an_unknown_gate_is_refused(self):
        env, _ = make_env(2.6, 1000, (0.1, 0.1))
        with self.assertRaises(ValueError):
            self.levels(env, torch.arange(4), gate="windowed")

    def test_the_env_cfg_keeps_the_legacy_gate(self):
        src = (TASK / "robonex_walking_v2_env_cfg.py").read_text()
        self.assertIn("CurrTerm(func=mdp.lin_vel_cmd_levels)", src)
        node = next(n for n in ast.parse((TASK / "mdp/curriculums.py").read_text()).body
                    if getattr(n, "name", None) == "lin_vel_cmd_levels")
        names = [a.arg for a in node.args.args]
        gate_default = node.args.defaults[names.index("gate") - (len(names) - len(node.args.defaults))]
        self.assertEqual(ast.literal_eval(gate_default), "legacy")


if __name__ == "__main__":
    unittest.main()
