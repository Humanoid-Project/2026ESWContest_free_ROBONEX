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
    cfg = types.SimpleNamespace(
        ranges=types.SimpleNamespace(**{axis: ranges}),
        limit_ranges=types.SimpleNamespace(**{axis: limits}),
    )
    env = types.SimpleNamespace()
    env.command_manager = types.SimpleNamespace(get_term=lambda _: types.SimpleNamespace(cfg=cfg))
    env.reward_manager = types.SimpleNamespace(
        get_term_cfg=lambda _: types.SimpleNamespace(weight=weight),
        _episode_sums={term: torch.full((4,), reward * 20.0)},
    )
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


if __name__ == "__main__":
    unittest.main()
