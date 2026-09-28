import ast
import math
import types
import unittest
from pathlib import Path

import torch

REWARDS = Path(__file__).resolve().parents[1] / (
    "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking/mdp/rewards.py"
)


def load(names, namespace):
    tree = ast.parse(REWARDS.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "rewards", "exec"), namespace)
    return namespace


def make_env(width, command, torque=None):
    feet = torch.zeros(len(width), 2, 3)
    feet[:, 0, 1] = torch.tensor(width) / 2
    feet[:, 1, 1] = -torch.tensor(width) / 2
    env = types.SimpleNamespace()
    env.command_manager = types.SimpleNamespace(get_command=lambda _: torch.tensor(command, dtype=torch.float32))
    asset = types.SimpleNamespace(data=types.SimpleNamespace(applied_torque=torch.tensor(torque or [[0.0, 0.0]] * len(width))))
    env.scene = {"robot": asset}
    return env, feet


class StandingTermsTest(unittest.TestCase):
    def setUp(self):
        ns = {"torch": torch, "ManagerBasedRLEnv": object, "SceneEntityCfg": object, "Articulation": object}
        load({"_bounded_square", "_saturate", "feet_stance_width_l2", "standing_joint_load_l1"}, ns)
        self.ns = ns

    def width(self, width, command, **params):
        env, feet = make_env(width, command)
        self.ns["_body_pos_b"] = lambda env, cfg: feet
        return self.ns["feet_stance_width_l2"](env, target_width=0.269, asset_cfg=None, standing_width=0.303, **params)

    def test_window_is_free_inside_and_walking_is_unchanged(self):
        stand, walk = [0.0, 0.0, 0.0], [0.3, 0.0, 0.0]
        widths = [0.26, 0.28, 0.30, 0.24, 0.32]
        new = self.width(widths, [stand] * 5, standing_window=(0.25, 0.30), standing_scale=0.0009)
        self.assertTrue(torch.allclose(new[:3], torch.zeros(3)))
        self.assertAlmostEqual(new[3].item(), 1 - math.exp(-(0.01 ** 2) / 0.0009), places=5)
        self.assertAlmostEqual(new[4].item(), 1 - math.exp(-(0.02 ** 2) / 0.0009), places=5)
        old_walk = self.width(widths, [walk] * 5)
        new_walk = self.width(widths, [walk] * 5, standing_window=(0.25, 0.30), standing_scale=0.0009)
        self.assertTrue(torch.allclose(old_walk, new_walk))

    def test_without_window_the_old_standing_target_applies(self):
        out = self.width([0.303, 0.269], [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
        self.assertTrue(torch.allclose(out, torch.zeros(2)))

    def test_hip_roll_load_only_when_standing(self):
        env, _ = make_env([0.27, 0.27], [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], torque=[[10.4, -10.4], [10.4, -10.4]])
        cfg = types.SimpleNamespace(name="robot", joint_ids=slice(None))
        out = self.ns["standing_joint_load_l1"](env, asset_cfg=cfg)
        self.assertAlmostEqual(out[0].item(), 20.8 / 20.0, places=5)
        self.assertEqual(out[1].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
