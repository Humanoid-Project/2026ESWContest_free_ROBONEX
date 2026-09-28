import ast
import types
import unittest
from pathlib import Path

import torch

TASK = Path(__file__).resolve().parents[1] / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"


def load(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class SteppingTermsTest(unittest.TestCase):
    def test_stand_still_is_gated_by_base_speed(self):
        ns = {"torch": torch, "ManagerBasedRLEnv": object, "Articulation": object,
              "SceneEntityCfg": lambda *a, **k: types.SimpleNamespace(name="robot")}
        load(TASK / "mdp/rewards.py", {"stand_still_airborne"}, ns)
        ns["_contact_mask"] = lambda env, cfg, threshold: torch.tensor([[True, False], [True, False], [True, True], [True, False]])
        env = types.SimpleNamespace()
        env.command_manager = types.SimpleNamespace(get_command=lambda _: torch.tensor([[0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0], [0.3, 0, 0]]))
        vel = torch.tensor([[0.05, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        env.scene = {"robot": types.SimpleNamespace(data=types.SimpleNamespace(root_lin_vel_b=vel))}
        gated = ns["stand_still_airborne"](env, sensor_cfg=None, speed_gate=0.15)
        self.assertEqual(gated.tolist(), [1.0, 0.0, 0.0, 0.0])
        ungated = ns["stand_still_airborne"](env, sensor_cfg=None)
        self.assertEqual(ungated.tolist(), [1.0, 1.0, 0.0, 0.0])

    def test_standing_push_only_touches_standing_envs(self):
        written = {}

        class Asset:
            data = types.SimpleNamespace(root_vel_w=torch.zeros(4, 6))

            def write_root_velocity_to_sim(self, vel, env_ids):
                written["ids"] = env_ids.tolist()
                written["vel"] = vel

        ns = {"torch": torch, "Articulation": object, "SceneEntityCfg": lambda *a, **k: types.SimpleNamespace(name="robot")}
        load(TASK / "mdp/events.py", {"push_standing_by_setting_velocity"}, ns)
        env = types.SimpleNamespace(scene={"robot": Asset()})
        env.command_manager = types.SimpleNamespace(get_command=lambda _: torch.tensor([[0.0, 0, 0], [0.3, 0, 0], [0.0, 0, 0], [0.0, 0.2, 0]]))
        ns["push_standing_by_setting_velocity"](env, torch.tensor([0, 1, 2, 3]), {"x": (-0.8, 0.8), "y": (-0.8, 0.8)})
        self.assertEqual(written["ids"], [0, 2])
        self.assertTrue(torch.all(written["vel"][:, :2].abs() <= 0.8))
        self.assertTrue(torch.all(written["vel"][:, 2:] == 0.0))


if __name__ == "__main__":
    unittest.main()
