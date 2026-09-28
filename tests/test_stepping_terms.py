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


class FakeEnv:
    def __init__(self, n=2):
        self.num_envs, self.device, self.step_dt = n, "cpu", 0.02
        self.common_step_counter = 0
        self.episode_length_buf = torch.full((n,), 10)
        self.vel = torch.zeros(n, 2, 3)
        self.forces = torch.full((n, 8, 2, 3), 50.0)
        mass = torch.tensor([[15.0, 6.4]] * n)
        data = types.SimpleNamespace(default_mass=mass)
        self.robot = types.SimpleNamespace(data=data)
        self.scene = types.SimpleNamespace()
        self.command = torch.zeros(n, 3)
        self.command_manager = types.SimpleNamespace(get_command=lambda _: self.command)

    def step(self):
        self.common_step_counter += 1
        self.robot.data.body_com_lin_vel_w = self.vel
        self.scene_obj.sensors["contact_forces"].data.net_forces_w_history = self.forces


class SteppingTermsTest(unittest.TestCase):
    def setUp(self):
        self.ns = {"torch": torch, "ManagerBasedRLEnv": object, "Articulation": object, "ContactSensor": object,
                   "SceneEntityCfg": lambda *a, **k: types.SimpleNamespace(name="robot")}
        load(TASK / "mdp/rewards.py", {"_recovery_quiet", "stand_still_airborne", "standing_joint_load_l1"}, self.ns)
        env = FakeEnv()
        sensor = types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w_history=env.forces))
        env.scene_obj = types.SimpleNamespace(sensors={"contact_forces": sensor})
        env.robot.data.body_com_lin_vel_w = env.vel
        env.scene = _Scene(env.robot, env.scene_obj.sensors)
        self.env = env

    def quiet(self):
        return self.ns["_recovery_quiet"](self.env, 0.20, 0.08, 0.2).tolist()

    def test_gate_opens_on_com_speed_and_latches_until_settled(self):
        env = self.env
        env.step(); self.assertEqual(self.quiet(), [1.0, 1.0])
        env.vel[0, :, 0] = 0.3; env.step(); self.assertEqual(self.quiet(), [0.0, 1.0])
        env.vel[0, :, 0] = 0.1; env.step(); self.assertEqual(self.quiet(), [0.0, 1.0])
        env.vel[0, :, 0] = 0.0
        for _ in range(9):
            env.step(); self.assertEqual(self.quiet()[0], 0.0)
        env.step(); self.assertEqual(self.quiet(), [1.0, 1.0])

    def test_single_support_keeps_the_gate_open(self):
        env = self.env
        env.vel[0, :, 0] = 0.3; env.step(); self.quiet()
        env.vel[0, :, 0] = 0.0; env.forces[0, :, 1, :] = 0.0
        for _ in range(20):
            env.step(); self.assertEqual(self.quiet()[0], 0.0)

    def test_gate_is_computed_once_per_step(self):
        env = self.env
        env.vel[0, :, 0] = 0.3; env.step(); self.quiet()
        env.vel[0, :, 0] = 0.0
        for _ in range(10):
            env.step(); self.quiet(); self.quiet()
        self.assertEqual(self.quiet()[0], 1.0)

    def test_stand_still_penalty_is_gated(self):
        env = self.env
        self.ns["_contact_mask"] = lambda env, cfg, threshold: torch.tensor([[True, False], [True, False]])
        env.vel[0, :, 0] = 0.3; env.step()
        out = self.ns["stand_still_airborne"](env, sensor_cfg=None, recovery_gate=(0.20, 0.08, 0.2))
        self.assertEqual(out.tolist(), [0.0, 1.0])
        self.assertEqual(self.ns["stand_still_airborne"](env, sensor_cfg=None).tolist(), [1.0, 1.0])

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


class _Scene(dict):
    def __init__(self, robot, sensors):
        super().__init__(robot=robot)
        self.sensors = sensors


if __name__ == "__main__":
    unittest.main()
