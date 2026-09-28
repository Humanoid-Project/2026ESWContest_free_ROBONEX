import ast
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
    node = next(n for n in ast.parse(OBS.read_text()).body if getattr(n, "name", None) == "delayed_joint_state")
    ns = {"torch": torch, "ManagerTermBase": Base, "ManagerBasedRLEnv": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "observations", "exec"), ns)
    return ns["delayed_joint_state"]


class DelayTest(unittest.TestCase):
    def test_delay_zero_and_one_and_reset(self):
        cls = load()
        data = types.SimpleNamespace(joint_pos=torch.zeros(2, 1), default_joint_pos=torch.zeros(2, 1))
        env = types.SimpleNamespace(num_envs=2, device="cpu", scene={"robot": types.SimpleNamespace(data=data)},
                                    common_step_counter=0, episode_length_buf=torch.zeros(2, dtype=torch.long))
        asset_cfg = types.SimpleNamespace(name="robot", joint_ids=slice(None))
        cfg = types.SimpleNamespace(params={"asset_cfg": asset_cfg, "max_delay_steps": 1})
        term = cls(cfg, env)
        term._delay = torch.tensor([0, 1])
        outputs = []
        for step in range(1, 5):
            env.common_step_counter = step
            env.episode_length_buf = torch.full((2,), step, dtype=torch.long)
            data.joint_pos = torch.full((2, 1), float(step))
            outputs.append(term(env, asset_cfg).squeeze(-1).tolist())
        self.assertEqual(outputs[1:], [[2.0, 1.0], [3.0, 2.0], [4.0, 3.0]])
        env.common_step_counter = 5
        env.episode_length_buf = torch.tensor([0, 5])
        data.joint_pos = torch.tensor([[10.0], [5.0]])
        self.assertEqual(term(env, asset_cfg).squeeze(-1).tolist(), [10.0, 4.0])
        env.common_step_counter = 6
        env.episode_length_buf = torch.tensor([1, 6])
        data.joint_pos = torch.tensor([[11.0], [6.0]])
        self.assertEqual(term(env, asset_cfg).squeeze(-1).tolist(), [11.0, 5.0])
        term._delay = torch.tensor([1, 0])
        env.common_step_counter = 7
        data.joint_pos = torch.tensor([[12.0], [7.0]])
        self.assertEqual(term(env, asset_cfg).squeeze(-1).tolist(), [11.0, 7.0])


class SharedDelayTest(unittest.TestCase):
    def test_position_and_velocity_share_one_delay(self):
        cls = load()
        data = types.SimpleNamespace(joint_pos=torch.zeros(64, 1), default_joint_pos=torch.zeros(64, 1),
                                     joint_vel=torch.zeros(64, 1), default_joint_vel=torch.zeros(64, 1))
        env = types.SimpleNamespace(num_envs=64, device="cpu", scene={"robot": types.SimpleNamespace(data=data)},
                                    common_step_counter=0, episode_length_buf=torch.zeros(64, dtype=torch.long))
        asset_cfg = types.SimpleNamespace(name="robot", joint_ids=slice(None))
        pos = cls(types.SimpleNamespace(params={"asset_cfg": asset_cfg, "max_delay_steps": 1, "field": "pos"}), env)
        vel = cls(types.SimpleNamespace(params={"asset_cfg": asset_cfg, "max_delay_steps": 1, "field": "vel"}), env)
        self.assertTrue(torch.equal(pos._delay, vel._delay))
        pos.reset(torch.arange(0, 32))
        vel.reset(torch.arange(0, 32))
        self.assertTrue(torch.equal(pos._delay, vel._delay))
        self.assertGreater(int(pos._delay.sum()), 0)
        self.assertLess(int(pos._delay.sum()), 64)


if __name__ == "__main__":
    unittest.main()
