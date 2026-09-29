from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.sensors import Imu, ImuCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(
    env: ManagerBasedRLEnv,
    period: float,
    command_name: str | None = None,
    command_deadband: float = 0.05,
) -> torch.Tensor:
    """Where the gait clock is, as (sin, cos) of the cycle phase.

    ``feet_gait`` rewards a contact pattern that matches this clock, and the
    clock is a function of time since reset. Without it in the observation the
    policy cannot tell which phase it is in, so the reward is unlearnable and
    sits at chance. H1 feeds it; G1 comments it out and leans on a much wider
    velocity command instead.

    Zeroed on a standing command. Every reward that consumes the clock is already
    gated off at zero command, but the observation was not, so the clock kept
    turning and the policy kept answering it: measured standing still after a
    0.3 m/s walk, the knees swung 37.4 deg peak-to-peak at 1.22 Hz against a
    1.25 Hz clock, one foot re-touching every cycle, for 26 W of mechanical power
    while the feet were supposed to be planted. Holding the pair at zero removes
    the cue instead of asking a later penalty to out-shout it. The width stays 2,
    so the deploy observation contract is unchanged.
    """
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        moving = (torch.linalg.norm(command, dim=1) > command_deadband).unsqueeze(-1)
        phase = phase * moving
    return phase


class delayed_joint_state(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset_cfg = cfg.params["asset_cfg"]
        self._max_delay = int(cfg.params.get("max_delay_steps", 1))
        shared = getattr(env, "_joint_observation_delay", None)
        if shared is None:
            shared = torch.randint(0, self._max_delay + 1, (env.num_envs,), device=env.device)
            env._joint_observation_delay = shared
        self._delay = shared
        self._frames = None
        self._counter = None

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        self._delay[env_ids] = torch.randint(0, self._max_delay + 1, (len(env_ids),), device=self.device)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg, field: str = "pos", max_delay_steps: int = 1):
        asset = env.scene[asset_cfg.name]
        if field == "pos":
            current = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        else:
            current = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
        if self._frames is None:
            self._frames = current.unsqueeze(0).repeat(self._max_delay + 1, 1, 1)
        elif self._counter != env.common_step_counter:
            self._frames = torch.cat([current.unsqueeze(0), self._frames[:-1]], dim=0)
        else:
            self._frames[0] = current
        self._counter = env.common_step_counter
        fresh = env.episode_length_buf == 0
        if torch.any(fresh):
            self._frames[:, fresh] = current[fresh]
        return self._frames[self._delay, torch.arange(current.shape[0], device=current.device)]


class DelayedImu(Imu):
    def _initialize_impl(self):
        super()._initialize_impl()
        if not 0 <= self.cfg.min_delay_steps <= self.cfg.max_delay_steps:
            raise ValueError(
                f"imu delay steps must satisfy 0 <= min <= max, got {self.cfg.min_delay_steps}..{self.cfg.max_delay_steps}"
            )
        slots = self.cfg.max_delay_steps + 1
        self._ring_w = torch.zeros(slots, self._num_envs, 3, device=self._device)
        self._ring_g = torch.zeros(slots, self._num_envs, 3, device=self._device)
        self._ring_g[..., 2] = -1.0
        self._head = 0
        self._fresh = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        self._delay = self._draw(self._num_envs)
        self._envs = torch.arange(self._num_envs, device=self._device)

    def _draw(self, count):
        return torch.randint(self.cfg.min_delay_steps, self.cfg.max_delay_steps + 1, (count,), device=self._device)

    def update(self, dt, force_recompute=False):
        self._head = (self._head + 1) % self._ring_w.shape[0]
        super().update(dt, force_recompute)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
            count = self._num_envs
        else:
            count = len(env_ids)
        self._fresh[env_ids] = True
        self._delay[env_ids] = self._draw(count)

    def _update_buffers_impl(self, env_ids):
        super()._update_buffers_impl(env_ids)
        ang_vel = self._data.ang_vel_b[env_ids]
        gravity = self._data.projected_gravity_b[env_ids]
        self._ring_w[self._head, env_ids] = ang_vel
        self._ring_g[self._head, env_ids] = gravity
        fresh = torch.zeros_like(self._fresh)
        fresh[env_ids] = self._fresh[env_ids]
        if torch.any(fresh):
            self._ring_w[:, fresh] = self._data.ang_vel_b[fresh]
            self._ring_g[:, fresh] = self._data.projected_gravity_b[fresh]
            self._fresh[fresh] = False

    def delayed(self):
        self._update_outdated_buffers()
        index = (self._head - self._delay) % self._ring_w.shape[0]
        return self._ring_w[index, self._envs], self._ring_g[index, self._envs]


@configclass
class DelayedImuCfg(ImuCfg):
    class_type: type = DelayedImu
    min_delay_steps: int = 0
    max_delay_steps: int = 0


def delayed_imu_ang_vel(env: ManagerBasedRLEnv, asset_cfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].delayed()[0]


def delayed_imu_projected_gravity(env: ManagerBasedRLEnv, asset_cfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].delayed()[1]
