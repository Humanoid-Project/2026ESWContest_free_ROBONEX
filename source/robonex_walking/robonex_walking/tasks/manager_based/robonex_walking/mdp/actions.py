from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def slew_limit_step(position, velocity, target, dt, max_speed, max_accel):
    error = target - position
    braking = torch.clamp(
        torch.sqrt((max_accel * dt) ** 2 + 2.0 * max_accel * error.abs()) - max_accel * dt, min=0.0
    )
    desired = torch.sign(error) * torch.clamp(braking, max=max_speed)
    change = torch.clamp(desired - velocity, -max_accel * dt, max_accel * dt)
    next_velocity = torch.clamp(velocity + change, -max_speed, max_speed)
    next_position = position + next_velocity * dt
    arrived = error * (target - next_position) <= 0.0
    next_position = torch.where(arrived, target, next_position)
    next_velocity = torch.where(arrived, torch.zeros_like(next_velocity), next_velocity)
    return next_position, next_velocity


class SlewLimitedJointPositionAction(JointPositionAction):
    cfg: SlewLimitedJointPositionActionCfg

    def __init__(self, cfg: SlewLimitedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._slew_dt = env.step_dt
        self._slew_position = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._slew_velocity = torch.zeros_like(self._slew_position)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        position, velocity = slew_limit_step(
            self._slew_position,
            self._slew_velocity,
            self._processed_actions,
            self._slew_dt,
            self.cfg.max_speed,
            self.cfg.max_accel,
        )
        self._slew_position[:] = position
        self._slew_velocity[:] = velocity
        self._processed_actions = position.clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._slew_position[ids] = self._asset.data.default_joint_pos[ids][:, self._joint_ids]
        self._slew_velocity[ids] = 0.0


@configclass
class SlewLimitedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = SlewLimitedJointPositionAction

    max_speed: float = 6.0
    max_accel: float = 120.0
