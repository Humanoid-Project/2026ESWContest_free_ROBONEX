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


def foot_roll(upper, lower, coeffs):
    c0, c1, c2, c3, c4, c5 = coeffs
    return c0 + c1 * upper + c2 * lower + c3 * upper * upper + c4 * upper * lower + c5 * lower * lower


def clip_foot_roll(upper, lower, upper_range, lower_range, coeffs, limit, iterations=8):
    _, c1, c2, c3, c4, c5 = coeffs
    for _ in range(iterations):
        roll = foot_roll(upper, lower, coeffs)
        excess = roll - torch.clamp(roll, -limit, limit)
        d_upper = c1 + 2.0 * c3 * upper + c4 * lower
        d_lower = c2 + c4 * upper + 2.0 * c5 * lower
        step = excess / (d_upper * d_upper + d_lower * d_lower)
        upper = torch.clamp(upper - step * d_upper, upper_range[0], upper_range[1])
        lower = torch.clamp(lower - step * d_lower, lower_range[0], lower_range[1])
    return upper, lower


class SlewLimitedJointPositionAction(JointPositionAction):
    cfg: SlewLimitedJointPositionActionCfg

    def __init__(self, cfg: SlewLimitedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._slew_dt = env.step_dt
        self._slew_position = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._slew_velocity = torch.zeros_like(self._slew_position)
        self.slew_lag = torch.zeros_like(self._slew_position)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        requested = self._processed_actions.clone()
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
        self.slew_lag = requested - position

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._slew_position[ids] = self._asset.data.default_joint_pos[ids][:, self._joint_ids]
        self._slew_velocity[ids] = 0.0
        self.slew_lag[ids] = 0.0


@configclass
class SlewLimitedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = SlewLimitedJointPositionAction

    max_speed: float = 6.0
    max_accel: float = 120.0


class Ver2JointPositionAction(SlewLimitedJointPositionAction):
    cfg: Ver2JointPositionActionCfg

    def __init__(self, cfg: Ver2JointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._roll_pairs = []
        for upper_name, lower_name, sign in cfg.foot_roll_pairs:
            iu, il = self._joint_names.index(upper_name), self._joint_names.index(lower_name)
            bounds = []
            for i in (iu, il):
                low, high = (sign * float(v) for v in self._clip[0, i])
                bounds.append((min(low, high), max(low, high)))
            self._roll_pairs.append((iu, il, sign, bounds[0], bounds[1]))

    def process_actions(self, actions: torch.Tensor):
        JointPositionAction.process_actions(self, actions)
        if self.cfg.foot_roll_limit > 0.0:
            for iu, il, sign, upper_range, lower_range in self._roll_pairs:
                upper, lower = clip_foot_roll(
                    sign * self._processed_actions[:, iu],
                    sign * self._processed_actions[:, il],
                    upper_range,
                    lower_range,
                    self.cfg.foot_roll_coeffs,
                    self.cfg.foot_roll_limit,
                )
                self._processed_actions[:, iu] = sign * upper
                self._processed_actions[:, il] = sign * lower
        if not self.cfg.slew_enabled:
            self._slew_position[:] = self._processed_actions
            self._slew_velocity[:] = 0.0
            return
        requested = self._processed_actions.clone()
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
        self.slew_lag = requested - position


@configclass
class Ver2JointPositionActionCfg(SlewLimitedJointPositionActionCfg):
    class_type: type = Ver2JointPositionAction

    slew_enabled: bool = True
    foot_roll_limit: float = 0.0
    foot_roll_coeffs: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    foot_roll_pairs: tuple = ()
