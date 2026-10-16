# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

from isaaclab.envs.mdp.events import push_by_setting_velocity as _push_by_setting_velocity
from .events import push_standing_by_setting_velocity as _push_standing_by_setting_velocity

from .joint_speed import joint_speed_excess
from .tracking import SentTrackingGateStats, sent_tracking_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _bounded_square(value: torch.Tensor, max_abs: float) -> torch.Tensor:
    value = torch.nan_to_num(value, nan=max_abs, posinf=max_abs, neginf=-max_abs)
    return torch.square(torch.clamp(value, min=-max_abs, max=max_abs))


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by ``quat`` (w, x, y, z), broadcasting over leading dims."""
    xyz = quat[..., 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + quat[..., 0:1] * t + torch.cross(xyz, t, dim=-1)


def _saturate(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Map a non-negative penalty onto [0, 1) with ``1 - exp(-q/scale)``.

    Raw L2 penalties span nine orders of magnitude across terms, so a hand-set
    weight says nothing about priority. Normalising every term first makes the
    weight the priority. ``scale`` is the error that already costs 63% of the
    term's maximum, so it can be chosen from physics instead of by trial.
    """
    value = torch.nan_to_num(value, nan=scale, posinf=scale, neginf=0.0)
    return 1.0 - torch.exp(-torch.clamp(value, min=0.0) / scale)


def _outside_limit(value: torch.Tensor, limit: float) -> torch.Tensor:
    value = value.reshape(value.shape[0], -1)
    finite = torch.isfinite(value)
    bounded_value = torch.where(finite, value, torch.zeros_like(value))
    return torch.any(~finite, dim=1) | torch.any(torch.abs(bounded_value) > limit, dim=1)


def _root_lin_vel_yaw_frame(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    return torch.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)


def _body_pos_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return selected body positions in robot base frame."""
    asset: Articulation = env.scene[asset_cfg.name]

    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    rel_pos_w = body_pos_w - asset.data.root_pos_w.unsqueeze(1)

    num_bodies = body_pos_w.shape[1]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)

    body_pos_b = quat_apply_inverse(
        root_quat_w.reshape(-1, 4),
        rel_pos_w.reshape(-1, 3),
    )
    return body_pos_b.reshape(env.num_envs, num_bodies, 3)


def _contact_mask(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    forces = torch.nan_to_num(forces, nan=threshold + 1.0, posinf=threshold + 1.0, neginf=-(threshold + 1.0))
    return torch.norm(forces, dim=-1).amax(dim=1) > threshold


def _recovery_quiet(
    env: ManagerBasedRLEnv,
    open_speed: float,
    close_speed: float,
    settle_s: float,
    max_open_s: float = 1.5,
    asset_name: str = "robot",
    sensor_name: str = "contact_forces",
    threshold: float = 1.0,
) -> torch.Tensor:
    step = int(env.common_step_counter)
    states = getattr(env, "_recovery_states", None)
    if states is None:
        states = env._recovery_states = {}
    key = (float(open_speed), float(close_speed), float(settle_s), float(max_open_s))
    state = states.get(key)
    if state is None or state["open"].shape[0] != env.num_envs:
        asset: Articulation = env.scene[asset_name]
        mass = asset.data.default_mass.to(env.device)
        state = states[key] = {
            "step": -1,
            "open": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "settle": torch.zeros(env.num_envs, device=env.device),
            "since_fast": torch.zeros(env.num_envs, device=env.device),
            "mass": mass / mass.sum(dim=1, keepdim=True),
        }
    if state["step"] != step:
        state["step"] = step
        asset: Articulation = env.scene[asset_name]
        velocity = torch.nan_to_num(asset.data.body_com_lin_vel_w[:, :, :2], nan=0.0, posinf=0.0, neginf=0.0)
        speed = torch.linalg.norm(torch.sum(velocity * state["mass"].unsqueeze(-1), dim=1), dim=1)
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        forces = torch.nan_to_num(sensor.data.net_forces_w_history, nan=0.0, posinf=0.0, neginf=0.0)
        double = torch.all(torch.norm(forces, dim=-1).amax(dim=1) > threshold, dim=1)
        fresh = env.episode_length_buf <= 1
        fast = speed > open_speed
        state["open"] = (state["open"] & ~fresh) | fast
        state["since_fast"] = torch.where(fast | fresh, torch.zeros_like(state["since_fast"]), state["since_fast"] + env.step_dt)
        calm = double & (speed < close_speed)
        state["settle"] = torch.where(calm & ~fresh, state["settle"] + env.step_dt, torch.zeros_like(state["settle"]))
        state["open"] = state["open"] & (state["settle"] < settle_s - 1.0e-6) & (state["since_fast"] < max_open_s - 1.0e-6)
    return (~state["open"]).float()


def track_lin_vel_x_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    target_speed = env.command_manager.get_command(command_name)[:, 0]
    vel_x = _root_lin_vel_yaw_frame(env, asset_cfg)[:, 0]
    error = _bounded_square(vel_x - target_speed, 5.0)
    return torch.exp(-error / std**2)


def lin_vel_y_l2_bounded(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.04,
) -> torch.Tensor:
    """Penalize lateral velocity away from the commanded one.

    Centred on the command rather than on zero: at c_y = 0 this is the pilot's
    term, and a commanded strafe later does not fight it. Kept additive and
    bounded rather than folded into the forward kernel - a multiplicative lateral
    factor at std 0.15 is what produced the double-stance shuffle.
    """
    command = env.command_manager.get_command(command_name)
    error = command[:, 1] - _root_lin_vel_yaw_frame(env, asset_cfg)[:, 1]
    return _saturate(_bounded_square(error, 2.0), scale)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward yaw-rate tracking, including the zero-yaw case.

    Gating this on a non-zero command left nothing holding yaw at zero, and the
    x tracking term is expressed in the yaw frame, so spinning in place was free.
    Measured cost of the gate: yaw drift went from 0.005 rad/s on the pilot, which
    kept a separate ang_vel_z penalty, to -0.129 rad/s without either.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = _bounded_square(command[:, 2] - asset.data.root_ang_vel_b[:, 2], 3.0)
    return torch.exp(-error / std**2)


def lin_vel_z_l2_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.04,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return _saturate(_bounded_square(asset.data.root_lin_vel_b[:, 2], 2.0), scale)


def ang_vel_xy_l2_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.5,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return _saturate(torch.sum(_bounded_square(asset.data.root_ang_vel_b[:, :2], 10.0), dim=1), scale)


def _mark_sway_push(env, env_ids):
    if not hasattr(env, "_sway_last_push_step"):
        env._sway_last_push_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
    env._sway_last_push_step[env_ids] = env.common_step_counter


def push_robot_with_sway_marker(
    env, env_ids, velocity_range,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    _push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
    _mark_sway_push(env, env_ids)


def push_standing_with_sway_marker(
    env, env_ids, velocity_range,
    command_name: str = "base_velocity",
    command_deadband: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    command = env.command_manager.get_command(command_name)[env_ids]
    standing = env_ids[torch.linalg.norm(command, dim=1) < command_deadband]
    _push_standing_by_setting_velocity(
        env, env_ids, velocity_range, command_name, command_deadband, asset_cfg
    )
    _mark_sway_push(env, standing)


class ang_vel_xy_push_gated(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not hasattr(env, "_sway_last_push_step"):
            env._sway_last_push_step = torch.full(
                (env.num_envs,), -1, dtype=torch.long, device=env.device
            )

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._env._sway_last_push_step[ids] = -1

    def __call__(
        self, env,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        scale: float = 0.5,
        recovery_s: float = 1.5,
    ) -> torch.Tensor:
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        if not math.isfinite(recovery_s) or recovery_s < 0:
            raise ValueError("recovery_s must be finite and nonnegative")
        rates = env.scene[asset_cfg.name].data.root_ang_vel_b[:, :2]
        cost = _saturate(torch.sum(_bounded_square(rates, 10.0), dim=1), scale)
        last = env._sway_last_push_step
        age = env.common_step_counter - last
        grace_steps = math.ceil(recovery_s / env.step_dt)
        recovery = (last >= 0) & (age >= 0) & (age <= grace_steps) & (grace_steps > 0)
        return torch.where(recovery, torch.zeros_like(cost), cost)


def flat_orientation_l2_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.04,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return _saturate(torch.sum(_bounded_square(asset.data.projected_gravity_b[:, :2], 1.0), dim=1), scale)


def base_height_l2_bounded(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.0025,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return _saturate(_bounded_square(asset.data.root_pos_w[:, 2] - target_height, 2.0), scale)


def action_rate_l2_bounded(env: ManagerBasedRLEnv, max_delta_rad: float = 1.0, scale: float = 0.5) -> torch.Tensor:
    term = env.action_manager.get_term("joint_pos")
    action_scale = torch.as_tensor(term._scale, device=env.device, dtype=torch.float32)
    action_delta = env.action_manager.action - env.action_manager.prev_action
    target_delta = action_delta * action_scale.abs()
    return _saturate(torch.sum(_bounded_square(target_delta, max_delta_rad), dim=1), scale)


class track_lin_vel_y_cycle_exp(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._buf = None
        self._count = None
        self._cmd = None
        self._idx = 0
        self._step = None

    def reset(self, env_ids=None):
        if self._buf is None:
            return
        if env_ids is None:
            self._buf.zero_()
            self._count.zero_()
            return
        self._buf[env_ids] = 0.0
        self._count[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        window_steps: int = 40,
        max_error: float = 2.0,
        command_name: str = "base_velocity",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        command = torch.nan_to_num(env.command_manager.get_command(command_name)[:, 1], nan=0.0)
        vel_y = _root_lin_vel_yaw_frame(env, asset_cfg)[:, 1]
        error = torch.clamp(command - vel_y, -max_error, max_error)
        window = max(1, int(window_steps))
        if self._buf is None or self._buf.shape != (error.shape[0], window):
            self._buf = torch.zeros(error.shape[0], window, device=error.device)
            self._count = torch.zeros(error.shape[0], dtype=torch.long, device=error.device)
            self._cmd = command.clone()
            self._idx = 0
        if self._step != env.common_step_counter:
            self._step = env.common_step_counter
            changed = (command - self._cmd).abs() > 1e-6
            self._buf[changed] = 0.0
            self._count[changed] = 0
            self._cmd = command.clone()
            self._buf[:, self._idx] = error
            self._idx = (self._idx + 1) % window
            self._count = torch.clamp(self._count + 1, max=window)
        mean = self._buf.sum(dim=1) / self._count.clamp(min=1)
        return torch.exp(-torch.square(mean) / std**2)


class action_smoothness_l2(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev = None
        self._prev2 = None
        self._fresh = None
        self._step = None

    def reset(self, env_ids=None):
        if self._prev is None:
            return
        if env_ids is None:
            self._prev = None
            self._prev2 = None
            self._fresh = None
            return
        self._fresh[env_ids] = True

    def _target(self, env):
        term = env.action_manager.get_term(self.cfg.params.get("term_name", "joint_pos"))
        scale = torch.as_tensor(term._scale, device=env.device, dtype=torch.float32)
        return torch.nan_to_num(env.action_manager.action, nan=0.0) * scale.abs()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        term_name: str = "joint_pos",
        max_delta_rad: float = 1.0,
        second_order: float = 1.0,
        ramp_start: float = 1.0,
        ramp_steps: int = 0,
    ) -> torch.Tensor:
        target = self._target(env)
        if self._prev is None:
            self._prev = target.clone()
            self._prev2 = target.clone()
            self._fresh = torch.zeros(target.shape[0], dtype=torch.bool, device=target.device)
        if self._step != env.common_step_counter:
            self._step = env.common_step_counter
            fresh = self._fresh.unsqueeze(1)
            self._prev = torch.where(fresh, target, self._prev)
            self._prev2 = torch.where(fresh, target, self._prev2)
            self._fresh[:] = False
            first = target - self._prev
            second = target - 2.0 * self._prev + self._prev2
            ramp = 1.0 if ramp_steps <= 0 else min(1.0, ramp_start + (1.0 - ramp_start) * env.common_step_counter / ramp_steps)
            self._value = ramp * (
                torch.sum(_bounded_square(first, max_delta_rad), dim=1)
                + second_order * torch.sum(_bounded_square(second, max_delta_rad), dim=1)
            )
            self._prev2 = self._prev
            self._prev = target.clone()
        return self._value


def joint_deviation_l1_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.4,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    angle = torch.nan_to_num(angle, nan=2.0, posinf=2.0, neginf=-2.0)
    return _saturate(torch.sum(torch.clamp(torch.abs(angle), max=2.0), dim=1), scale)


def stand_still_pose_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    command_deadband: float = 0.05,
    max_abs: float = 1.0,
    recovery_gate: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """Penalize deviation from the default pose while the velocity command is zero.

    Gated on the standing command, so it cannot buy a shorter, faster stride the way an
    ungated pose penalty did. Plain bounded L1 rather than the saturating kernel: this is a
    hold term, and saturation removes the gradient exactly where the deviation is largest.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    angle = torch.nan_to_num(angle, nan=max_abs, posinf=max_abs, neginf=-max_abs)
    deviation = torch.sum(torch.clamp(torch.abs(angle), max=max_abs), dim=1)
    command = env.command_manager.get_command(command_name)
    penalty = deviation * (torch.linalg.norm(command, dim=1) < command_deadband)
    if recovery_gate is not None:
        penalty = penalty * _recovery_quiet(env, *recovery_gate)
    return penalty


def torque_overrun_l2(
    env: ManagerBasedRLEnv,
    rated_standstill: dict[str, float],
    rated_spinning: dict[str, float],
    spin_speed: float = 10.47,
    margin: float = 0.8,
    max_ratio: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize torque above the motor's continuous rating for its current speed.

    The datasheet ratings are speed dependent — RS03 holds 13 N.m at a standstill but 20 N.m
    at 100 rpm, RS02 6 and 7 — so the limit ramps with |joint velocity| toward ``spin_speed``,
    which is that 100 rpm reference in rad/s. ``margin`` derates the official figures, which
    assume a heatsink this assembly does not have.

    The excess is divided by its own limit so RS02 and RS03 joints are comparable, then
    squared, so a joint at 2.4x its rating contributes 11x one at 0.7x. No saturating kernel:
    the overrun is largest exactly where the gradient has to survive.

    Unlike a plain torque penalty this cannot push the knees straight. Holding the design
    pose costs about 11.7 N.m on hip_roll against a 10.4 N.m derated zero-speed limit, so a
    correct posture scores 0.016 per hip — not zero, but two orders below the 5.75 a
    squeezing stance scores, which is what makes the term safe for posture.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    names = asset.joint_names if isinstance(joint_ids, slice) else [asset.joint_names[i] for i in joint_ids]
    device = asset.data.joint_pos.device
    low = torch.tensor([rated_standstill[n] for n in names], device=device) * margin
    high = torch.tensor([rated_spinning[n] for n in names], device=device) * margin
    speed = torch.nan_to_num(asset.data.joint_vel[:, joint_ids], nan=0.0, posinf=0.0, neginf=0.0).abs()
    limit = low + (high - low) * torch.clamp(speed / spin_speed, max=1.0)
    torque = torch.nan_to_num(
        asset.data.applied_torque[:, joint_ids], nan=0.0, posinf=0.0, neginf=0.0
    ).abs()
    ratio = torch.clamp((torque - limit) / limit, min=0.0, max=max_ratio)
    return torch.sum(torch.square(ratio), dim=1)



def energy_l2_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 120.0,
) -> torch.Tensor:
    """Penalize mechanical power: sum(|joint velocity| * |applied torque|)."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel = torch.nan_to_num(asset.data.joint_vel[:, asset_cfg.joint_ids], nan=0.0, posinf=0.0, neginf=0.0)
    torque = torch.nan_to_num(
        asset.data.applied_torque[:, asset_cfg.joint_ids], nan=0.0, posinf=0.0, neginf=0.0
    )
    vel = torch.clamp(vel.abs(), max=50.0)
    torque = torch.clamp(torque.abs(), max=100.0)
    return _saturate(torch.sum(vel * torque, dim=1), scale)


def joint_vel_l2_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 48.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return _saturate(torch.sum(_bounded_square(asset.data.joint_vel[:, asset_cfg.joint_ids], 50.0), dim=1), scale)


def joint_acc_l2_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 120000.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return _saturate(torch.sum(_bounded_square(asset.data.joint_acc[:, asset_cfg.joint_ids], 500.0), dim=1), scale)


def foot_slip_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.08,
) -> torch.Tensor:
    contacts = _contact_mask(env, sensor_cfg, threshold)
    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return _saturate(torch.sum(torch.sum(_bounded_square(foot_vel_xy, 2.0), dim=-1) * contacts, dim=1), scale)


def foot_slip_latest_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.08,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :]
    forces = torch.nan_to_num(forces, nan=threshold + 1.0, posinf=threshold + 1.0, neginf=-(threshold + 1.0))
    contacts = torch.norm(forces, dim=-1) > threshold
    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return _saturate(torch.sum(torch.sum(_bounded_square(foot_vel_xy, 2.0), dim=-1) * contacts, dim=1), scale)


def stand_still_airborne(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_deadband: float = 0.05,
    threshold: float = 1.0,
    recovery_gate: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """Penalize not being in double stance while the velocity command is zero.

    Pose deviation is the wrong observable: a policy can shrink it by taking smaller,
    faster steps, which is what raising the pose penalty actually produced.
    """
    contacts = _contact_mask(env, sensor_cfg, threshold)
    not_double = (torch.sum(contacts.int(), dim=1) < 2).float()
    command = env.command_manager.get_command(command_name)
    penalty = not_double * (torch.linalg.norm(command, dim=1) < command_deadband)
    if recovery_gate is not None:
        penalty = penalty * _recovery_quiet(env, *recovery_gate)
    return penalty


def target_clip_excess_l2(env: ManagerBasedRLEnv, max_excess: float = 2.0, scale: float = 0.01) -> torch.Tensor:
    """Penalize how far the pre-fence joint target runs past its clip, in radians.

    The fence is asymmetric per joint, so a symmetric threshold on the raw action
    cannot see this: a joint can saturate at |action| = 1.44 while another is free
    out to 8.55.
    """
    term = env.action_manager.get_term("joint_pos")
    clip = getattr(term, "_clip", None)
    if clip is None:
        return torch.zeros(env.num_envs, device=env.device)
    actions = getattr(env, "raw_policy_action", None)
    if actions is None:
        actions = env.action_manager.action
    action_scale = torch.as_tensor(term._scale, device=env.device, dtype=torch.float32)
    offset = torch.as_tensor(term._offset, device=env.device, dtype=torch.float32)
    target = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0) * action_scale + offset
    low, high = clip[..., 0], clip[..., 1]
    excess = torch.clamp(target - high, min=0.0) + torch.clamp(low - target, min=0.0)
    excess = torch.clamp(excess, max=max_excess)
    return _saturate(torch.sum(torch.square(excess), dim=1), scale)


def foot_roll_clip_excess_l2(env: ManagerBasedRLEnv, scale: float = 0.01, term_name: str = "joint_pos") -> torch.Tensor:
    excess = getattr(env.action_manager.get_term(term_name), "foot_roll_excess", None)
    if excess is None:
        return torch.zeros(env.num_envs, device=env.device)
    return _saturate(torch.nan_to_num(excess, nan=scale, posinf=scale, neginf=0.0), scale)


def _selected(names, ids) -> list[str]:
    names = list(names)
    if isinstance(ids, slice):
        return names[ids]
    return [names[int(i)] for i in ids]


def _leg_side(name: str) -> str:
    side = name.split("_", 1)[0]
    if side not in ("l", "r") or "_" not in name:
        raise ValueError(f"cannot assign {name!r} to a leg: expected an l_ or r_ prefix")
    return side


def _stance_gate_layout(env, asset, joint_ids, sensor_cfg: SceneEntityCfg):
    key = (id(asset), repr(joint_ids), sensor_cfg.name, repr(sensor_cfg.body_ids))
    cache = getattr(env, "_sent_tracking_gate_layout", None)
    if cache is not None and cache[0] == key:
        return cache[1], cache[2]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet = _selected(sensor.body_names, sensor_cfg.body_ids)
    columns = {}
    for column, foot in enumerate(feet):
        side = _leg_side(foot)
        if side in columns:
            raise ValueError(f"stance gate: two {side} feet in {feet}")
        columns[side] = column
    joints = _selected(asset.joint_names, joint_ids)
    missing = sorted({_leg_side(j) for j in joints} - set(columns))
    if missing:
        raise ValueError(f"stance gate: no foot for side(s) {missing} in {feet}")
    index = torch.tensor([columns[_leg_side(j)] for j in joints], dtype=torch.long, device=env.device)
    env._sent_tracking_gate_layout = (key, index, tuple(feet))
    return index, tuple(feet)


def sent_tracking_excess_l2(
    env: ManagerBasedRLEnv,
    floor_deg: float = 20.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stance_gate: bool = False,
    contact_threshold_n: float = 20.0,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    error = sent_tracking_error(asset.data.joint_pos_target[:, joint_ids], asset.data.joint_pos[:, joint_ids])
    excess = torch.clamp(error - math.radians(floor_deg), min=0.0)
    if not stance_gate:
        return torch.sum(torch.square(excess), dim=1)
    if sensor_cfg is None:
        raise ValueError("sent_tracking_excess_l2: stance_gate needs sensor_cfg (the foot contact sensor)")
    index, feet = _stance_gate_layout(env, asset, joint_ids, sensor_cfg)
    loaded = _contact_mask(env, sensor_cfg, contact_threshold_n)
    squared = torch.square(excess)
    gated = squared * loaded[:, index]
    stats = getattr(env, "sent_tracking_gate_stats", None)
    if stats is None or stats.feet != feet:
        stats = SentTrackingGateStats(feet, device=squared.device)
        env.sent_tracking_gate_stats = stats
    stats.add(squared, gated, loaded)
    return torch.sum(gated, dim=1)


def joint_vel_excess_l2(
    env: ManagerBasedRLEnv,
    floor_rad_s: float = 8.0,
    cap_rad_s: float = 20.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return joint_speed_excess(asset.data.joint_vel[:, asset_cfg.joint_ids], floor_rad_s, cap_rad_s)


def both_feet_off_ground(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    contacts = _contact_mask(env, sensor_cfg, threshold)
    return (torch.sum(contacts.int(), dim=1) == 0).float()


def feet_stance_width_l2(
    env: ManagerBasedRLEnv,
    target_width: float,
    asset_cfg: SceneEntityCfg,
    scale: float = 0.0025,
    standing_width: float | None = None,
    command_name: str = "base_velocity",
    command_deadband: float = 0.05,
    standing_window: tuple[float, float] | None = None,
    standing_scale: float | None = None,
    lateral_deadband: float | None = None,
) -> torch.Tensor:
    """Penalize feet width deviation from the target for the current command.

    `target_width` is the nominal geometry (legs straight down). Standing wants a wider base
    than walking does, and the S18-vs-S19 evaluation measured it as a single-variable change:
    the standstill fall rate went from 57/512 envs to 12/512. Walking is left alone — every
    policy measured so far converges to 292-309 mm while moving, below the nominal target
    either way.
    """
    feet_pos_b = _body_pos_b(env, asset_cfg)
    foot_width = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    target = torch.full_like(foot_width, target_width)
    if standing_window is not None:
        command = env.command_manager.get_command(command_name)
        standing = torch.linalg.norm(command, dim=1) < command_deadband
        deviation = foot_width - target
        if lateral_deadband is not None:
            band = lateral_deadband * torch.abs(command[:, 1])
            deviation = torch.sign(deviation) * torch.clamp(torch.abs(deviation) - band, min=0.0)
        walking_penalty = _saturate(_bounded_square(deviation, 2.0), scale)
        outside = torch.clamp(standing_window[0] - foot_width, min=0.0) + torch.clamp(
            foot_width - standing_window[1], min=0.0
        )
        standing_penalty = _saturate(
            _bounded_square(outside, 2.0), scale if standing_scale is None else standing_scale
        )
        return torch.where(standing, standing_penalty, walking_penalty)
    if lateral_deadband is not None:
        raise ValueError("lateral_deadband needs standing_window")
    if standing_width is not None:
        command = env.command_manager.get_command(command_name)
        standing = torch.linalg.norm(command, dim=1) < command_deadband
        target = torch.where(standing, torch.full_like(foot_width, standing_width), target)
    return _saturate(_bounded_square(foot_width - target, 2.0), scale)


def standing_joint_load_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    reference_torque: float = 20.0,
    max_torque: float = 60.0,
    command_name: str = "base_velocity",
    command_deadband: float = 0.05,
    recovery_gate: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    torque = torch.nan_to_num(
        asset.data.applied_torque[:, asset_cfg.joint_ids], nan=max_torque, posinf=max_torque, neginf=-max_torque
    )
    load = torch.sum(torch.clamp(torque.abs(), max=max_torque), dim=1) / reference_torque
    command = env.command_manager.get_command(command_name)
    penalty = load * (torch.linalg.norm(command, dim=1) < command_deadband)
    if recovery_gate is not None:
        penalty = penalty * _recovery_quiet(env, *recovery_gate)
    return penalty


def joint_load_rating_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    rated: dict[str, float],
    max_ratio: float = 5.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    names = asset.joint_names if isinstance(joint_ids, slice) else [asset.joint_names[i] for i in joint_ids]
    torque = asset.data.applied_torque[:, joint_ids]
    rating = torch.tensor([rated[n] for n in names], device=torque.device, dtype=torque.dtype)
    ratio = torch.nan_to_num(torque.abs() / rating, nan=max_ratio, posinf=max_ratio, neginf=max_ratio)
    return torch.sum(torch.square(torch.clamp(ratio, min=0.0, max=max_ratio)), dim=1)


def unstable_joint_vel(
    env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return _outside_limit(asset.data.joint_vel[:, asset_cfg.joint_ids], limit)


def feet_clearance_clock_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg,
    period: float,
    offset: list[float],
    sole_corners,
    stance_fraction: float = 0.55,
    scale: float = 0.0018,
    command_name: str | None = None,
    command_deadband: float = 0.05,
) -> torch.Tensor:
    """Penalize a low swing foot, weighted by how far into the swing the clock is.

    The contact-gated form charges the full penalty on the frame contact breaks, when
    the foot is still at rest height and 6 cm of clearance is not yet physically
    reachable: a 0.43 step discontinuity paid for starting the correct motion. G1's
    foot-speed gate has no such step but hands a planted foot a perfect score, because
    a stationary foot multiplies its own error by tanh(0).

    Weighting by sin^2 over the swing window is zero at both swing boundaries, so
    lift-off and touchdown cost nothing, and one at mid-swing, where a foot still on
    the ground pays the full penalty. The clock is already in the observation as
    ``gait_phase`` and already rewarded by ``feet_gait``.

    The command gate is not optional once the contact mask is gone. The clock keeps
    running on a zero command, so a correctly standing robot would meet a mid-swing
    window with both feet down and pay the full penalty for standing still -- the
    exact pressure that produces marching in place.

    ``target_height`` is clearance of the SOLE above the ground, not an absolute height
    of the link origin. The origin sits 0.151 m behind the toe, so reading its height
    lets a policy pitch the toe down and collect the reward with the toe still low: at
    500 iterations a policy trained against the origin held +11.7 deg at mid-swing, where
    the origin read 62.8 mm and the sole was at 35.3 mm. The lowest of the four sole
    corners is checked against the full 13716-vertex mesh -- never optimistic, at most
    5.02 mm conservative over pitch -16..20 deg and roll -10..10 deg.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :].unsqueeze(2)
    corners = torch.as_tensor(sole_corners, dtype=body_pos.dtype, device=body_pos.device)
    corners = corners.view(1, 1, -1, 3).expand(quat.shape[0], quat.shape[1], -1, -1)
    corner_z = _quat_apply(quat.expand(-1, -1, corners.shape[2], -1), corners)[..., 2]
    foot_height = (body_pos[:, :, 2].unsqueeze(-1) + corner_z).amin(dim=-1)
    foot_height = foot_height - env.scene.env_origins[:, 2].unsqueeze(1)
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    leg_phase = torch.cat([(global_phase + value) % 1.0 for value in offset], dim=-1)
    swing = torch.clamp(leg_phase - stance_fraction, min=0.0) / (1.0 - stance_fraction)
    window = torch.square(torch.sin(math.pi * swing))
    error = _bounded_square(foot_height - target_height, 1.0)
    penalty = _saturate(torch.sum(error * window, dim=1), scale)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        penalty = penalty * (torch.linalg.norm(command, dim=1) > command_deadband)
    return penalty


def feet_gait_centred(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.55,
    command_name: str | None = None,
    command_deadband: float = 0.05,
) -> torch.Tensor:
    """Reward a clock-matching contact pattern, with the planted baseline removed.

    An uncentred score of one point per leg whose contact state matches the clock lets a
    robot that simply keeps both feet down still collect ``2 * stance_fraction`` = 1.10
    of the 2.0 maximum. Rescaling so that the planted pattern scores 0 and a perfect
    gait scores 1 makes the weight the actual anti-shuffle margin; a pattern worse than
    planted now goes negative instead of merely earning less.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = torch.nan_to_num(
        sensor.data.current_contact_time[:, sensor_cfg.body_ids], nan=0.0, posinf=0.0, neginf=0.0
    )
    is_contact = contact_time > 0.0
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    leg_phase = torch.cat([(global_phase + value) % 1.0 for value in offset], dim=-1)

    matched = torch.zeros(env.num_envs, device=env.device)
    for index in range(is_contact.shape[1]):
        is_stance = leg_phase[:, index] < threshold
        matched = matched + (~(is_stance ^ is_contact[:, index])).float()

    n_legs = float(is_contact.shape[1])
    reward = (matched - n_legs * threshold) / (n_legs * (1.0 - threshold))
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        reward = reward * (torch.linalg.norm(command, dim=1) > command_deadband)
    return reward


def feet_contact_force_l2(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg,
    scale: float = 10000.0,
) -> torch.Tensor:
    """Penalize contact force above a threshold."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    forces = torch.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)
    peak = torch.norm(forces, dim=-1).amax(dim=1)
    return _saturate(torch.sum(torch.square(torch.clamp(peak - threshold, min=0.0)), dim=1), scale)


def feet_contact_force_mean_l2(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg,
    scale: float = 10000.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, : env.cfg.decimation, sensor_cfg.body_ids, :]
    forces = torch.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)
    mean = torch.norm(forces.mean(dim=1), dim=-1)
    return _saturate(torch.sum(torch.square(torch.clamp(mean - threshold, min=0.0)), dim=1), scale)


def nonfinite_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    return ~torch.isfinite(forces).reshape(env.num_envs, -1).all(dim=1)


def invalid_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    data = asset.data
    invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=data.root_pos_w.device)
    for value in (
        data.root_pos_w,
        data.root_quat_w,
        data.root_lin_vel_w,
        data.root_ang_vel_w,
        data.joint_pos,
        data.joint_vel,
        data.projected_gravity_b,
    ):
        invalid |= ~torch.isfinite(value).reshape(env.num_envs, -1).all(dim=1)
    invalid |= (torch.linalg.vector_norm(data.root_quat_w, dim=-1) - 1.0).abs() > 0.1
    invalid |= (torch.linalg.vector_norm(data.projected_gravity_b, dim=-1) - 1.0).abs() > 0.1
    return invalid


def slew_lag_l2(env: ManagerBasedRLEnv, scale: float = 0.0025, term_name: str = "joint_pos") -> torch.Tensor:
    lag = getattr(env.action_manager.get_term(term_name), "slew_lag", None)
    if lag is None:
        return torch.zeros(env.num_envs, device=env.device)
    return _saturate(torch.sum(_bounded_square(lag, 2.0), dim=1), scale)


class com_forward_margin_l2(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._ema = None
        self._still = None
        self._step = None
        self._weights = None

    def reset(self, env_ids=None):
        return

    def margin(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, foot_cfg: SceneEntityCfg) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        if self._weights is None:
            mass = asset.data.default_mass.to(env.device)
            self._weights = mass / mass.sum(dim=1, keepdim=True)
        com = torch.sum(asset.data.body_com_pos_w * self._weights.unsqueeze(-1), dim=1)
        feet = asset.data.body_pos_w[:, foot_cfg.body_ids, :].mean(dim=1)
        offset = com - feet
        offset[:, 2] = 0.0
        local = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), offset)
        return torch.nan_to_num(local[:, 0], nan=0.0, posinf=0.0, neginf=0.0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target: float = 0.065,
        scale: float = 0.02,
        tau_s: float = 0.8,
        recovery_gate: tuple[float, float, float, float] | None = None,
        standing: str = "floor",
        standing_cap: float = 0.060,
        settle_s: float = 2.5,
        command_name: str = "base_velocity",
        command_deadband: float = 0.05,
        push_grace_s: float = 0.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["l_foot", "r_foot"]),
    ) -> torch.Tensor:
        if not math.isfinite(push_grace_s) or push_grace_s < 0:
            raise ValueError("push_grace_s must be finite and nonnegative")
        if standing not in ("floor", "cap", "off"):
            raise ValueError(f"com_forward_margin standing must be floor, cap or off, not {standing!r}")
        dx = self.margin(env, asset_cfg, foot_cfg)
        if self._ema is None or self._ema.shape != dx.shape:
            self._ema = dx.clone()
            self._still = torch.zeros_like(dx)
        if self._step != env.common_step_counter:
            self._step = env.common_step_counter
            alpha = env.step_dt / (tau_s + env.step_dt)
            self._ema = self._ema + alpha * (dx - self._ema)
            fresh = env.episode_length_buf <= 1
            self._ema = torch.where(fresh, dx, self._ema)
            command = env.command_manager.get_command(command_name)
            still = torch.linalg.norm(command, dim=1) < command_deadband
            self._still = torch.where(still & ~fresh, self._still + env.step_dt, torch.zeros_like(self._still))
        shortfall = torch.clamp(target - self._ema, min=0.0)
        penalty = _saturate(torch.square(shortfall), scale * scale)
        if standing != "floor":
            settled = self._still >= settle_s - 0.5 * env.step_dt
            excess = torch.clamp(self._ema - standing_cap, min=0.0)
            above = _saturate(torch.square(excess), scale * scale) if standing == "cap" else torch.zeros_like(penalty)
            penalty = torch.where(settled, above, penalty)
        if recovery_gate is not None:
            penalty = penalty * _recovery_quiet(env, *recovery_gate)
        last = getattr(env, "_sway_last_push_step", None)
        if push_grace_s > 0 and last is not None:
            age = env.common_step_counter - last
            grace = (last >= 0) & (age >= 0) & (age <= math.ceil(push_grace_s / env.step_dt))
            penalty = torch.where(grace, torch.zeros_like(penalty), penalty)
        return penalty
