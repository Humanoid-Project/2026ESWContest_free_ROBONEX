# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _bounded_square(value: torch.Tensor, max_abs: float) -> torch.Tensor:
    value = torch.nan_to_num(value, nan=max_abs, posinf=max_abs, neginf=-max_abs)
    return torch.square(torch.clamp(value, min=-max_abs, max=max_abs))


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
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.04,
) -> torch.Tensor:
    return _saturate(_bounded_square(_root_lin_vel_yaw_frame(env, asset_cfg)[:, 1], 2.0), scale)


def lin_vel_z_l2_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.04,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return _saturate(_bounded_square(asset.data.root_lin_vel_b[:, 2], 2.0), scale)


def ang_vel_z_l2_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.8,
) -> torch.Tensor:
    """Penalize z-axis base angular velocity (yaw rotation)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _saturate(_bounded_square(asset.data.root_ang_vel_b[:, 2], 3.0), scale)


def ang_vel_xy_l2_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.5,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return _saturate(torch.sum(_bounded_square(asset.data.root_ang_vel_b[:, :2], 10.0), dim=1), scale)


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


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)


def joint_deviation_l1_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.4,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    angle = torch.nan_to_num(angle, nan=2.0, posinf=2.0, neginf=-2.0)
    return _saturate(torch.sum(torch.clamp(torch.abs(angle), max=2.0), dim=1), scale)


def joint_torques_l2_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(_bounded_square(asset.data.applied_torque[:, asset_cfg.joint_ids], 100.0), dim=1)


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


class _GaitTracker:
    def __init__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, grace_period: float):
        self.sensor_cfg = sensor_cfg
        self.grace_period = grace_period
        self.step = None
        n, dev = env.num_envs, env.device
        self.prev_contact = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.started = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_td_foot = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.since_single = torch.full((n,), float("inf"), device=dev)
        self.touchdown_credit = torch.zeros(n, device=dev)
        self.single_recent = torch.zeros(n, dtype=torch.bool, device=dev)
        self.pending_reset = torch.zeros(n, dtype=torch.bool, device=dev)

    def update(self, env: ManagerBasedRLEnv) -> "_GaitTracker":
        if self.step == env.common_step_counter:
            return self
        self.step = env.common_step_counter
        sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        forces = sensor.data.net_forces_w_history[:, :, self.sensor_cfg.body_ids, :]
        forces = torch.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)
        contact = torch.norm(forces, dim=-1).amax(dim=1) > 1.0

        # A reset lands after the reward is computed, so the flag raised on the
        # terminal step is consumed here, on the first step of the new episode.
        fresh = ~self.started | self.pending_reset
        dones = getattr(getattr(env, "termination_manager", None), "dones", None)
        self.pending_reset = dones.bool().clone() if dones is not None else torch.zeros_like(self.started)
        if bool(fresh.any()):
            self.prev_contact[fresh] = contact[fresh]
            self.last_td_foot[fresh] = -1
            self.since_single[fresh] = float("inf")
            self.started |= fresh

        n_contact = contact.sum(dim=1)
        self.since_single = torch.where(
            n_contact == 1, torch.zeros_like(self.since_single), self.since_single + env.step_dt
        )
        self.single_recent = self.since_single <= self.grace_period

        # Isaac Lab zeroes current_air_time on touchdown and moves the elapsed
        # swing into last_air_time, so the credit must read last_air_time.
        air_time = sensor.data.last_air_time[:, self.sensor_cfg.body_ids]
        air_time = torch.nan_to_num(air_time, nan=0.0, posinf=0.0, neginf=0.0)
        touchdown = contact & ~self.prev_contact
        simultaneous = touchdown.all(dim=1)

        credit = torch.zeros_like(self.touchdown_credit)
        for foot in (0, 1):
            hit = touchdown[:, foot] & ~simultaneous
            if not bool(hit.any()):
                continue
            alternating = hit & (self.last_td_foot != foot)
            credit = torch.where(alternating, air_time[:, foot], credit)
            self.last_td_foot[hit] = foot
        if bool(simultaneous.any()):
            self.last_td_foot[simultaneous] = -1
        self.touchdown_credit = credit
        self.prev_contact = contact
        return self


def _gait_tracker(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, grace_period: float):
    tracker = getattr(env, "_walk_gait_tracker", None)
    if tracker is None:
        tracker = _GaitTracker(env, sensor_cfg, grace_period)
        env._walk_gait_tracker = tracker
    return tracker.update(env)


def feet_air_time_biped(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    grace_period: float = 0.2,
) -> torch.Tensor:
    tracker = _gait_tracker(env, sensor_cfg, grace_period)
    return torch.clamp(tracker.touchdown_credit, min=0.0, max=threshold)


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


def both_feet_off_ground(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    contacts = _contact_mask(env, sensor_cfg, threshold)
    return (torch.sum(contacts.int(), dim=1) == 0).float()


def feet_stance_width_l2(env: ManagerBasedRLEnv, target_width: float, asset_cfg: SceneEntityCfg, scale: float = 0.0025) -> torch.Tensor:
    """Penalize feet width deviation from target width."""
    feet_pos_b = _body_pos_b(env, asset_cfg)
    foot_width = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    return _saturate(_bounded_square(foot_width - target_width, 2.0), scale)


def unstable_joint_vel(
    env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return _outside_limit(asset.data.joint_vel[:, asset_cfg.joint_ids], limit)


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.55,
    command_name: str | None = None,
    command_deadband: float = 0.05,
) -> torch.Tensor:
    """Reward a contact pattern that matches a fixed-period gait clock."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = torch.nan_to_num(
        sensor.data.current_contact_time[:, sensor_cfg.body_ids], nan=0.0, posinf=0.0, neginf=0.0
    )
    is_contact = contact_time > 0.0
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    leg_phase = torch.cat([(global_phase + value) % 1.0 for value in offset], dim=-1)

    # body_ids is a slice when the sensor already covers exactly the wanted bodies
    reward = torch.zeros(env.num_envs, device=env.device)
    for index in range(is_contact.shape[1]):
        is_stance = leg_phase[:, index] < threshold
        reward = reward + (~(is_stance ^ is_contact[:, index])).float()
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        reward = reward * (torch.linalg.norm(command, dim=1) > command_deadband)
    return reward


def feet_clearance_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    scale: float = 0.0018,
) -> torch.Tensor:
    """Penalize a swing foot that is not at the target height.

    Gated on contact rather than on horizontal foot speed: a velocity weight
    collapses to zero when the robot shuffles without translating, which hands
    out a near-perfect score for never lifting the foot at all.
    """
    contacts = _contact_mask(env, sensor_cfg, threshold)
    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    error = _bounded_square(foot_height - target_height, 1.0)
    return _saturate(torch.sum(error * (~contacts), dim=1), scale)

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
