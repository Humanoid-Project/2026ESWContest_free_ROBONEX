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
    target_speed: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    vel_x = _root_lin_vel_yaw_frame(env, asset_cfg)[:, 0]
    error = _bounded_square(vel_x - target_speed, 5.0)
    return torch.exp(-error / std**2)


def lin_vel_y_l2_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return _bounded_square(_root_lin_vel_yaw_frame(env, asset_cfg)[:, 1], 10.0)


def lin_vel_z_l2_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return _bounded_square(asset.data.root_lin_vel_b[:, 2], 10.0)


def ang_vel_z_l2_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize z-axis base angular velocity (yaw rotation)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _bounded_square(asset.data.root_ang_vel_b[:, 2], 10.0)


def ang_vel_xy_l2_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(_bounded_square(asset.data.root_ang_vel_b[:, :2], 10.0), dim=1)


def flat_orientation_l2_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(_bounded_square(asset.data.projected_gravity_b[:, :2], 1.0), dim=1)


def base_height_l2_bounded(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return _bounded_square(asset.data.root_pos_w[:, 2] - target_height, 2.0)


def action_rate_l2_bounded(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    scale = env.action_manager.get_term(action_name)._scale
    action_delta = (env.action_manager.action - env.action_manager.prev_action) * scale
    return torch.sum(_bounded_square(action_delta, 6.0), dim=1)


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)


def joint_deviation_l1_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    angle = torch.nan_to_num(angle, nan=2.0, posinf=2.0, neginf=-2.0)
    return torch.sum(torch.clamp(torch.abs(angle), max=2.0), dim=1)


def joint_torques_l2_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(_bounded_square(asset.data.applied_torque[:, asset_cfg.joint_ids], 100.0), dim=1)


def joint_acc_l2_bounded(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(_bounded_square(asset.data.joint_acc[:, asset_cfg.joint_ids], 2000.0), dim=1)


def foot_slip_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    contacts = _contact_mask(env, sensor_cfg, threshold)
    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.sum(_bounded_square(foot_vel_xy, 10.0), dim=-1) * contacts, dim=1)


def feet_air_time_biped(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    air_time = torch.nan_to_num(air_time, nan=0.0, posinf=threshold, neginf=0.0)
    contact_time = torch.nan_to_num(contact_time, nan=0.0, posinf=threshold, neginf=0.0)
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    return torch.clamp(reward, min=0.0, max=threshold)


def both_feet_off_ground(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    contacts = _contact_mask(env, sensor_cfg, threshold)
    return (torch.sum(contacts.int(), dim=1) == 0).float()


def feet_clearance_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    contacts = _contact_mask(env, sensor_cfg, threshold)
    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    error = _bounded_square(foot_height - target_height, 1.0)
    return torch.sum(error * (~contacts), dim=1)


def feet_stance_width_l2(env: ManagerBasedRLEnv, target_width: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet width deviation from target width."""
    feet_pos_b = _body_pos_b(env, asset_cfg)
    foot_width = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    return _bounded_square(foot_width - target_width, 2.0)


def unstable_joint_vel(
    env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return _outside_limit(asset.data.joint_vel[:, asset_cfg.joint_ids], limit)
