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


def track_lin_vel_y_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward lateral tracking, including the zero-lateral case.

    Ungated, like ``track_lin_vel_x_exp`` and ``track_ang_vel_z_exp``. The earlier
    ``|cy| > deadband`` gate meant the term paid nothing while walking straight, so
    holding the line sideways was worth nothing and the lateral axis was carried by a
    -0.3 penalty instead. Measured after 313 iterations trained with y in +/-0.2, that
    policy realised 12-30% of a lateral command and answered vy=-0.2 by yawing 76 deg.
    No reference penalises lateral velocity; every one of them rewards tracking it.
    """
    command = env.command_manager.get_command(command_name)
    vel_y = _root_lin_vel_yaw_frame(env, asset_cfg)[:, 1]
    error = _bounded_square(command[:, 1] - vel_y, 5.0)
    return torch.exp(-error / std**2)


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
    return deviation * (torch.linalg.norm(command, dim=1) < command_deadband)


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



class torque_thermal_l2(ManagerTermBase):
    """Penalize the *time-integrated* load on each motor, not the instantaneous excess.

    ``torque_overrun_l2`` scores the per-step excess over the rating. Thermal damage is the
    integral of ``tau^2``, and the difference is not academic: hardware measured l_hip_roll at
    12.05 N.m RMS against a 13 N.m rating -- inside the threshold 65.6% of samples, so for two
    thirds of the trajectory the old term's gradient was exactly zero.

    Here ``h`` is an exponential running mean of ``(tau / limit)^2`` with time constant
    ``time_constant``. At a steady ``tau = r * limit`` it converges to ``r^2``, so ``h > 1``
    means "this joint has been averaging above its derated rating" and ``relu`` leaves a
    correct posture at zero cost. Because ``dh/dtau > 0`` at every step, a joint already in
    debt is pushed on every sample rather than only while over the line.

    The excess is taken **linearly**, not squared. ``h`` is already a mean of ``tau^2``, so
    ``relu(h - 1)`` is already quadratic in torque; squaring again would be quartic, which has
    no thermal justification and is a reward-hacking trap. At a weight matched to the term it
    replaces, the squared form scores a 35 N.m squeeze at -39 against a -5 termination
    penalty, so falling over deliberately would be the cheaper policy.

    ``max_excess`` caps ``h`` and therefore the penalty. **Where that cap sits is the whole
    design.** S20 set it to 2, putting the ceiling at 18.2 N.m -- inside the operating range,
    where S19 already spent 29.5% of its samples -- and the policy climbed onto the plateau,
    because past a ceiling more torque is free. It settled at 18.34 N.m RMS, 36% worse than the
    baseline it was supposed to improve. At 6 the ceiling is 27.9 N.m, twice the operating
    torque, and only 0.9% of baseline samples reach it.

    ``limit`` is the same speed-interpolated, margin-derated rating ``torque_overrun_l2``
    uses. ``h`` is per-environment state, zeroed on reset, and is not observed by the critic.
    ``time_constant`` is a training proxy covering a few gait periods -- not a motor thermal
    constant, which is minutes.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        asset_cfg: SceneEntityCfg = params["asset_cfg"]
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._joint_ids = asset_cfg.joint_ids
        names = (
            self._asset.joint_names
            if isinstance(self._joint_ids, slice)
            else [self._asset.joint_names[i] for i in self._joint_ids]
        )
        margin = params.get("margin", 0.8)
        device = env.device
        self._low = torch.tensor([params["rated_standstill"][n] for n in names], device=device) * margin
        self._high = torch.tensor([params["rated_spinning"][n] for n in names], device=device) * margin
        self._spin_speed = params.get("spin_speed", 10.47)
        self._max_excess = params.get("max_excess", 6.0)
        time_constant = params.get("time_constant", 5.0)
        self._decay = math.exp(-float(env.step_dt) / time_constant)
        self._heat = torch.zeros(env.num_envs, len(names), device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._heat.zero_()
        else:
            self._heat[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        rated_standstill: dict[str, float],
        rated_spinning: dict[str, float],
        asset_cfg: SceneEntityCfg,
        spin_speed: float = 10.47,
        margin: float = 0.8,
        max_excess: float = 6.0,
        time_constant: float = 5.0,
    ) -> torch.Tensor:
        speed = torch.nan_to_num(
            self._asset.data.joint_vel[:, self._joint_ids], nan=0.0, posinf=0.0, neginf=0.0
        ).abs()
        limit = self._low + (self._high - self._low) * torch.clamp(speed / self._spin_speed, max=1.0)
        torque = torch.nan_to_num(
            self._asset.data.applied_torque[:, self._joint_ids], nan=0.0, posinf=0.0, neginf=0.0
        ).abs()
        load = torch.clamp((torque / limit) ** 2, max=1.0 + self._max_excess)
        self._heat.mul_(self._decay).add_(load, alpha=1.0 - self._decay)
        return torch.sum(torch.clamp(self._heat - 1.0, min=0.0), dim=1)


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
    def __init__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
        self.sensor_cfg = sensor_cfg
        self.step = None
        n, dev = env.num_envs, env.device
        self.prev_contact = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.started = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_td_foot = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.touchdown_credit = torch.zeros(n, device=dev)
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
            self.started |= fresh

        n_contact = contact.sum(dim=1)

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


def _gait_tracker(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    tracker = getattr(env, "_walk_gait_tracker", None)
    if tracker is None:
        tracker = _GaitTracker(env, sensor_cfg)
        env._walk_gait_tracker = tracker
    return tracker.update(env)


def feet_air_time_biped(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    command_name: str | None = None,
    command_deadband: float = 0.05,
) -> torch.Tensor:
    tracker = _gait_tracker(env, sensor_cfg)
    reward = torch.clamp(tracker.touchdown_credit, min=0.0, max=threshold)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        reward = reward * (torch.linalg.norm(command, dim=1) > command_deadband)
    return reward


def stand_still_airborne(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_deadband: float = 0.05,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize not being in double stance while the velocity command is zero.

    Pose deviation is the wrong observable: a policy can shrink it by taking smaller,
    faster steps, which is what raising the pose penalty actually produced.
    """
    contacts = _contact_mask(env, sensor_cfg, threshold)
    not_double = (torch.sum(contacts.int(), dim=1) < 2).float()
    command = env.command_manager.get_command(command_name)
    return not_double * (torch.linalg.norm(command, dim=1) < command_deadband)


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


def feet_stance_width_l2(
    env: ManagerBasedRLEnv,
    target_width: float,
    asset_cfg: SceneEntityCfg,
    scale: float = 0.0025,
    standing_width: float | None = None,
    command_name: str = "base_velocity",
    command_deadband: float = 0.05,
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
    if standing_width is not None:
        command = env.command_manager.get_command(command_name)
        standing = torch.linalg.norm(command, dim=1) < command_deadband
        target = torch.where(standing, torch.full_like(foot_width, standing_width), target)
    return _saturate(_bounded_square(foot_width - target, 2.0), scale)


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

    ``feet_gait`` returns one point per leg whose contact state matches the clock, so a
    robot that simply keeps both feet down still collects ``2 * stance_fraction`` = 1.10
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
