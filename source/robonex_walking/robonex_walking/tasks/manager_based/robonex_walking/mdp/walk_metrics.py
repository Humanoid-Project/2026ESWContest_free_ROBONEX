from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

FEET = ("l_foot", "r_foot")
CONTACT_FORCE_THRESHOLD = 1.0


class WalkMetrics:
    def __init__(self, env: ManagerBasedRLEnv, rest_height: float, dt: float) -> None:
        self.robot: Articulation = env.scene["robot"]
        self.sensor: ContactSensor = env.scene.sensors["contact_forces"]
        self.body_ids, _ = self.robot.find_bodies(list(FEET), preserve_order=True)
        self.sensor_ids, _ = self.sensor.find_bodies(list(FEET), preserve_order=True)
        self.action = env.action_manager.get_term("joint_pos")
        self.joint_names = tuple(self.action._joint_names)
        self.rest_height = rest_height
        self.dt = dt
        self.num_envs = env.num_envs
        self.device = env.device
        self.step = None
        self.log: dict[str, torch.Tensor] = {}
        self._reset_state()
        self._reset_accumulators()

    def _reset_state(self) -> None:
        n, dev = self.num_envs, self.device
        self.prev_contact = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.started = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_td_foot = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.since_last_td = torch.zeros(n, device=dev)
        self.swing_peak = torch.zeros(n, 2, device=dev)

    def _reset_accumulators(self) -> None:
        dev = self.device
        self.n_steps = torch.zeros((), device=dev)
        self.contact_steps = torch.zeros(2, device=dev)
        self.phase_steps = torch.zeros(3, device=dev)
        self.td_count = torch.zeros(2, device=dev)
        self.td_alt = torch.zeros((), device=dev)
        self.td_same = torch.zeros((), device=dev)
        self.td_both = torch.zeros((), device=dev)
        self.step_dur_sum = torch.zeros((), device=dev)
        self.step_dur_n = torch.zeros((), device=dev)
        self.swing_peak_sum = torch.zeros((), device=dev)
        self.swing_peak_n = torch.zeros((), device=dev)
        self.slip_sq_sum = torch.zeros((), device=dev)
        self.slip_n = torch.zeros((), device=dev)
        self.action_sum = torch.zeros(len(self.joint_names), device=dev)
        self.action_abs_max = torch.zeros(len(self.joint_names), device=dev)
        self.action_n = torch.zeros((), device=dev)

    def record_action(self, actions: torch.Tensor) -> None:
        with torch.no_grad():
            finite = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
            self.action_sum += finite.sum(dim=0)
            self.action_abs_max = torch.maximum(self.action_abs_max, finite.abs().amax(dim=0))
            self.action_n += actions.shape[0]

    def update(self, env: ManagerBasedRLEnv) -> "WalkMetrics":
        if self.step == env.common_step_counter:
            return self
        self.step = env.common_step_counter
        with torch.no_grad():
            self._update(env)
        return self

    def _update(self, env: ManagerBasedRLEnv) -> None:
        forces = self.sensor.data.net_forces_w_history[:, :, self.sensor_ids, :]
        forces = torch.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)
        contact = torch.norm(forces, dim=-1).amax(dim=1) > CONTACT_FORCE_THRESHOLD

        dones = getattr(getattr(env, "termination_manager", None), "dones", None)
        fresh = torch.zeros_like(self.started)
        if dones is not None:
            fresh |= dones.bool()
        fresh |= ~self.started
        if bool(fresh.any()):
            self.prev_contact[fresh] = contact[fresh]
            self.last_td_foot[fresh] = -1
            self.since_last_td[fresh] = 0.0
            self.swing_peak[fresh] = 0.0
            self.started |= fresh

        body_pos = self.robot.data.body_pos_w[:, self.body_ids, :]
        height = body_pos[:, :, 2] - env.scene.env_origins[:, 2].unsqueeze(1) - self.rest_height
        airborne = ~contact
        self.swing_peak = torch.where(airborne, torch.maximum(self.swing_peak, height), self.swing_peak)

        body_vel = self.robot.data.body_lin_vel_w[:, self.body_ids, :2]
        body_vel = torch.nan_to_num(body_vel, nan=0.0, posinf=0.0, neginf=0.0)
        slip = torch.square(body_vel).sum(dim=-1)[contact]
        if slip.numel() > 0:
            self.slip_sq_sum += slip.sum()
            self.slip_n += slip.numel()

        n_contact = contact.sum(dim=1)
        self.phase_steps[0] += (n_contact == 1).sum()
        self.phase_steps[1] += (n_contact == 2).sum()
        self.phase_steps[2] += (n_contact == 0).sum()
        self.contact_steps += contact.sum(dim=0)
        self.n_steps += contact.shape[0]

        touchdown = contact & ~self.prev_contact
        self.td_count += touchdown.sum(dim=0)
        self.since_last_td += self.dt

        simultaneous = touchdown.all(dim=1)
        self.td_both += simultaneous.sum()

        for foot in (0, 1):
            hit = touchdown[:, foot]
            peak = self.swing_peak[hit, foot]
            self.swing_peak_sum += peak.sum()
            self.swing_peak_n += peak.numel()
            self.swing_peak[hit, foot] = 0.0

            hit = hit & ~simultaneous
            if not bool(hit.any()):
                continue
            prev = self.last_td_foot[hit]
            known = prev >= 0
            same = known & (prev == foot)
            self.td_same += same.sum()
            self.td_alt += (known & ~same).sum()
            dur = self.since_last_td[hit][known]
            if dur.numel() > 0:
                self.step_dur_sum += dur.sum()
                self.step_dur_n += dur.numel()
            self.last_td_foot[hit] = foot
            self.since_last_td[hit] = 0.0

        if bool(simultaneous.any()):
            self.last_td_foot[simultaneous] = -1
            self.since_last_td[simultaneous] = 0.0

        self.prev_contact = contact

    def take_log(self) -> dict[str, float]:
        steps = torch.clamp(self.n_steps, min=1.0)
        seconds = torch.clamp(self.n_steps * self.dt, min=1e-6)
        td_total = torch.clamp(self.td_alt + self.td_same, min=1.0)
        values = {
            "Gait/duty_l": (self.contact_steps[0] / steps).item(),
            "Gait/duty_r": (self.contact_steps[1] / steps).item(),
            "Gait/single_stance_frac": (self.phase_steps[0] / steps).item(),
            "Gait/double_stance_frac": (self.phase_steps[1] / steps).item(),
            "Gait/flight_frac": (self.phase_steps[2] / steps).item(),
            "Gait/touchdown_hz_l": (self.td_count[0] / seconds).item(),
            "Gait/touchdown_hz_r": (self.td_count[1] / seconds).item(),
            "Gait/same_foot_td_frac": (self.td_same / td_total).item(),
            "Gait/simultaneous_td_hz": (self.td_both / seconds).item(),
            "Gait/step_duration_s": (
                self.step_dur_sum / torch.clamp(self.step_dur_n, min=1.0)
            ).item(),
            "Gait/swing_peak_m": (
                self.swing_peak_sum / torch.clamp(self.swing_peak_n, min=1.0)
            ).item(),
            "Gait/contact_slip_m_s": torch.sqrt(
                self.slip_sq_sum / torch.clamp(self.slip_n, min=1.0)
            ).item(),
        }
        action_n = torch.clamp(self.action_n, min=1.0)
        for index, name in enumerate(self.joint_names):
            values[f"ActionMean/{name}"] = (self.action_sum[index] / action_n).item()
            values[f"ActionAbsMax/{name}"] = self.action_abs_max[index].item()
        self._reset_accumulators()
        return values


def walk_metrics(env: ManagerBasedRLEnv, rest_height: float, dt: float) -> WalkMetrics:
    if not hasattr(env, "_walk_metrics"):
        env._walk_metrics = WalkMetrics(env, rest_height, dt)
    return env._walk_metrics.update(env)
