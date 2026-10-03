from __future__ import annotations

import torch

JOINT_SPEED_THRESHOLDS_RAD_S = (8.0, 10.0)


def joint_speed(velocity: torch.Tensor, cap: float) -> torch.Tensor:
    speed = velocity.abs()
    speed = torch.where(torch.isfinite(speed), speed, torch.full_like(speed, cap))
    return torch.clamp(speed, max=cap)


def joint_speed_excess(velocity: torch.Tensor, floor: float, cap: float) -> torch.Tensor:
    excess = torch.clamp(joint_speed(velocity, cap) - floor, min=0.0)
    return torch.sum(torch.square(excess), dim=1)


class JointSpeedMetrics:
    def __init__(self, robot, joint_ids, joint_names, thresholds=JOINT_SPEED_THRESHOLDS_RAD_S, prefix="Policy/"):
        self.robot = robot
        self.joint_ids = joint_ids
        self.joint_names = tuple(joint_names)
        self.thresholds = tuple(float(t) for t in thresholds)
        self.prefix = prefix
        self.device = robot.data.joint_vel.device
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        n, dev = len(self.joint_names), self.device
        self.samples = torch.zeros((), device=dev)
        self.peak_sum = torch.zeros((), device=dev)
        self.peak_max = torch.zeros((), device=dev)
        self.nonfinite = torch.zeros((), device=dev)
        self.any_over = torch.zeros(len(self.thresholds), device=dev)
        self.joint_over = torch.zeros(len(self.thresholds), n, device=dev)

    def sample(self) -> torch.Tensor:
        with torch.no_grad():
            velocity = self.robot.data.joint_vel[:, self.joint_ids]
            finite = torch.isfinite(velocity)
            speed = torch.where(finite, velocity.abs(), torch.full_like(velocity, float("inf")))
            peak = speed.amax(dim=1)
            finite_peak = torch.where(finite, speed, torch.zeros_like(speed)).amax(dim=1)
            self.samples += peak.shape[0]
            self.peak_sum += finite_peak.sum()
            self.peak_max = torch.maximum(self.peak_max, finite_peak.amax())
            self.nonfinite += (~finite).sum()
            for index, threshold in enumerate(self.thresholds):
                over = speed > threshold
                self.any_over[index] += over.any(dim=1).sum()
                self.joint_over[index] += over.sum(dim=0)
        return speed

    def take_log(self) -> dict[str, float]:
        samples = max(1.0, float(self.samples.item()))
        p = self.prefix
        values = {
            f"{p}joint_vel_excess_peak_rad_s_mean": self.peak_sum.item() / samples,
            f"{p}joint_vel_excess_peak_rad_s_max": self.peak_max.item(),
            f"{p}joint_vel_excess_nonfinite_fraction": self.nonfinite.item() / (samples * len(self.joint_names)),
        }
        for index, threshold in enumerate(self.thresholds):
            tag = f"gt{threshold:g}"
            values[f"{p}joint_vel_excess_any_{tag}_fraction"] = self.any_over[index].item() / samples
            for j, name in enumerate(self.joint_names):
                values[f"{p}joint_vel_excess_{tag}_fraction/{name}"] = self.joint_over[index, j].item() / samples
        self._reset_accumulators()
        return values
