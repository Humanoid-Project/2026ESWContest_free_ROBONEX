from __future__ import annotations

import math

import torch

TRACKING_THRESHOLDS_DEG = (20.0, 25.0)


def sent_tracking_error(target: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    delta = target - position
    error = (torch.remainder(delta + math.pi, 2.0 * math.pi) - math.pi).abs()
    return torch.where(torch.isfinite(error), error, torch.full_like(error, math.pi))


class SentTrackingMetrics:
    def __init__(self, robot, joint_ids, joint_names, thresholds_deg=TRACKING_THRESHOLDS_DEG, prefix="Policy/"):
        self.robot = robot
        self.joint_ids = joint_ids
        self.joint_names = tuple(joint_names)
        self.thresholds_deg = tuple(float(t) for t in thresholds_deg)
        self.prefix = prefix
        self.device = robot.data.joint_pos.device
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        n, dev = len(self.joint_names), self.device
        self.samples = torch.zeros((), device=dev)
        self.peak_sum = torch.zeros((), device=dev)
        self.peak_max = torch.zeros((), device=dev)
        self.nonfinite = torch.zeros((), device=dev)
        self.any_over = torch.zeros(len(self.thresholds_deg), device=dev)
        self.joint_over = torch.zeros(len(self.thresholds_deg), n, device=dev)

    def sample(self) -> torch.Tensor:
        with torch.no_grad():
            data = self.robot.data
            target = data.joint_pos_target[:, self.joint_ids]
            position = data.joint_pos[:, self.joint_ids]
            error = torch.rad2deg(sent_tracking_error(target, position))
            peak = error.amax(dim=1)
            self.samples += peak.shape[0]
            self.peak_sum += peak.sum()
            self.peak_max = torch.maximum(self.peak_max, peak.amax())
            self.nonfinite += (~torch.isfinite(target - position)).sum()
            for index, threshold in enumerate(self.thresholds_deg):
                over = error > threshold
                self.any_over[index] += over.any(dim=1).sum()
                self.joint_over[index] += over.sum(dim=0)
        return error

    def take_log(self) -> dict[str, float]:
        samples = max(1.0, float(self.samples.item()))
        p = self.prefix
        values = {
            f"{p}sent_tracking_peak_deg_mean": self.peak_sum.item() / samples,
            f"{p}sent_tracking_peak_deg_max": self.peak_max.item(),
            f"{p}sent_tracking_nonfinite_fraction": self.nonfinite.item() / (samples * len(self.joint_names)),
        }
        for index, threshold in enumerate(self.thresholds_deg):
            tag = f"gt{threshold:g}"
            values[f"{p}sent_tracking_any_{tag}_fraction"] = self.any_over[index].item() / samples
            for j, name in enumerate(self.joint_names):
                values[f"{p}sent_tracking_{tag}_fraction/{name}"] = self.joint_over[index, j].item() / samples
        self._reset_accumulators()
        return values
