from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """Where the gait clock is, as (sin, cos) of the cycle phase.

    ``feet_gait`` rewards a contact pattern that matches this clock, and the
    clock is a function of time since reset. Without it in the observation the
    policy cannot tell which phase it is in, so the reward is unlearnable and
    sits at chance. H1 feeds it; G1 comments it out and leans on a much wider
    velocity command instead.
    """
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase
