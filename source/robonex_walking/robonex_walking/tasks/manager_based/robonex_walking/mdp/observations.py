from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(
    env: ManagerBasedRLEnv,
    period: float,
    command_name: str | None = None,
    command_deadband: float = 0.05,
) -> torch.Tensor:
    """Where the gait clock is, as (sin, cos) of the cycle phase.

    ``feet_gait`` rewards a contact pattern that matches this clock, and the
    clock is a function of time since reset. Without it in the observation the
    policy cannot tell which phase it is in, so the reward is unlearnable and
    sits at chance. H1 feeds it; G1 comments it out and leans on a much wider
    velocity command instead.

    Zeroed on a standing command. Every reward that consumes the clock is already
    gated off at zero command, but the observation was not, so the clock kept
    turning and the policy kept answering it: measured standing still after a
    0.3 m/s walk, the knees swung 37.4 deg peak-to-peak at 1.22 Hz against a
    1.25 Hz clock, one foot re-touching every cycle, for 26 W of mechanical power
    while the feet were supposed to be planted. Holding the pair at zero removes
    the cue instead of asking a later penalty to out-shout it. The width stays 2,
    so the deploy observation contract is unchanged.
    """
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        moving = (torch.linalg.norm(command, dim=1) > command_deadband).unsqueeze(-1)
        phase = phase * moving
    return phase
