from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids,
    reward_term_name: str = "track_lin_vel_x",
    command_name: str = "base_velocity",
    delta: float = 0.1,
    success_ratio: float = 0.8,
) -> float:
    """Widen the velocity command once tracking is good enough.

    The manager only runs on a reset batch, so gating on an exact
    ``common_step_counter % max_episode_length == 0`` match silently never fires
    when resets drift off that boundary. Track the last update step instead.
    """
    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
    reward = reward / env.max_episode_length_s

    last = getattr(env, "_lin_vel_cmd_last_step", None)
    if last is None:
        last = env._lin_vel_cmd_last_step = 0
    due = env.common_step_counter - last >= env.max_episode_length
    if due and reward > reward_term.weight * success_ratio:
        env._lin_vel_cmd_last_step = env.common_step_counter
        low, high = ranges.lin_vel_x
        lo_limit, hi_limit = limit_ranges.lin_vel_x
        ranges.lin_vel_x = (max(low - delta, lo_limit), min(high + delta, hi_limit))
    return float(ranges.lin_vel_x[1])
