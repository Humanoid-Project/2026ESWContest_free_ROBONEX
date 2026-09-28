import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def reset_closed_loop_to_default(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(asset.data.default_joint_vel[env_ids])
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    active_pos = joint_pos[:, asset_cfg.joint_ids]
    active_vel = joint_vel[:, asset_cfg.joint_ids]
    asset.set_joint_position_target(
        active_pos,
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,
    )
    asset.set_joint_velocity_target(
        active_vel,
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,
    )


def push_standing_by_setting_velocity(
    env,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    command_name: str = "base_velocity",
    command_deadband: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    command = env.command_manager.get_command(command_name)[env_ids]
    standing = env_ids[torch.linalg.norm(command, dim=1) < command_deadband]
    if standing.numel() == 0:
        return
    asset: Articulation = env.scene[asset_cfg.name]
    vel_w = asset.data.root_vel_w[standing].clone()
    for index, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
        low, high = velocity_range.get(key, (0.0, 0.0))
        vel_w[:, index] += torch.empty(standing.numel(), device=vel_w.device).uniform_(low, high)
    asset.write_root_velocity_to_sim(vel_w, env_ids=standing)
