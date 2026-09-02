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
