"""Record a per-step joint trace in sim, in the same column layout as the hardware telemetry.

Written so a sim run and a `policy_to_real.py --telemetry` run can go through one analysis.
"""

import argparse
import csv
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a sim joint trace comparable to hardware telemetry.")
parser.add_argument("--task", type=str, default="RoboNex-Walking-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--vx", type=float, default=0.0)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--warmup_steps", type=int, default=150)
parser.add_argument("--measure_steps", type=int, default=1650)
parser.add_argument("--env_index", type=int, default=0, help="Which env's trace to write")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.utils.math as math_utils
import robonex_walking.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    # long enough that the measurement window never crosses a timeout reset
    env_cfg.episode_length_s = 1.0e6

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    term = unwrapped.command_manager.get_term("base_velocity")
    forced = torch.zeros(unwrapped.num_envs, 3, device=unwrapped.device)
    forced[:, 0] = args_cli.vx
    forced[:, 1] = args_cli.vy
    forced[:, 2] = args_cli.wz
    term._resample_command = lambda env_ids: None
    term._update_command = lambda: term.vel_command_b.copy_(forced)

    asset = unwrapped.scene["robot"]
    action_term = unwrapped.action_manager.get_term("joint_pos")
    joint_names = list(action_term._joint_names)
    joint_ids = action_term._joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(len(joint_names)))

    # Bodies and the contact sensor, for the lateral-balance decomposition.
    # The stance hip_roll moment is CoP_offset * F_vertical + F_lateral * lever_arm, and the
    # joint trace alone cannot separate them: it records the sum. Net contact force per foot
    # gives the second term directly, so the first follows by difference.
    body_names = list(asset.data.body_names)
    foot_ids = [body_names.index(n) for n in ("l_foot", "r_foot")]
    hip_ids = [body_names.index(n) for n in ("l_hip_roll_link", "r_hip_roll_link")]
    contact = unwrapped.scene.sensors["contact_forces"]
    contact_bodies = list(contact.body_names)
    contact_ids = [contact_bodies.index(n) for n in ("l_foot", "r_foot")]

    # Mirrors of the reward-side quantities that world-frame columns cannot reconstruct.
    # feet_stance_width_l2 reads the foot separation in the BASE frame; a yawing robot makes
    # the world-frame separation a different number entirely.
    from robonex_walking.tasks.manager_based.robonex_walking.robot_contract import FOOT_SOLE_CORNERS
    from robonex_common.runtime import GAIT_PERIOD_S
    sole = torch.as_tensor(FOOT_SOLE_CORNERS, dtype=torch.float32, device=unwrapped.device)
    stance_fraction = 0.55
    # _clip, _scale and _offset may or may not carry a leading env dimension depending on
    # how the term was configured; reduce each to this one env's per-joint row.
    def _row(value):
        if value is None:
            return None
        t = torch.as_tensor(value, dtype=torch.float32, device=unwrapped.device)
        while t.ndim > 0 and t.shape[0] == unwrapped.num_envs and t.ndim > 1:
            t = t[args_cli.env_index]
        return t

    clip = _row(getattr(action_term, "_clip", None))
    act_scale = _row(action_term._scale)
    act_offset = _row(action_term._offset)
    print(f"[trace] clip={None if clip is None else tuple(clip.shape)} "
          f"scale={tuple(act_scale.shape)} offset={tuple(act_offset.shape)}")

    short = [n.replace("_joint", "") for n in joint_names]
    header = ["t_s", "step", "dt_ms", "ramp", "cmd_vx", "cmd_vy", "cmd_wz",
              "gravity_x", "gravity_y", "gravity_z", "gyro_x", "gyro_y", "gyro_z",
              "root_x", "root_y", "root_z", "root_vx_b", "root_vy_b",
              "root_vx_w", "root_vy_w",
              "l_foot_x", "l_foot_y", "l_foot_z", "r_foot_x", "r_foot_y", "r_foot_z",
              "l_hip_y", "l_hip_z", "r_hip_y", "r_hip_z",
              "l_foot_fx", "l_foot_fy", "l_foot_fz",
              "r_foot_fx", "r_foot_fy", "r_foot_fz",
              "root_vz_b", "l_foot_by", "r_foot_by", "stance_width_b",
              "l_sole_z", "r_sole_z", "l_phase", "r_phase"]
    for s in short:
        header += [f"{s}.pos", f"{s}.vel", f"{s}.torque", f"{s}.target",
                   f"{s}.raw_action", f"{s}.clip_excess"]

    idx = args_cli.env_index
    dt = float(unwrapped.step_dt)
    rows = []
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.warmup_steps + args_cli.measure_steps):
            actions = policy(obs)
            raw = actions[idx].clone()
            obs, _, _, _ = env.step(actions)
            if step < args_cli.warmup_steps:
                continue
            n = step - args_cli.warmup_steps
            grav = asset.data.projected_gravity_b[idx]
            gyro = asset.data.root_ang_vel_b[idx]
            root_p = asset.data.root_pos_w[idx]
            root_v = asset.data.root_lin_vel_b[idx]
            body_p = asset.data.body_pos_w[idx]
            force = contact.data.net_forces_w[idx]
            origin = unwrapped.scene.env_origins[idx]
            row = [round(n * dt, 4), n, round(dt * 1000.0, 3), 1.0,
                   args_cli.vx, args_cli.vy, args_cli.wz,
                   *[round(float(v), 5) for v in grav],
                   *[round(float(v), 5) for v in gyro],
                   *[round(float(root_p[k] - origin[k]), 5) for k in range(3)],
                   round(float(root_v[0]), 5), round(float(root_v[1]), 5),
                   round(float(asset.data.root_lin_vel_w[idx, 0]), 5),
                   round(float(asset.data.root_lin_vel_w[idx, 1]), 5)]
            for b in foot_ids:
                row += [round(float(body_p[b, k] - origin[k]), 5) for k in range(3)]
            for b in hip_ids:
                row += [round(float(body_p[b, k] - origin[k]), 5) for k in (1, 2)]
            for c in contact_ids:
                row += [round(float(force[c, k]), 4) for k in range(3)]

            quat = asset.data.root_quat_w[idx]
            rel = body_p[foot_ids] - root_p.unsqueeze(0)
            foot_b = math_utils.quat_apply_inverse(quat.unsqueeze(0).expand(2, -1), rel)
            fq = asset.data.body_quat_w[idx, foot_ids]
            corner_z = math_utils.quat_apply(
                fq.unsqueeze(1).expand(-1, sole.shape[0], -1), sole.unsqueeze(0).expand(2, -1, -1)
            )[..., 2]
            sole_z = (body_p[foot_ids, 2].unsqueeze(-1) + corner_z).amin(dim=-1) - origin[2]
            gphase = (float(unwrapped.episode_length_buf[idx]) * dt) % GAIT_PERIOD_S / GAIT_PERIOD_S
            row += [round(float(root_v[2]), 5),
                    round(float(foot_b[0, 1]), 5), round(float(foot_b[1, 1]), 5),
                    round(float(torch.abs(foot_b[0, 1] - foot_b[1, 1])), 5),
                    round(float(sole_z[0]), 5), round(float(sole_z[1]), 5),
                    round(gphase % 1.0, 5), round((gphase + 0.5) % 1.0, 5)]

            pos = asset.data.joint_pos[idx, joint_ids]
            vel = asset.data.joint_vel[idx, joint_ids]
            tau = asset.data.applied_torque[idx, joint_ids]
            tgt = asset.data.joint_pos_target[idx, joint_ids]
            pre = act_offset + act_scale * raw
            if clip is not None:
                excess = pre - torch.clamp(pre, clip[..., 0], clip[..., 1])
            else:
                excess = torch.zeros_like(pre)
            excess = excess.reshape(-1)
            for j in range(len(joint_names)):
                row += [round(float(pos[j]), 5), round(float(vel[j]), 5),
                        round(float(tau[j]), 5), round(float(tgt[j]), 5),
                        round(float(raw[j]), 5), round(float(excess[j]), 6)]
            rows.append(row)

    out = os.path.abspath(args_cli.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows ({len(rows) * dt:.1f} s at {1/dt:.0f} Hz) to {out}")
    env.close()


main()
simulation_app.close()
