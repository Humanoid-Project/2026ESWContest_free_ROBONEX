"""Evaluate a trained checkpoint on a fixed grid of velocity commands."""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a RoboNex walking policy per velocity command.")
parser.add_argument("--task", type=str, default="RoboNex-Walking-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--warmup_steps", type=int, default=120)
parser.add_argument("--measure_steps", type=int, default=300)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--cells", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import robonex_walking.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from robonex_walking.tasks.manager_based.robonex_walking.mdp.walk_metrics import WalkMetrics
from robonex_walking.tasks.manager_based.robonex_walking.robot_contract import FOOT_ORIGIN_REST_HEIGHT

CELLS = {
    "fwd_slow": (0.1, 0.0, 0.0),
    "fwd_max": (0.3, 0.0, 0.0),
    "stop": (0.0, 0.0, 0.0),
    "back": (-0.2, 0.0, 0.0),
    "turn_l": (0.1, 0.0, 0.4),
    "turn_r": (0.1, 0.0, -0.4),
    "turn_still": (0.0, 0.0, 0.4),
    "strafe_l": (0.0, 0.2, 0.0),
    "strafe_r": (0.0, -0.2, 0.0),
    "diag": (0.2, 0.1, 0.2),
}


def yaw_frame_lin_vel(env):
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    asset = env.scene["robot"]
    return quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # never resample or zero a command: the harness drives it directly
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    term = unwrapped.command_manager.get_term("base_velocity")
    forced = torch.zeros(unwrapped.num_envs, 3, device=unwrapped.device)
    term._resample_command = lambda env_ids: None
    term._update_command = lambda: term.vel_command_b.copy_(forced)

    metrics = WalkMetrics(unwrapped, FOOT_ORIGIN_REST_HEIGHT, unwrapped.step_dt)

    action_term = unwrapped.action_manager.get_term("joint_pos")
    joint_names = list(action_term._joint_names)
    act_scale = torch.as_tensor(action_term._scale, device=unwrapped.device)
    act_offset = torch.as_tensor(action_term._offset, device=unwrapped.device)
    if act_scale.dim() == 1:
        act_scale = act_scale.reshape(1, -1)
    if act_offset.dim() == 1:
        act_offset = act_offset.reshape(1, -1)
    clip_low = action_term._clip[..., 0]
    clip_high = action_term._clip[..., 1]

    wanted = args_cli.cells.split(",") if args_cli.cells else list(CELLS)
    results = {}
    inference = torch.inference_mode()
    inference.__enter__()
    for name in wanted:
        command = CELLS[name]
        forced[:] = torch.tensor(command, device=unwrapped.device)
        env.reset()
        obs = env.get_observations()
        metrics._reset_state()
        metrics.take_log()  # clear accumulators

        err = torch.zeros(3, device=unwrapped.device)
        got = torch.zeros(3, device=unwrapped.device)
        falls = torch.zeros((), device=unwrapped.device)
        n_joints = len(joint_names)
        clip_hits = torch.zeros(n_joints, device=unwrapped.device)
        worst_excess = torch.zeros(n_joints, device=unwrapped.device)
        min_margin = torch.full((n_joints,), float("inf"), device=unwrapped.device)
        counted = torch.zeros((), device=unwrapped.device)
        target = torch.tensor(command, device=unwrapped.device)
        for step in range(args_cli.warmup_steps + args_cli.measure_steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            forced[:] = target
            # replicate the deployed pipeline: runner clip, scale+offset, then the target fence
            pre = torch.clamp(actions, -agent_cfg.clip_actions, agent_cfg.clip_actions)
            pre = pre * act_scale + act_offset
            over = torch.clamp(pre - clip_high, min=0.0) + torch.clamp(clip_low - pre, min=0.0)
            margin = torch.minimum(clip_high - pre, pre - clip_low)
            if step >= args_cli.warmup_steps:
                clip_hits += (over > 0.0).float().mean(dim=0)
                worst_excess = torch.maximum(worst_excess, over.amax(dim=0))
                min_margin = torch.minimum(min_margin, margin.amin(dim=0))
            if step < args_cli.warmup_steps:
                metrics._reset_state()
                metrics.take_log()
                continue
            metrics.update(unwrapped)
            lin = yaw_frame_lin_vel(unwrapped)
            ang = unwrapped.scene["robot"].data.root_ang_vel_b
            err[0] += (lin[:, 0] - command[0]).abs().mean()
            err[1] += (lin[:, 1] - command[1]).abs().mean()
            err[2] += (ang[:, 2] - command[2]).abs().mean()
            got[0] += lin[:, 0].mean()
            got[1] += lin[:, 1].mean()
            got[2] += ang[:, 2].mean()
            falls += unwrapped.termination_manager.get_term("fall_down").float().mean()
            counted += 1.0

        counted = torch.clamp(counted, min=1.0)
        row = {k.split("/", 1)[-1]: v for k, v in metrics.take_log().items() if k.startswith("Gait/")}
        row["err_vx"] = (err[0] / counted).item()
        row["err_vy"] = (err[1] / counted).item()
        row["err_wz"] = (err[2] / counted).item()
        row["got_vx"] = (got[0] / counted).item()
        row["got_vy"] = (got[1] / counted).item()
        row["got_wz"] = (got[2] / counted).item()
        row["fall_rate"] = (falls / counted).item()
        frac = (clip_hits / counted)
        row["clip_frac"] = {n: frac[i].item() for i, n in enumerate(joint_names)}
        row["clip_frac_max"] = frac.amax().item()
        row["clip_frac_max_joint"] = joint_names[int(frac.argmax())]
        row["worst_excess_rad"] = {n: worst_excess[i].item() for i, n in enumerate(joint_names)}
        row["min_margin_rad"] = {n: min_margin[i].item() for i, n in enumerate(joint_names)}
        row["min_margin_rad_min"] = min_margin.amin().item()
        row["min_margin_rad_min_joint"] = joint_names[int(min_margin.argmin())]
        row["command"] = list(command)
        results[name] = row
        print(
            "CELL %-11s cmd %+.2f/%+.2f/%+.2f got %+.3f/%+.3f/%+.3f | err %.3f/%.3f/%.3f | "
            "fall %.4f | swing %.4f single %.3f duty %.3f td_hz %.2f | "
            "clip %.4f@%s margin %+.4f@%s"
            % (name, command[0], command[1], command[2], row["got_vx"], row["got_vy"], row["got_wz"],
               row["err_vx"], row["err_vy"], row["err_wz"],
               row["fall_rate"], row["swing_peak_m"], row["single_stance_frac"], row["duty_l"],
               row["touchdown_hz_l"],
               row["clip_frac_max"], row["clip_frac_max_joint"].replace("_joint", ""),
               row["min_margin_rad_min"], row["min_margin_rad_min_joint"].replace("_joint", "")),
            flush=True,
        )

    inference.__exit__(None, None, None)

    if args_cli.out:
        with open(args_cli.out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print("wrote %s" % args_cli.out)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
