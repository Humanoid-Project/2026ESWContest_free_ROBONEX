"""Record a per-step joint trace in sim, in the same column layout as the hardware telemetry.

Written so a sim run and a `policy_to_real.py --telemetry` run can go through one analysis.
"""

import argparse
import copy
import csv
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a sim joint trace comparable to hardware telemetry.")
parser.add_argument("--task", type=str, default="RoboNex-Walking-V2-Edu-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--vx", type=float, default=0.0)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--scenario", type=str, default=None, metavar="T:VX,VY,WZ;...",
                    help="Command schedule in seconds from the start of the measurement, e.g. 0:0,0,0;10:0.1,0,0;20:0.2,0,0; "
                    "replaces --vx/--vy/--wz (the warm-up uses the first command)")
parser.add_argument("--warmup_steps", type=int, default=150)
parser.add_argument("--measure_steps", type=int, default=1650)
parser.add_argument("--env_index", type=int, default=0, help="Which env's trace to write")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--registry_cfg", action="store_true",
                    help="Use the current task config instead of the checkpoint's params/env.yaml")
parser.add_argument("--keep_randomization", action="store_true",
                    help="Keep domain randomization and pushes (default: only the reset events)")
parser.add_argument("--slew_limit", action="store_true",
                    help="Apply the deploy AxisLimiter (6 rad/s, 120 rad/s^2) to the joint targets")
parser.add_argument("--com_pin", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="Fix the base CoM offset from nominal (m) on every env; default without this flag: nominal (0 0 0)")
parser.add_argument("--com_keep_training", action="store_true",
                    help="Keep the base CoM randomisation range of the config instead of pinning it to nominal")
parser.add_argument("--load_pin", type=float, nargs=6, default=None, metavar=("FX", "FY", "FZ", "TX", "TY", "TZ"),
                    help="Fix the sustained base_link load (N, N*m, base link frame, at the link origin) on every env; "
                         "default without this flag: no load")
parser.add_argument("--load_keep_training", action="store_true",
                    help="Keep the sustained base load range of the config instead of pinning it to zero")
parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                    help="Override a float in the env config, e.g. scene.robot.actuators.rs02.stiffness.l_hip_yaw_joint=25")
parser.add_argument("--actuation_delay", type=int, nargs=2, default=None, metavar=("LOW", "HIGH"),
                    help="Pin the action term's actuation delay to U{LOW..HIGH} physics substeps for every env (0 0 = off)")
parser.add_argument("--obs_delay", type=int, default=0,
                    help="Delay joint_pos_rel and joint_vel_rel in the policy observation by N steps")
parser.add_argument("--obs_delay_pin", type=int, default=-1,
                    help="For a policy trained with a random joint observation delay: fix that delay at N steps for every env")
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
from deploy_effects import (
    BASE_COM_EVENT,
    BASE_LOAD_EVENT,
    apply_base_com_pin,
    apply_actuation_delay,
    apply_base_load_pin,
    apply_training_env_cfg,
    disable_randomization,
    install_joint_obs_delay,
    install_slew_limiter,
    observation_delay_state,
    pin_joint_obs_delay,
    widen_joint_obs_delay,
    summarize_delay_states,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def sample_before_reset(unwrapped, env_index, sample):
    terminal = []
    compute = unwrapped.termination_manager.compute

    def compute_and_sample():
        result = compute()
        if bool(result[env_index]):
            terminal.append(sample())
        return result

    unwrapped.termination_manager.compute = compute_and_sample
    return terminal


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    com_template = copy.deepcopy(getattr(env_cfg.events, BASE_COM_EVENT, None))
    load_template = copy.deepcopy(getattr(env_cfg.events, BASE_LOAD_EVENT, None))
    if not args_cli.registry_cfg:
        print(f"[trace] training config: {apply_training_env_cfg(env_cfg, args_cli.checkpoint)}")
        env_cfg.scene.num_envs = args_cli.num_envs
    if not args_cli.keep_randomization:
        com_kept = (BASE_COM_EVENT,) if args_cli.com_keep_training or args_cli.com_pin is not None else ()
        load_kept = (BASE_LOAD_EVENT,) if args_cli.load_keep_training or args_cli.load_pin is not None else ()
        print(f"[trace] randomization off; kept events: {disable_randomization(env_cfg, com_kept + load_kept)}")
    for item in args_cli.set:
        path, value = item.split("=", 1)
        *parents, leaf = path.split(".")
        node = env_cfg
        for key in parents:
            node = node[key] if isinstance(node, dict) else getattr(node, key)
        if isinstance(node, dict):
            if leaf not in node:
                raise KeyError(f"--set {path}: {leaf!r} not in {sorted(node)}")
            node[leaf] = float(value)
        else:
            if not hasattr(node, leaf):
                raise AttributeError(f"--set {path}: no attribute {leaf!r}")
            setattr(node, leaf, float(value))
        print(f"[trace] set {path} = {float(value)}")
    base_com = apply_base_com_pin(env_cfg, args_cli.com_pin, com_template, args_cli.com_keep_training)
    print(f"[trace] base CoM: {base_com}")
    base_load = apply_base_load_pin(env_cfg, args_cli.load_pin, load_template, args_cli.load_keep_training)
    print(f"[trace] base load: {base_load}")
    actuation_delay = apply_actuation_delay(env_cfg, args_cli.actuation_delay)
    print(f"[trace] actuation delay: {actuation_delay}")
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    # long enough that the measurement window never crosses a timeout reset
    env_cfg.episode_length_s = 1.0e6

    obs_widened = widen_joint_obs_delay(env_cfg, args_cli.obs_delay_pin)
    if obs_widened:
        print(f"[trace] joint observation delay buffer widened for the pin: {obs_widened}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    if args_cli.slew_limit:
        install_slew_limiter(unwrapped)
        print("[trace] deploy slew limiter on joint targets")
    obs_delayed = install_joint_obs_delay(unwrapped, args_cli.obs_delay) if args_cli.obs_delay else []
    if obs_delayed:
        print(f"[trace] observation delay {args_cli.obs_delay} step(s) on {obs_delayed}")
    obs_pinned = pin_joint_obs_delay(unwrapped, args_cli.obs_delay_pin) if args_cli.obs_delay_pin >= 0 else []
    if obs_pinned:
        print(f"[trace] joint observation delay pinned at {args_cli.obs_delay_pin} step(s) on {obs_pinned}")
    term = unwrapped.command_manager.get_term("base_velocity")
    schedule = [(0.0, (args_cli.vx, args_cli.vy, args_cli.wz))]
    if args_cli.scenario:
        schedule = []
        for part in args_cli.scenario.split(";"):
            start, values = part.split(":")
            schedule.append((float(start), tuple(float(v) for v in values.split(","))))
        schedule.sort()
        if schedule[0][0] != 0.0 or any(len(v) != 3 for _, v in schedule):
            raise ValueError("--scenario must start at 0 and give vx,vy,wz for every segment")

    def command_at(t):
        return [v for start, v in schedule if start <= t][-1]

    forced = torch.zeros(unwrapped.num_envs, 3, device=unwrapped.device)
    forced[:] = torch.tensor(command_at(0.0), device=unwrapped.device)
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
    from robonex_common.runtime import GAIT_PERIOD_S
    sole = torch.as_tensor(unwrapped.cfg.foot_sole_corners, dtype=torch.float32, device=unwrapped.device)
    stance_fraction = 0.55
    # _clip, _scale and _offset may or may not carry a leading env dimension depending on
    # how the term was configured; reduce each to this one env's per-joint row.
    def _row(value, ndim):
        if value is None:
            return None
        t = torch.as_tensor(value, dtype=torch.float32, device=unwrapped.device)
        while t.ndim > ndim:
            t = t[args_cli.env_index] if t.shape[0] == unwrapped.num_envs else t[0]
        return t

    clip = _row(getattr(action_term, "_clip", None), 2)
    act_scale = _row(action_term._scale, 1)
    act_offset = _row(action_term._offset, 1)
    runner_clip = float(agent_cfg.clip_actions) if agent_cfg.clip_actions else float("inf")
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
              "l_sole_z", "r_sole_z", "l_phase", "r_phase", "reset"]
    for s in short:
        header += [f"{s}.pos", f"{s}.vel", f"{s}.torque", f"{s}.target",
                   f"{s}.raw_action", f"{s}.clip_excess"]

    idx = args_cli.env_index
    dt = float(unwrapped.step_dt)
    rows = []
    delay_states = []

    def sample_state():
        return {
            "delay": observation_delay_state(unwrapped, idx),
            "grav": asset.data.projected_gravity_b[idx].clone(),
            "gyro": asset.data.root_ang_vel_b[idx].clone(),
            "root_p": asset.data.root_pos_w[idx].clone(),
            "root_v": asset.data.root_lin_vel_b[idx].clone(),
            "root_v_w": asset.data.root_lin_vel_w[idx].clone(),
            "body_p": asset.data.body_pos_w[idx].clone(),
            "force": contact.data.net_forces_w[idx].clone(),
            "quat": asset.data.root_quat_w[idx].clone(),
            "foot_quat": asset.data.body_quat_w[idx, foot_ids].clone(),
            "episode_length": float(unwrapped.episode_length_buf[idx]),
            "pos": asset.data.joint_pos[idx, joint_ids].clone(),
            "vel": asset.data.joint_vel[idx, joint_ids].clone(),
            "tau": asset.data.applied_torque[idx, joint_ids].clone(),
            "tgt": asset.data.joint_pos_target[idx, joint_ids].clone(),
        }

    terminal = sample_before_reset(unwrapped, idx, sample_state)
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.warmup_steps + args_cli.measure_steps):
            command = command_at(max(0, step - args_cli.warmup_steps) * dt)
            forced[:] = torch.tensor(command, device=unwrapped.device)
            actions = policy(obs)
            raw = actions[idx].clone()
            obs, _, dones, _ = env.step(actions)
            state = terminal.pop() if terminal else None
            if step < args_cli.warmup_steps:
                continue
            n = step - args_cli.warmup_steps
            if state is None:
                state = sample_state()
            delay_states.append(state["delay"])
            grav = state["grav"]
            gyro = state["gyro"]
            root_p = state["root_p"]
            root_v = state["root_v"]
            body_p = state["body_p"]
            force = state["force"]
            origin = unwrapped.scene.env_origins[idx]
            row = [round(n * dt, 4), n, round(dt * 1000.0, 3), 1.0,
                   *command,
                   *[round(float(v), 5) for v in grav],
                   *[round(float(v), 5) for v in gyro],
                   *[round(float(root_p[k] - origin[k]), 5) for k in range(3)],
                   round(float(root_v[0]), 5), round(float(root_v[1]), 5),
                   round(float(state["root_v_w"][0]), 5),
                   round(float(state["root_v_w"][1]), 5)]
            for b in foot_ids:
                row += [round(float(body_p[b, k] - origin[k]), 5) for k in range(3)]
            for b in hip_ids:
                row += [round(float(body_p[b, k] - origin[k]), 5) for k in (1, 2)]
            for c in contact_ids:
                row += [round(float(force[c, k]), 4) for k in range(3)]

            quat = state["quat"]
            rel = body_p[foot_ids] - root_p.unsqueeze(0)
            foot_b = math_utils.quat_apply_inverse(quat.unsqueeze(0).expand(2, -1), rel)
            fq = state["foot_quat"]
            corner_z = math_utils.quat_apply(
                fq.unsqueeze(1).expand(-1, sole.shape[0], -1), sole.unsqueeze(0).expand(2, -1, -1)
            )[..., 2]
            sole_z = (body_p[foot_ids, 2].unsqueeze(-1) + corner_z).amin(dim=-1) - origin[2]
            gphase = (state["episode_length"] * dt) % GAIT_PERIOD_S / GAIT_PERIOD_S
            row += [round(float(root_v[2]), 5),
                    round(float(foot_b[0, 1]), 5), round(float(foot_b[1, 1]), 5),
                    round(float(torch.abs(foot_b[0, 1] - foot_b[1, 1])), 5),
                    round(float(sole_z[0]), 5), round(float(sole_z[1]), 5),
                    round(gphase % 1.0, 5), round((gphase + 0.5) % 1.0, 5), int(dones[idx])]

            pos = state["pos"]
            vel = state["vel"]
            tau = state["tau"]
            tgt = state["tgt"]
            pre = act_offset + act_scale * torch.clamp(raw, -runner_clip, runner_clip)
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
    meta = {
        "checkpoint": os.path.abspath(args_cli.checkpoint),
        "task": args_cli.task,
        "seed": args_cli.seed,
        "num_envs": unwrapped.num_envs,
        "env_index": idx,
        "command": [args_cli.vx, args_cli.vy, args_cli.wz],
        "scenario": args_cli.scenario,
        "warmup_steps": args_cli.warmup_steps,
        "measure_steps": args_cli.measure_steps,
        "rows": len(rows),
        "step_dt": dt,
        "training_cfg": not args_cli.registry_cfg,
        "keep_randomization": bool(args_cli.keep_randomization),
        "slew_limit": bool(args_cli.slew_limit),
        "set": list(args_cli.set),
        "obs_delay_steps": int(args_cli.obs_delay),
        "obs_delay_terms": obs_delayed,
        "obs_delay_pin": int(args_cli.obs_delay_pin),
        "obs_delay_pinned_terms": obs_pinned,
        "obs_delay_widened": obs_widened,
        "delays": summarize_delay_states(delay_states),
        "base_com": base_com,
        "base_load": base_load,
        "actuation_delay": actuation_delay,
        "actuation_delay_env_k": None if actuation_delay["substeps"] is None
        else int(action_term._delay_substeps[idx]),
    }
    meta_path = out + ".meta.json"
    with open(meta_path + ".partial", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    os.replace(meta_path + ".partial", meta_path)
    print(f"[trace] delays used: {json.dumps(meta['delays'])}")
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows ({len(rows) * dt:.1f} s at {1/dt:.0f} Hz) to {out}")
    env.close()


main()
simulation_app.close()
