"""Evaluate a trained checkpoint on a fixed grid of velocity commands."""

import argparse
import hashlib
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a RoboNex walking policy per velocity command.")
parser.add_argument("--task", type=str, default="RoboNex-Walking-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--warmup_steps", type=int, default=120)
parser.add_argument("--measure_steps", type=int, default=880)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--cells", type=str, default=None)
parser.add_argument("--seed", type=int, default=0, help="Environment seed; keep fixed across compared checkpoints")
parser.add_argument("--strict", action="store_true",
                    help="Exit instead of evaluating when the live env disagrees with the checkpoint's params/env.yaml")
parser.add_argument("--training_cfg", action="store_true",
                    help="Build the env from the checkpoint's params/env.yaml instead of the current code")
parser.add_argument("--slew_limit", action="store_true",
                    help="Apply the deploy AxisLimiter (6 rad/s, 120 rad/s^2) to the joint targets")
parser.add_argument("--obs_delay", type=int, default=0,
                    help="Delay joint_pos_rel and joint_vel_rel in the policy observation by N steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import pathlib
import torch
from rsl_rl.runners import OnPolicyRunner

import robonex_walking.tasks  # noqa: F401
from deploy_effects import apply_training_env_cfg, install_joint_obs_delay, install_slew_limiter
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from robonex_walking.tasks.manager_based.robonex_walking.mdp.walk_metrics import WalkMetrics
from robonex_walking.tasks.manager_based.robonex_walking.robot_contract import (
    FOOT_ORIGIN_REST_HEIGHT,
    FOOT_SOLE_CORNERS,
)

CELLS = {
    "fwd_slow": (0.1, 0.0, 0.0),
    "fwd_max": (0.3, 0.0, 0.0),
    "fwd_goal": (0.5, 0.0, 0.0),
    "stop": (0.0, 0.0, 0.0),
    "back": (-0.2, 0.0, 0.0),
    "turn_l": (0.1, 0.0, 0.2),
    "turn_r": (0.1, 0.0, -0.2),
    "turn_still": (0.0, 0.0, 0.2),
    "strafe_l": (0.0, 0.2, 0.0),
    "strafe_r": (0.0, -0.2, 0.0),
    "diag": (0.2, 0.1, 0.2),
    "turn_walk_l": (0.3, 0.0, 0.2),
    "turn_walk_r": (0.3, 0.0, -0.2),
}


def yaw_frame_lin_vel(env):
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    asset = env.scene["robot"]
    return quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w)


_MISSING = object()
_SKIP = object()


def _scalar_param(value):
    """Comparable form of a reward param, or _SKIP for anything not worth diffing.

    params carry SceneEntityCfg objects and tensors that never compare cleanly across a
    yaml round-trip; only plain scalars and flat scalar sequences are diffed.
    """
    if value is _MISSING:
        return None
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 9)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        items = [_scalar_param(v) for v in value]
        if any(i is _SKIP for i in items):
            return _SKIP
        return tuple(items)
    return _SKIP


def _diff_config(saved, live, path="", out=None, depth=0):
    """Recursively report scalar leaves that differ between two config dicts.

    Both sides come from isaaclab's ``class_to_dict``, so nested SceneEntityCfg, tuples and
    slices already have the same shape on each side and compare directly.
    """
    if out is None:
        out = []
    if depth > 8:
        return out
    if isinstance(saved, dict) and isinstance(live, dict):
        for key in sorted(set(saved) | set(live)):
            here = f"{path}.{key}" if path else str(key)
            if key not in saved:
                out.append(f"{here}: added, evaluating {live[key]!r}")
            elif key not in live:
                out.append(f"{here}: trained {saved[key]!r}, removed")
            else:
                _diff_config(saved[key], live[key], here, out, depth + 1)
        return out
    if isinstance(saved, (list, tuple)) and isinstance(live, (list, tuple)):
        if len(saved) != len(live):
            out.append(f"{path}: trained {saved!r}, evaluating {live!r}")
        else:
            for i, (a, b) in enumerate(zip(saved, live)):
                _diff_config(a, b, f"{path}[{i}]", out, depth + 1)
        return out
    if isinstance(saved, float) and isinstance(live, float):
        if abs(saved - live) > 1e-9:
            out.append(f"{path}: trained {saved}, evaluating {live}")
        return out
    try:
        same = bool(saved == live)
    except Exception:
        same = True
    if not same:
        out.append(f"{path}: trained {saved!r}, evaluating {live!r}")
    return out


def _json_safe(obj, depth=0):
    """Recursively make a config dict JSON-serialisable.

    class_to_dict leaves slices and arbitrary objects in place; both sides of the comparison
    get the same treatment so diffing still works on the sanitised form.
    """
    if depth > 8:
        return repr(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in obj]
    return repr(obj)


def _live_events(unwrapped):
    """Event/domain-randomization config as a plain dict, matching the saved env.yaml shape."""
    from isaaclab.utils import class_to_dict

    try:
        return _json_safe(class_to_dict(unwrapped.cfg.events))
    except Exception:
        return None


def _live_reward_specs(unwrapped):
    """Weights and scalar params per reward term, JSON-safe.

    Only params that survive _scalar_param are kept: the rest (SceneEntityCfg, tensors)
    are not comparable across a yaml round-trip and are not serialisable either.
    """
    manager = unwrapped.reward_manager
    specs = {}
    for name, cfg in zip(manager.active_terms, manager._term_cfgs):
        params = {}
        for key, value in (cfg.params or {}).items():
            scalar = _scalar_param(value)
            if scalar is _SKIP:
                continue
            params[key] = list(scalar) if isinstance(scalar, tuple) else scalar
        specs[name] = {"weight": float(cfg.weight), "params": params}
    return specs


def _live_gait_period(unwrapped):
    """The period the running env actually feeds the policy, read from the live term."""
    manager = unwrapped.observation_manager
    names = manager.active_terms["policy"]
    if "gait_phase" not in names:
        return None
    cfg = manager._group_obs_term_cfgs["policy"][names.index("gait_phase")]
    value = cfg.params.get("period")
    return float(value) if value is not None else None


def check_training_config(checkpoint, applied):
    """Compare the live env against params/env.yaml of the run that saved this checkpoint."""
    import yaml

    path = pathlib.Path(checkpoint).parent / "params" / "env.yaml"
    if not path.is_file():
        return ["params/env.yaml missing next to the checkpoint; cannot verify"]
    try:
        ref = yaml.unsafe_load(path.read_text())
    except Exception as error:  # the yaml carries pickled cfg objects
        return ["params/env.yaml unreadable: %s" % error]
    issues = []
    action = ref.get("actions", {}).get("joint_pos", {})
    for field, live in (("scale", applied["action_scale"]), ("offset", applied["action_offset"])):
        saved = action.get(field)
        if not isinstance(saved, dict):
            continue
        for name, value in live.items():
            if name not in saved:
                continue
            trained = float(saved[name])
            if abs(trained - value) > max(1e-6, 1e-6 * abs(trained)):
                issues.append(
                    "%s[%s] trained %.9f, evaluating %.9f" % (field, name, trained, value)
                )
    policy = ref.get("observations", {}).get("policy", {})
    history = policy.get("history_length")
    if history is not None:
        frame = applied["observation_shape"][-1] / max(int(history), 1)
        if abs(frame - round(frame)) > 1e-6:
            issues.append("observation %s not divisible by history_length %s"
                          % (applied["observation_shape"], history))

    # The clock is the highest-risk silent divergence: a changed period alters what the robot
    # does without changing any tensor shape, so nothing downstream can catch it.
    gait = policy.get("gait_phase", {}).get("params", {})
    trained_period = gait.get("period")
    if trained_period is not None and applied.get("gait_period_s") is not None:
        if abs(float(trained_period) - float(applied["gait_period_s"])) > 1e-9:
            issues.append("gait period trained %s, evaluating %s"
                          % (trained_period, applied["gait_period_s"]))

    for key, label in (("decimation", "decimation"), ("episode_length_s", "episode length")):
        saved = ref.get(key)
        live = applied.get(key)
        if saved is not None and live is not None and abs(float(saved) - float(live)) > 1e-9:
            issues.append("%s trained %s, evaluating %s" % (label, saved, live))

    saved_terms = list(policy.keys()) if isinstance(policy, dict) else []
    saved_terms = [t for t in saved_terms
                   if t not in ("history_length", "flatten_history_dim", "enable_corruption",
                                "concatenate_terms", "concatenate_dim")]
    live_terms = applied.get("observation_terms")
    if saved_terms and live_terms and list(live_terms) != saved_terms:
        issues.append("observation term order trained %s, evaluating %s" % (saved_terms, live_terms))

    saved_reward_cfgs = {k: v for k, v in (ref.get("rewards") or {}).items() if not k.startswith("_")}
    saved_rewards = sorted(saved_reward_cfgs)
    live_rewards = applied.get("reward_terms")
    if saved_rewards and live_rewards and sorted(live_rewards) != saved_rewards:
        added = sorted(set(live_rewards) - set(saved_rewards))
        dropped = sorted(set(saved_rewards) - set(live_rewards))
        issues.append("reward terms differ: added %s, dropped %s" % (added or "-", dropped or "-"))

    saved_events = _json_safe(ref.get("events"))
    live_events = applied.get("events")
    if isinstance(saved_events, dict) and isinstance(live_events, dict):
        for line in _diff_config(saved_events, live_events, "events"):
            issues.append(line)

    live_specs = applied.get("reward_specs") or {}
    for name in sorted(set(saved_reward_cfgs) & set(live_specs)):
        saved_cfg = saved_reward_cfgs[name]
        if not isinstance(saved_cfg, dict):
            continue
        live_spec = live_specs[name]
        saved_weight = saved_cfg.get("weight")
        live_weight = live_spec.get("weight")
        if saved_weight is not None and live_weight is not None:
            if abs(float(saved_weight) - float(live_weight)) > 1e-9:
                issues.append("reward %s weight trained %s, evaluating %s"
                              % (name, saved_weight, live_weight))
        saved_params = saved_cfg.get("params") or {}
        live_params = live_spec.get("params") or {}
        if not isinstance(saved_params, dict):
            continue
        for key in sorted(set(saved_params) | set(live_params)):
            if key not in live_params and _scalar_param(saved_params.get(key, _MISSING)) is _SKIP:
                continue
            saved_value = _scalar_param(saved_params.get(key, _MISSING))
            live_value = _scalar_param(live_params.get(key, _MISSING))
            if saved_value is _SKIP or live_value is _SKIP:
                continue
            if isinstance(saved_value, tuple):
                saved_value = list(saved_value)
            if isinstance(live_value, tuple):
                live_value = list(live_value)
            if saved_value != live_value:
                issues.append("reward %s param %s trained %r, evaluating %r"
                              % (name, key, saved_value, live_value))
    return issues


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.training_cfg:
        print(f"[eval] training config: {apply_training_env_cfg(env_cfg, args_cli.checkpoint)}")
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
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
    if args_cli.slew_limit:
        install_slew_limiter(unwrapped)
    obs_delayed = install_joint_obs_delay(unwrapped, args_cli.obs_delay) if args_cli.obs_delay else []
    term = unwrapped.command_manager.get_term("base_velocity")
    forced = torch.zeros(unwrapped.num_envs, 3, device=unwrapped.device)
    term._resample_command = lambda env_ids: None
    term._update_command = lambda: term.vel_command_b.copy_(forced)

    metrics = WalkMetrics(
        unwrapped,
        getattr(unwrapped.cfg, "foot_origin_rest_height", FOOT_ORIGIN_REST_HEIGHT),
        unwrapped.step_dt,
        sole_corners=getattr(unwrapped.cfg, "foot_sole_corners", FOOT_SOLE_CORNERS),
    )

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

    # The env is rebuilt from the CURRENT code, not from the run that produced the
    # checkpoint. Record what was actually applied and compare against the run's own
    # params/env.yaml so a silent scale or gain mismatch cannot pass unnoticed.
    checkpoint = os.path.abspath(args_cli.checkpoint)
    digest = hashlib.sha256(pathlib.Path(checkpoint).read_bytes()).hexdigest()[:16]
    applied = {
        "deploy_effects": {
            "training_cfg": bool(args_cli.training_cfg),
            "slew_limit": bool(args_cli.slew_limit),
            "obs_delay_steps": int(args_cli.obs_delay),
            "obs_delay_terms": obs_delayed,
        },
        "joint_order": joint_names,
        "action_scale": {n: float(act_scale[0, i]) for i, n in enumerate(joint_names)},
        "action_offset": {n: float(act_offset[0, i]) for i, n in enumerate(joint_names)},
        "action_clip": {
            n: [float(clip_low[0, i]), float(clip_high[0, i])]
            for i, n in enumerate(joint_names)
        },
        "observation_shape": list(unwrapped.observation_space["policy"].shape),
        "step_dt": float(unwrapped.step_dt),
        "runner_clip": float(agent_cfg.clip_actions),
        "observation_terms": list(unwrapped.observation_manager.active_terms["policy"]),
        "reward_terms": list(unwrapped.reward_manager.active_terms),
        "reward_specs": _live_reward_specs(unwrapped),
        "events": _live_events(unwrapped),
        "decimation": int(unwrapped.cfg.decimation),
        "episode_length_s": float(unwrapped.cfg.episode_length_s),
        "gait_period_s": _live_gait_period(unwrapped),
    }
    meta = {
        "checkpoint": checkpoint,
        "checkpoint_sha256_16": digest,
        "num_envs": unwrapped.num_envs,
        "warmup_steps": args_cli.warmup_steps,
        "measure_steps": args_cli.measure_steps,
        "seed": args_cli.seed,
        "applied": applied,
        "config_mismatch": check_training_config(checkpoint, applied),
    }
    for line in meta["config_mismatch"]:
        print("CONFIG MISMATCH %s" % line, flush=True)
    if meta["config_mismatch"] and args_cli.strict:
        raise SystemExit(
            "--strict: refusing to evaluate a checkpoint against a disagreeing environment"
        )
    if not meta["config_mismatch"]:
        print(
            "CONFIG match: action scale/offset, gait period, decimation, episode length, "
            "observation term order and reward term set agree with the checkpoint's "
            "params/env.yaml. Gains and command ranges are NOT compared.",
            flush=True,
        )

    wanted = args_cli.cells.split(",") if args_cli.cells else list(CELLS)
    results = {}
    inference = torch.inference_mode()
    inference.__enter__()
    for name in wanted:
        command = CELLS[name]
        forced[:] = torch.tensor(command, device=unwrapped.device)
        term.vel_command_b.copy_(forced)
        env.reset()
        obs = env.get_observations()
        metrics._reset_state()
        metrics.take_log()  # clear accumulators

        err = torch.zeros(3, device=unwrapped.device)
        got = torch.zeros(3, device=unwrapped.device)
        falls = torch.zeros((), device=unwrapped.device)
        fail_terms = [
            name
            for name in unwrapped.termination_manager.active_terms
            if name != "time_out"
        ]
        ever_failed = torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)
        fail_counts = {name: torch.zeros((), device=unwrapped.device) for name in fail_terms}
        first_fail_step = torch.full(
            (unwrapped.num_envs,), float("nan"), device=unwrapped.device
        )
        n_joints = len(joint_names)
        clip_hits = torch.zeros(n_joints, device=unwrapped.device)
        # which fence was hit matters: the ankle cranks sit off-centre, so the two
        # directions have very different headroom and an unsigned fraction hides it
        clip_lo = torch.zeros(n_joints, device=unwrapped.device)
        clip_hi = torch.zeros(n_joints, device=unwrapped.device)
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
            below = pre < clip_low
            above = pre > clip_high
            over = torch.clamp(pre - clip_high, min=0.0) + torch.clamp(clip_low - pre, min=0.0)
            margin = torch.minimum(clip_high - pre, pre - clip_low)
            if step >= args_cli.warmup_steps:
                clip_hits += (over > 0.0).float().mean(dim=0)
                clip_lo += below.float().mean(dim=0)
                clip_hi += above.float().mean(dim=0)
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
            step_failed = torch.zeros(
                unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device
            )
            for term_name in fail_terms:
                hit = unwrapped.termination_manager.get_term(term_name)
                fail_counts[term_name] += hit.float().sum()
                step_failed |= hit
            falls += unwrapped.termination_manager.get_term("fall_down").float().mean()
            fresh = step_failed & ~ever_failed
            first_fail_step = torch.where(
                fresh,
                torch.full_like(first_fail_step, float(step - args_cli.warmup_steps)),
                first_fail_step,
            )
            ever_failed |= step_failed
            counted += 1.0

        counted = torch.clamp(counted, min=1.0)
        row = {k.split("/", 1)[-1]: v for k, v in metrics.take_log().items() if k.startswith("Gait/")}
        row["err_vx"] = (err[0] / counted).item()
        row["err_vy"] = (err[1] / counted).item()
        row["err_wz"] = (err[2] / counted).item()
        row["got_vx"] = (got[0] / counted).item()
        row["got_vy"] = (got[1] / counted).item()
        row["got_wz"] = (got[2] / counted).item()
        # falls counted per env-timestep: NOT an episode failure rate
        row["falls_per_env_step"] = (falls / counted).item()
        row["falls_per_env_second"] = (falls / counted / unwrapped.step_dt).item()
        row["failed_env_fraction"] = ever_failed.float().mean().item()
        finite_first = first_fail_step[torch.isfinite(first_fail_step)]
        row["first_failure_step_mean"] = (
            finite_first.mean().item() if finite_first.numel() else float("nan")
        )
        row["termination_counts"] = {
            name: value.item() for name, value in fail_counts.items()
        }
        frac = (clip_hits / counted)
        row["clip_frac"] = {n: frac[i].item() for i, n in enumerate(joint_names)}
        row["clip_frac_max"] = frac.amax().item()
        row["clip_frac_max_joint"] = joint_names[int(frac.argmax())]
        row["worst_excess_rad"] = {n: worst_excess[i].item() for i, n in enumerate(joint_names)}
        row["min_margin_rad"] = {n: min_margin[i].item() for i, n in enumerate(joint_names)}
        row["min_margin_rad_min"] = min_margin.amin().item()
        row["clip_frac_low"] = {n: (clip_lo[i] / counted).item() for i, n in enumerate(joint_names)}
        row["clip_frac_high"] = {n: (clip_hi[i] / counted).item() for i, n in enumerate(joint_names)}
        row["clip_hit_count"] = int(clip_hits.sum().item() * unwrapped.num_envs)
        row["clip_consistent"] = bool(
            (row["clip_hit_count"] > 0) == (row["min_margin_rad_min"] < 0.0)
        )
        row["min_margin_rad_min_joint"] = joint_names[int(min_margin.argmin())]
        row["command"] = list(command)
        results[name] = row
        print(
            "CELL %-11s cmd %+.2f/%+.2f/%+.2f got %+.3f/%+.3f/%+.3f | err %.3f/%.3f/%.3f | "
            "failed_env %.3f (%.3f/s) | swing %.4f single %.3f duty %.3f td_hz %.2f | "
            "clip %.4f@%s margin %+.4f@%s"
            % (name, command[0], command[1], command[2], row["got_vx"], row["got_vy"], row["got_wz"],
               row["err_vx"], row["err_vy"], row["err_wz"],
               row["failed_env_fraction"], row["falls_per_env_second"],
               row["swing_peak_m"], row["single_stance_frac"], row["duty_l"],
               row["touchdown_hz_l"],
               row["clip_frac_max"], row["clip_frac_max_joint"].replace("_joint", ""),
               row["min_margin_rad_min"], row["min_margin_rad_min_joint"].replace("_joint", "")),
            flush=True,
        )

    inference.__exit__(None, None, None)

    if args_cli.out:
        with open(args_cli.out, "w", encoding="utf-8") as handle:
            json.dump({"_meta": meta, **results}, handle, indent=2)
        print("wrote %s" % args_cli.out)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
