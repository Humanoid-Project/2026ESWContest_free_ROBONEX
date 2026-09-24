import pathlib

import torch
import yaml

DEPLOY_MAX_SPEED = 6.0
DEPLOY_MAX_ACCEL = 120.0


class _SavedCfgLoader(yaml.SafeLoader):
    pass


_SavedCfgLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)
_SavedCfgLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: slice(*loader.construct_sequence(node)),
)


def apply_training_env_cfg(env_cfg, checkpoint):
    from isaaclab.utils.dict import update_class_from_dict

    path = pathlib.Path(checkpoint).resolve().parent / "params" / "env.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; the checkpoint's training config cannot be restored")
    with path.open() as handle:
        saved = yaml.load(handle, Loader=_SavedCfgLoader)
    for key in ("viewer", "log_dir", "seed"):
        saved.pop(key, None)
    update_class_from_dict(env_cfg, saved)
    return path


def disable_randomization(env_cfg):
    kept = []
    for name in list(vars(env_cfg.events)):
        if name.startswith("_"):
            continue
        if name.startswith("reset_"):
            kept.append(name)
        else:
            setattr(env_cfg.events, name, None)
    return kept


def limit_step(position, velocity, target, dt, max_speed, max_accel):
    error = target - position
    braking = torch.clamp(
        torch.sqrt((max_accel * dt) ** 2 + 2.0 * max_accel * error.abs()) - max_accel * dt, min=0.0
    )
    desired = torch.sign(error) * torch.clamp(braking, max=max_speed)
    change = torch.clamp(desired - velocity, -max_accel * dt, max_accel * dt)
    next_velocity = torch.clamp(velocity + change, -max_speed, max_speed)
    next_position = position + next_velocity * dt
    arrived = error * (target - next_position) <= 0.0
    next_position = torch.where(arrived, target, next_position)
    next_velocity = torch.where(arrived, torch.zeros_like(next_velocity), next_velocity)
    return next_position, next_velocity


def install_slew_limiter(unwrapped, max_speed=DEPLOY_MAX_SPEED, max_accel=DEPLOY_MAX_ACCEL):
    term = unwrapped.action_manager.get_term("joint_pos")
    asset = unwrapped.scene["robot"]
    dt = float(unwrapped.step_dt)
    state = {
        "position": asset.data.default_joint_pos[:, term._joint_ids].clone(),
        "velocity": torch.zeros_like(asset.data.default_joint_pos[:, term._joint_ids]),
    }
    process_actions = term.process_actions
    reset = term.reset

    def limited_process_actions(actions):
        process_actions(actions)
        position, velocity = limit_step(
            state["position"], state["velocity"], term._processed_actions, dt, max_speed, max_accel
        )
        state["position"], state["velocity"] = position, velocity
        term._processed_actions = position.clone()

    def limited_reset(env_ids=None):
        reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        state["position"][ids] = asset.data.default_joint_pos[ids][:, term._joint_ids]
        state["velocity"][ids] = 0.0

    term.process_actions = limited_process_actions
    term.reset = limited_reset
    return state


def install_joint_obs_delay(unwrapped, steps=1, group="policy", names=("joint_pos_rel", "joint_vel_rel")):
    manager = unwrapped.observation_manager
    term_names = manager._group_obs_term_names[group]
    wrapped = []
    for name, term_cfg in zip(term_names, manager._group_obs_term_cfgs[group]):
        if name not in names:
            continue
        term_cfg.func = _delayed(term_cfg.func, steps)
        wrapped.append(name)
    missing = set(names) - set(wrapped)
    if missing:
        raise KeyError(f"observation terms not found in group {group!r}: {sorted(missing)}")
    return wrapped


def _delayed(func, steps):
    buffer = {"step": None, "frames": [], "last": None}

    def delayed(env, **kwargs):
        if kwargs.get("inspect"):
            return func(env, **kwargs)
        if buffer["step"] == env.common_step_counter and buffer["last"] is not None:
            return buffer["last"]
        current = func(env, **kwargs).clone()
        frames = buffer["frames"]
        frames.append(current)
        del frames[:-(steps + 1)]
        oldest = frames[0]
        fresh = (env.episode_length_buf == 0).unsqueeze(-1)
        for frame in frames:
            frame[fresh.expand_as(frame)] = current[fresh.expand_as(current)]
        output = oldest.clone()
        buffer["step"], buffer["last"] = env.common_step_counter, output
        return output

    return delayed
