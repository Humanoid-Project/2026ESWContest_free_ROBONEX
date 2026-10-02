import copy
import math
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
    _add_trained_params(env_cfg, saved)
    update_class_from_dict(env_cfg, saved)
    remap_moved_usd(env_cfg)
    for manager in ("events", "rewards", "terminations", "curriculum"):
        section = getattr(env_cfg, manager, None)
        trained = saved.get(manager) or {}
        if section is None:
            continue
        for name in list(vars(section)):
            if not name.startswith("_") and name not in trained:
                setattr(section, name, None)
            else:
                _drop_untrained_params(getattr(section, name, None), trained.get(name))
    for group_name, group in vars(getattr(env_cfg, "observations", object())).items():
        trained_group = (saved.get("observations") or {}).get(group_name) or {}
        if group_name.startswith("_") or not isinstance(trained_group, dict):
            continue
        for name, term in vars(group).items():
            if not name.startswith("_"):
                _drop_untrained_params(term, trained_group.get(name))
    return path


def _drop_untrained_params(term, trained):
    params = getattr(term, "params", None)
    if not isinstance(params, dict) or not isinstance(trained, dict) or not isinstance(trained.get("params"), dict):
        return
    for key in [k for k in params if k not in trained["params"]]:
        del params[key]


def _add_trained_params(env_cfg, saved):
    for manager in ("events", "rewards", "terminations", "curriculum"):
        section = getattr(env_cfg, manager, None)
        for name, trained in (saved.get(manager) or {}).items():
            params = getattr(getattr(section, name, None), "params", None)
            if not isinstance(params, dict) or not isinstance(trained, dict) or not isinstance(trained.get("params"), dict):
                continue
            for key, value in trained["params"].items():
                if key not in params:
                    params[key] = _trained_value(value)


def _trained_value(value):
    if isinstance(value, dict) and "name" in value and ("joint_names" in value or "body_names" in value):
        from isaaclab.managers import SceneEntityCfg

        return SceneEntityCfg(name=value["name"])
    if isinstance(value, dict):
        return {k: _trained_value(v) for k, v in value.items()}
    return value


def pin_joint_friction(env_cfg, levels, func):
    from robonex_common.joints import ACTUATED_JOINTS

    term = getattr(env_cfg.events, "randomize_joint_friction", None)
    if term is None:
        raise KeyError("the env has no randomize_joint_friction event to pin")
    if set(levels) != {joint.motor_model for joint in ACTUATED_JOINTS} or min(levels.values()) < 0.0:
        raise ValueError(f"joint friction needs one non-negative N*m level per motor model: {levels}")
    term.func = func
    term.params = {
        "asset_cfg": term.params["asset_cfg"],
        "friction_range": {joint.model_name: (levels[joint.motor_model],) * 2 for joint in ACTUATED_JOINTS},
        "static_ratio": 1.0,
        "viscous": 0.0,
    }
    return {model: float(level) for model, level in sorted(levels.items())}


BASE_COM_EVENT = "randomize_base_com"
BASE_COM_AXES = ("x", "y", "z")


def apply_base_com_pin(env_cfg, offset=None, template=None, keep_training=False):
    if keep_training and offset is not None:
        raise ValueError("--com_pin and --com_keep_training exclude each other")
    term = getattr(env_cfg.events, BASE_COM_EVENT, None)
    pin = None
    if keep_training:
        source = "training"
    else:
        pin = [float(v) for v in (offset if offset is not None else (0.0, 0.0, 0.0))]
        if len(pin) != 3 or not all(math.isfinite(v) for v in pin):
            raise ValueError(f"the base CoM pin needs three finite offsets in metres: {offset}")
        source = "pin" if offset is not None else "default_nominal"
        if term is None and any(pin):
            if template is None:
                raise KeyError(f"the env has no {BASE_COM_EVENT} event to pin a non-zero offset with")
            term = copy.deepcopy(template)
            setattr(env_cfg.events, BASE_COM_EVENT, term)
        if term is not None:
            term.params["com_range"] = {axis: (value, value) for axis, value in zip(BASE_COM_AXES, pin)}
    com_range = None
    if term is not None:
        saved = term.params.get("com_range") or {}
        com_range = {axis: [float(v) for v in saved.get(axis, (0.0, 0.0))] for axis in BASE_COM_AXES}
    return {
        "source": source,
        "pin_m": pin,
        "event_present": term is not None,
        "range_m": com_range,
        "nominal": com_range is None or all(v == 0.0 for bounds in com_range.values() for v in bounds),
    }


MOVED_DESCRIPTION_PATHS = (
    ("/robonex-description/new_urdf/", "/robonex-description/ver2/",
     "Ver.2 USD regenerated 2026-09-27 with the decided hip limits; runs before W73 trained on the older limits"),
)


def remap_moved_usd(env_cfg):
    spawn = env_cfg.scene.robot.spawn
    path = getattr(spawn, "usd_path", None)
    if not path or pathlib.Path(path).is_file():
        return None
    for old, new, note in MOVED_DESCRIPTION_PATHS:
        if old in path and pathlib.Path(path.replace(old, new)).is_file():
            spawn.usd_path = path.replace(old, new)
            print(f"[deploy_effects] saved usd_path {path} no longer exists; using {spawn.usd_path} ({note})")
            return spawn.usd_path
    raise FileNotFoundError(f"saved usd_path {path} does not exist and has no known new location")


def disable_randomization(env_cfg, keep=()):
    kept = []
    for name in list(vars(env_cfg.events)):
        if name.startswith("_"):
            continue
        if name.startswith("reset_") or (name in keep and getattr(env_cfg.events, name) is not None):
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
    if hasattr(term, "_slew_position") and getattr(term.cfg, "slew_enabled", True):
        raise ValueError("the action term already applies the deploy slew limiter; do not add a second one")
    if hasattr(term, "_slew_position"):
        if term.cfg.max_speed != max_speed or term.cfg.max_accel != max_accel:
            raise ValueError("the action term's slew stage has different limits from the deploy limiter")
        term.cfg.slew_enabled = True
        term.reset()
        return {"position": term._slew_position, "velocity": term._slew_velocity}
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
        if hasattr(term_cfg.func, "_max_delay"):
            raise ValueError(f"{name} is already delayed by the training observation term; do not add a second delay")
        term_cfg.func = _delayed(term_cfg.func, steps)
        wrapped.append(name)
    missing = set(names) - set(wrapped)
    if missing:
        raise KeyError(f"observation terms not found in group {group!r}: {sorted(missing)}")
    return wrapped


def pin_joint_obs_delay(unwrapped, steps, group="policy"):
    manager = unwrapped.observation_manager
    pinned = []
    for name, term_cfg in zip(manager._group_obs_term_names[group], manager._group_obs_term_cfgs[group]):
        term = term_cfg.func
        if not hasattr(term, "_max_delay"):
            continue
        if not 0 <= steps <= term._max_delay:
            raise ValueError(f"{name} keeps {term._max_delay} delayed frame(s); cannot pin a delay of {steps}")
        term._delay.fill_(steps)
        term.reset = lambda env_ids=None, term=term: term._delay.fill_(steps)
        pinned.append(name)
    if not pinned:
        raise KeyError(f"no observation term in group {group!r} is delayed by training; use install_joint_obs_delay")
    return pinned


def observation_delay_state(unwrapped, env_index, group="policy", sensor="imu"):
    manager = unwrapped.observation_manager
    joint_max = None
    for term_cfg in manager._group_obs_term_cfgs[group]:
        if hasattr(term_cfg.func, "_max_delay"):
            joint_max = int(term_cfg.func._max_delay)
            break
    joint = getattr(unwrapped, "_joint_observation_delay", None)
    imu = (getattr(unwrapped.scene, "sensors", None) or {}).get(sensor)
    imu_delay = getattr(imu, "_delay", None)
    imu_cfg = getattr(imu, "cfg", None)
    imu_range = None
    if imu_delay is not None and hasattr(imu_cfg, "min_delay_steps"):
        imu_range = [int(imu_cfg.min_delay_steps), int(imu_cfg.max_delay_steps)]
    return {
        "joint_obs_delay": None if joint is None or joint_max is None else int(joint[env_index]),
        "joint_obs_max_delay": joint_max,
        "imu_delay_physics_steps": None if imu_delay is None else int(imu_delay[env_index]),
        "imu_delay_range_physics_steps": imu_range,
    }


def summarize_delay_states(states):
    summary = {}
    for key in ("joint_obs_delay", "imu_delay_physics_steps"):
        values = [state[key] for state in states]
        if not values or all(v is None for v in values):
            summary[key] = None
            continue
        summary[key] = {
            "first": values[0],
            "last": values[-1],
            "values": sorted(set(values)),
            "changes": sum(1 for a, b in zip(values, values[1:]) if a != b),
        }
    for key in ("joint_obs_max_delay", "imu_delay_range_physics_steps"):
        summary[key] = states[0][key] if states else None
    return summary


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
