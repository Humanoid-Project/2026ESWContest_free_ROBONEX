import json
import math

from robonex_common.actuators import ACTUATOR_PARAMETERS
from robonex_common.joints import ACTUATED_JOINTS
from robonex_common.limits import DEFAULT_ACTION_MARGIN_RAD, RUNNER_ACTION_CLIP, MAX_ACTION_SCALE_RAD, ACTION_REACH_SIGMA
from robonex_common.motors import RATED_TORQUE
from robonex_common.paths import DESCRIPTION_REPO_NAMES, repo_file

VARIANTS = ("edu", "pro", "max")
CONSTANTS_PATH = repo_file(
    DESCRIPTION_REPO_NAMES,
    "ver2/ver2_constants.json",
    env_var="ROBONEX_DESCRIPTION_ROOT",
    anchors=(__file__,),
)
CONSTANTS = json.loads(CONSTANTS_PATH.read_text())


def robot_usd(variant):
    return repo_file(
        DESCRIPTION_REPO_NAMES,
        "ver2/isaac/%s/closed_loop_mesh/robonex_%s_closed_loop_mesh.usd" % (variant, variant),
        env_var="ROBONEX_DESCRIPTION_ROOT",
        anchors=(__file__,),
    )


BASE_HEIGHT = CONSTANTS["home_base_height"]
FOOT_ORIGIN_REST_HEIGHT = CONSTANTS["foot_origin_rest_height"]
FOOT_SOLE_CORNERS = tuple(tuple(c) for c in CONSTANTS["foot_sole_corners"])
STANCE_WIDTH_DEFAULT = CONSTANTS["stance_width_default"]
HIP_PITCH_HEIGHT = CONSTANTS["hip_pitch_height_default"]
JOINT_LIMITS = {name: tuple(v) for name, v in CONSTANTS["provisional_limits"].items()}
DEFAULT_ACTUATED_POS = dict(CONSTANTS["default_actuated_pos"])

LEG_JOINTS = tuple(joint.model_name for joint in ACTUATED_JOINTS)
_RATED_SPINNING = {"rs02": 7.0, "rs03": 20.0}
RATED_TORQUE_STANDSTILL = {joint.model_name: RATED_TORQUE[joint.motor_model] for joint in ACTUATED_JOINTS}
RATED_TORQUE_SPINNING = {joint.model_name: _RATED_SPINNING[joint.motor_model] for joint in ACTUATED_JOINTS}


def _isaac_passive_name(name):
    for ax, idx in (("_x", ":0"), ("_y", ":1"), ("_z", ":2")):
        if name.startswith(("l_ankle_coupler_joint_", "r_ankle_coupler_joint_")) and name.endswith(ax):
            return name[: -len(ax)] + idx
    return name


CLOSED_LOOP_DEFAULT_JOINT_POS = {**DEFAULT_ACTUATED_POS, **{
    _isaac_passive_name(n): v for n, v in CONSTANTS["home_passive_pos"].items()}}


def _action_scale(name, margin=DEFAULT_ACTION_MARGIN_RAD):
    lower, upper = JOINT_LIMITS[name]
    default = DEFAULT_ACTUATED_POS[name]
    near = min(default - (lower + margin), (upper - margin) - default)
    far = max(default - (lower + margin), (upper - margin) - default)
    scale = max(near / ACTION_REACH_SIGMA, far / RUNNER_ACTION_CLIP)
    return min(MAX_ACTION_SCALE_RAD, math.ceil(scale * 1e6) / 1e6)


def action_normalization(margin=DEFAULT_ACTION_MARGIN_RAD):
    offsets, scales, clips = {}, {}, {}
    for name in LEG_JOINTS:
        lower, upper = JOINT_LIMITS[name]
        clip_lower, clip_upper = lower + margin, upper - margin
        default = DEFAULT_ACTUATED_POS[name]
        if not clip_lower <= default <= clip_upper:
            raise ValueError(f"default pose for {name} is outside its clipped range: {default}")
        scale = _action_scale(name, margin)
        if scale * RUNNER_ACTION_CLIP < max(default - clip_lower, clip_upper - default):
            raise ValueError(f"action scale for {name} cannot reach its target clip: {scale}")
        offsets[name], scales[name], clips[name] = default, scale, (clip_lower, clip_upper)
    return offsets, scales, clips


ACTION_OFFSETS, ACTION_SCALES, ACTION_CLIPS = action_normalization(0.01)

__all__ = [
    "ACTION_CLIPS",
    "ACTION_OFFSETS",
    "ACTION_SCALES",
    "ACTUATOR_PARAMETERS",
    "BASE_HEIGHT",
    "CLOSED_LOOP_DEFAULT_JOINT_POS",
    "CONSTANTS",
    "FOOT_ORIGIN_REST_HEIGHT",
    "FOOT_SOLE_CORNERS",
    "HIP_PITCH_HEIGHT",
    "JOINT_LIMITS",
    "LEG_JOINTS",
    "RATED_TORQUE_SPINNING",
    "RATED_TORQUE_STANDSTILL",
    "STANCE_WIDTH_DEFAULT",
    "VARIANTS",
    "robot_usd",
]
