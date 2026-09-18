from robonex_common.actuators import ACTUATOR_PARAMETERS
from robonex_common.joints import ACTUATED_JOINTS
from robonex_common.motors import RATED_TORQUE
from robonex_common.limits import action_normalization
from robonex_common.paths import DESCRIPTION_REPO_NAMES, repo_file

ROBOT_USD = repo_file(
    DESCRIPTION_REPO_NAMES,
    "isaac/closed_loop_mesh/robonex_closed_loop_mesh.usd",
    env_var="ROBONEX_DESCRIPTION_ROOT",
    anchors=(__file__,),
)
DESCRIPTION_ROOT = ROBOT_USD.parents[2]

BASE_HEIGHT = 1.0710
FOOT_ORIGIN_REST_HEIGHT = 0.06545
# Sole corners in the foot body frame, from the collision mesh in
# robonex-description/mujoco/robot/scene.xml (13716 verts, sole plane at z=-0.06540).
# The body origin sits 0.151 m behind the toe and 0.062 m ahead of the heel, so a
# toe-down pitch lifts the origin while the toe stays low: at 10 deg the origin reads
# 60 mm of clearance while the sole is 36 mm off the ground. Checked against the full
# mesh over pitch -16..20 deg and roll -10..10 deg, these 4 points are never optimistic
# and are at most 4.8 mm conservative.
FOOT_SOLE_CORNERS = (
    (-0.0623, -0.0445, -0.0654),
    (-0.0623, 0.0845, -0.0654),
    (0.1505, -0.0445, -0.0654),
    (0.1505, 0.0845, -0.0654),
)
_RATED_SPINNING = {"rs02": 7.0, "rs03": 20.0}

LEG_JOINTS = tuple(joint.model_name for joint in ACTUATED_JOINTS)

RATED_TORQUE_STANDSTILL = {joint.model_name: RATED_TORQUE[joint.motor_model] for joint in ACTUATED_JOINTS}
RATED_TORQUE_SPINNING = {joint.model_name: _RATED_SPINNING[joint.motor_model] for joint in ACTUATED_JOINTS}
ACTION_OFFSETS, ACTION_SCALES, ACTION_CLIPS = action_normalization(0.01)
CLOSED_LOOP_DEFAULT_JOINT_POS = {
    "l_hip_yaw_joint": 0.0,
    "r_hip_yaw_joint": 0.0,
    "l_hip_pitch_joint": 0.1,
    "r_hip_pitch_joint": -0.1,
    "l_hip_roll_joint": 0.0,
    "r_hip_roll_joint": 0.0,
    "l_knee_joint": -0.29669690697178175,
    "l_knee_pitch_joint": -0.38578,
    "r_knee_joint": 0.2966969070362342,
    "r_knee_pitch_joint": 0.38578,
    "l_ankle_lower_joint": -0.2056595,
    "l_ankle_roll_joint": -0.00039827559348155157,
    "l_ankle_upper_joint": 0.2056595,
    "l_knee_coupler_joint_a": 0.22857922748055304,
    "r_ankle_lower_joint": 0.2056595,
    "r_ankle_roll_joint": 0.00039827559356127827,
    "r_ankle_upper_joint": -0.2056595,
    "r_knee_coupler_joint_a": 0.22857922749663748,
    "l_ankle_coupler_joint_a:0": -0.0001806448180792951,
    "l_ankle_coupler_joint_a:1": 0.17045883823718327,
    "l_ankle_coupler_joint_a:2": 0.0,
    "l_ankle_pitch_joint": 0.19686707888571855,
    "l_ankle_coupler_joint_b:0": -0.00010323723148808417,
    "l_ankle_coupler_joint_b:1": 0.18560610755019352,
    "l_ankle_coupler_joint_b:2": 0.0,
    "r_ankle_coupler_joint_a:0": 0.0001806448180893788,
    "r_ankle_coupler_joint_a:1": 0.1704588382352967,
    "r_ankle_coupler_joint_a:2": 0.0,
    "r_ankle_pitch_joint": 0.1968670788838191,
    "r_ankle_coupler_joint_b:0": 0.00010323723147981379,
    "r_ankle_coupler_joint_b:1": 0.18560610754814208,
    "r_ankle_coupler_joint_b:2": 0.0,
}

__all__ = [
    "ACTION_CLIPS",
    "ACTION_OFFSETS",
    "ACTION_SCALES",
    "ACTUATOR_PARAMETERS",
    "BASE_HEIGHT",
    "CLOSED_LOOP_DEFAULT_JOINT_POS",
    "DESCRIPTION_ROOT",
    "FOOT_ORIGIN_REST_HEIGHT",
    "FOOT_SOLE_CORNERS",
    "LEG_JOINTS",
    "RATED_TORQUE_STANDSTILL",
    "RATED_TORQUE_SPINNING",
    "ROBOT_USD",
]
