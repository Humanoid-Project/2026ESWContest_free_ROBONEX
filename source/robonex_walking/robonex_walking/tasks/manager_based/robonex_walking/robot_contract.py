from robonex_common.actuators import ACTUATOR_PARAMETERS
from robonex_common.joints import ACTUATED_JOINTS
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
LEG_JOINTS = tuple(joint.model_name for joint in ACTUATED_JOINTS)
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
    "LEG_JOINTS",
    "ROBOT_USD",
]
