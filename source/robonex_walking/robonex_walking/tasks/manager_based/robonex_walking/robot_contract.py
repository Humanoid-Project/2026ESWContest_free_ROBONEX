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

LEG_JOINTS = tuple(joint.model_name for joint in ACTUATED_JOINTS)
ACTION_OFFSETS, ACTION_SCALES, ACTION_CLIPS = action_normalization(0.01)

__all__ = [
    "ACTION_CLIPS",
    "ACTION_OFFSETS",
    "ACTION_SCALES",
    "ACTUATOR_PARAMETERS",
    "DESCRIPTION_ROOT",
    "LEG_JOINTS",
    "ROBOT_USD",
]
