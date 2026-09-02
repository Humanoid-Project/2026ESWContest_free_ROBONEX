import os
from pathlib import Path

from robonex_common.joints import ACTUATED_JOINTS
from robonex_common.limits import action_normalization
from robonex_common.motors import MOTOR_CONTROL_KD, MOTOR_CONTROL_KP, MOTOR_PHYSICS

DESCRIPTION_REPO_NAMES = ("robonex-description", "robonex_description")


def resolve_description_root():
    configured = os.environ.get("ROBONEX_DESCRIPTION_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"ROBONEX_DESCRIPTION_ROOT does not exist: {root}")
        return root
    anchors = (Path.cwd().resolve(), Path(__file__).resolve())
    for anchor in anchors:
        for parent in (anchor, *anchor.parents):
            for name in DESCRIPTION_REPO_NAMES:
                candidate = parent / name
                if candidate.is_dir():
                    return candidate
    raise FileNotFoundError(
        "robonex-description checkout not found; clone it beside robonex-walking or set ROBONEX_DESCRIPTION_ROOT"
    )


DESCRIPTION_ROOT = resolve_description_root()
ROBOT_USD = DESCRIPTION_ROOT / "isaac" / "closed_loop_mesh" / "robonex_closed_loop_mesh.usd"
if not ROBOT_USD.is_file():
    raise FileNotFoundError(f"Isaac USD not found: {ROBOT_USD}")

LEG_JOINTS = tuple(joint.model_name for joint in ACTUATED_JOINTS)
ACTION_OFFSETS, ACTION_SCALES, ACTION_CLIPS = action_normalization(0.01)
ACTUATOR_PARAMETERS = {
    model: {
        "stiffness": MOTOR_CONTROL_KP,
        "damping": MOTOR_CONTROL_KD,
        "armature": values["armature"],
        "friction": values["frictionloss"],
    }
    for model, values in MOTOR_PHYSICS.items()
}
