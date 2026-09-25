from isaaclab.utils import configclass

from . import mdp
from .robonex_walking_env_cfg import (
    CONTACT_FORCE_LIMIT,
    FOOT_CLEARANCE,
    STANCE_WIDTH,
    STANDING_STANCE_WIDTH,
    RoboNexWalkingEnvCfg,
)
from .robot_contract_v2 import (
    ACTION_CLIPS,
    ACTION_OFFSETS,
    ACTION_SCALES,
    BASE_HEIGHT,
    CLOSED_LOOP_DEFAULT_JOINT_POS,
    CONSTANTS,
    FOOT_ORIGIN_REST_HEIGHT,
    FOOT_SOLE_CORNERS,
    STANCE_WIDTH_DEFAULT,
    robot_usd,
)

GRAVITY = 9.81
V1_MASS = 20.513910
V1_BASE_HEIGHT = 1.0710
V1_LEG_LENGTH = 0.7505
V2_LEG_LENGTH = 0.6635
V1_FALL_HEIGHT = 0.6

CONTACT_FORCE_PER_WEIGHT = CONTACT_FORCE_LIMIT / (V1_MASS * GRAVITY)
STANDING_WIDENING = STANDING_STANCE_WIDTH - STANCE_WIDTH
FOOT_CLEARANCE_V2 = round(FOOT_CLEARANCE * V2_LEG_LENGTH / V1_LEG_LENGTH, 3)
FALL_HEIGHT_V2 = round(V1_FALL_HEIGHT * BASE_HEIGHT / V1_BASE_HEIGHT, 3)

PHYSICS_HZ = 400
DECIMATION = 8
POSITION_ITERATIONS = 32


@configclass
class RoboNexWalkingV2EnvCfg(RoboNexWalkingEnvCfg):
    variant: str = "edu"
    foot_origin_rest_height: float = FOOT_ORIGIN_REST_HEIGHT
    foot_sole_corners: tuple = FOOT_SOLE_CORNERS

    def __post_init__(self) -> None:
        super().__post_init__()

        robot = self.scene.robot
        robot.spawn.usd_path = str(robot_usd(self.variant))
        robot.spawn.articulation_props.solver_position_iteration_count = POSITION_ITERATIONS
        robot.init_state.pos = (0.0, 0.0, BASE_HEIGHT)
        robot.init_state.joint_pos = CLOSED_LOOP_DEFAULT_JOINT_POS

        self.sim.dt = 1.0 / PHYSICS_HZ
        self.decimation = DECIMATION
        self.sim.render_interval = DECIMATION
        self.scene.contact_forces.history_length = DECIMATION

        action = self.actions.joint_pos
        action.offset = ACTION_OFFSETS
        action.scale = ACTION_SCALES
        action.clip = ACTION_CLIPS

        self.rewards.base_height.params["target_height"] = BASE_HEIGHT
        self.rewards.feet_width.params["target_width"] = STANCE_WIDTH_DEFAULT
        self.rewards.feet_width.params["standing_width"] = round(STANCE_WIDTH_DEFAULT + STANDING_WIDENING, 4)
        self.rewards.feet_clearance.params["target_height"] = FOOT_CLEARANCE_V2
        self.rewards.feet_clearance.params["sole_corners"] = FOOT_SOLE_CORNERS
        self.rewards.feet_contact_force.func = mdp.feet_contact_force_mean_l2
        self.rewards.feet_contact_force.params["threshold"] = round(
            CONTACT_FORCE_PER_WEIGHT * CONSTANTS["variant_mass_kg"][self.variant] * GRAVITY, 1)

        self.terminations.fall_down.params["minimum_height"] = FALL_HEIGHT_V2


@configclass
class RoboNexWalkingV2EduEnvCfg(RoboNexWalkingV2EnvCfg):
    variant: str = "edu"


@configclass
class RoboNexWalkingV2ProEnvCfg(RoboNexWalkingV2EnvCfg):
    variant: str = "pro"


@configclass
class RoboNexWalkingV2MaxEnvCfg(RoboNexWalkingV2EnvCfg):
    variant: str = "max"
