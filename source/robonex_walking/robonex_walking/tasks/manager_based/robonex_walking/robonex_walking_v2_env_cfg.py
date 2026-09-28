from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
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
    FOOT_ROLL_COEFFS,
    FOOT_ROLL_LIMIT_RAD,
    FOOT_ROLL_PAIRS,
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
DEPLOY_MAX_SPEED = 6.0
DEPLOY_MAX_ACCEL = 120.0
SLEW_LAG_WEIGHT = -1.0
SLEW_LAG_SCALE = 0.0025
FOOT_ROLL_EXCESS_WEIGHT = -1.0
FOOT_ROLL_EXCESS_SCALE = 0.01
STANDING_WIDTH_WINDOW = (0.24, 0.30)
STANDING_WIDTH_SCALE = 0.0009
STANDING_HIP_ROLL_LOAD_WEIGHT = -0.2
HIP_ROLL_LOAD_REFERENCE = 20.0


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
        self.scene.illegal_contacts.history_length = DECIMATION

        action = self.actions.joint_pos
        self.actions.joint_pos = mdp.Ver2JointPositionActionCfg(
            asset_name=action.asset_name,
            joint_names=action.joint_names,
            offset=ACTION_OFFSETS,
            scale=ACTION_SCALES,
            clip=ACTION_CLIPS,
            use_default_offset=action.use_default_offset,
            max_speed=DEPLOY_MAX_SPEED,
            max_accel=DEPLOY_MAX_ACCEL,
            foot_roll_limit=FOOT_ROLL_LIMIT_RAD,
            foot_roll_coeffs=FOOT_ROLL_COEFFS,
            foot_roll_pairs=FOOT_ROLL_PAIRS,
        )

        self.rewards.slew_lag = RewTerm(func=mdp.slew_lag_l2, weight=SLEW_LAG_WEIGHT, params={"scale": SLEW_LAG_SCALE})
        self.rewards.foot_roll_clip_excess = RewTerm(
            func=mdp.foot_roll_clip_excess_l2, weight=FOOT_ROLL_EXCESS_WEIGHT, params={"scale": FOOT_ROLL_EXCESS_SCALE}
        )

        self.rewards.base_height.params["target_height"] = BASE_HEIGHT
        self.rewards.feet_width.params["target_width"] = STANCE_WIDTH_DEFAULT
        self.rewards.feet_width.params["standing_width"] = round(STANCE_WIDTH_DEFAULT + STANDING_WIDENING, 4)
        self.rewards.feet_width.params["standing_window"] = STANDING_WIDTH_WINDOW
        self.rewards.feet_width.params["standing_scale"] = STANDING_WIDTH_SCALE
        self.rewards.stand_hip_roll_load = RewTerm(
            func=mdp.standing_joint_load_l1,
            weight=STANDING_HIP_ROLL_LOAD_WEIGHT,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"]),
                "reference_torque": HIP_ROLL_LOAD_REFERENCE,
            },
        )
        self.rewards.feet_clearance.params["target_height"] = FOOT_CLEARANCE_V2
        self.rewards.feet_clearance.params["sole_corners"] = FOOT_SOLE_CORNERS
        self.rewards.foot_slip.func = mdp.foot_slip_latest_l2
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
