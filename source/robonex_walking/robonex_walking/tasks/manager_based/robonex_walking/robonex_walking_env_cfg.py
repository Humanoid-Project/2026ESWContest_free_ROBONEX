# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from robonex_common.runtime import OBSERVATION_HISTORY_LENGTH

from . import mdp
from .robot_contract import (
    ACTION_CLIPS,
    ACTION_OFFSETS,
    ACTION_SCALES,
    ACTUATOR_PARAMETERS,
    BASE_HEIGHT,
    CLOSED_LOOP_DEFAULT_JOINT_POS,
    FOOT_ORIGIN_REST_HEIGHT,
    LEG_JOINTS,
    ROBOT_USD,
)

STANCE_WIDTH = 0.321
TARGET_SPEED_X = 0.3
TARGET_SPEED_STD = 0.15
STEP_AIR_TIME = 0.35
FOOT_CLEARANCE = 0.06
CONTACT_FORCE_LIMIT = 300.0
GAIT_PERIOD_S = 0.8
GAIT_STANCE_FRACTION = 0.55


@configclass
class RoboNexWalkingSceneCfg(InteractiveSceneCfg):
    """Configuration for the RoboNex Walking training scene."""

    # Ground
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.6,
            ),
        ),
    )

    # Light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=500.0,
        ),
    )

    # Robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROBOT_USD),
            activate_contact_sensors=True,
            articulation_props=sim_utils.schemas.ArticulationRootPropertiesCfg(
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=4,
            ),
        ),
        soft_joint_pos_limit_factor=0.9,
        # Initial State (m, rad)
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, BASE_HEIGHT),
            joint_pos=CLOSED_LOOP_DEFAULT_JOINT_POS,
        ),
        # Actuators
        actuators={
            "rs02": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_ankle_upper_joint",
                    ".*_ankle_lower_joint",
                ],
                **ACTUATOR_PARAMETERS["rs02"],
            ),
            "rs03": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_pitch_joint",
                    ".*_hip_roll_joint",
                    ".*_knee_pitch_joint",
                ],
                **ACTUATOR_PARAMETERS["rs03"],
            ),
        },
    )

    # IMU
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=ImuCfg.OffsetCfg(
            pos=(0.060, 0.0, 0.035),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )

    # Contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        history_length=3,
        track_air_time=True,
        debug_vis=False,
    )

    illegal_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(base_link|.*_knee_link|.*_ankle_link|.*_hip_.*_link)",
        history_length=1,
        track_air_time=False,
        debug_vis=False,
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.1, 0.1), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, TARGET_SPEED_X), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        offset=ACTION_OFFSETS,
        scale=ACTION_SCALES,
        clip=ACTION_CLIPS,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group. (47 x history)"""

        # Joint Position (12) (rad)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
        )
        # Joint Velocity (12) (rad/s)
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
            noise=GaussianNoiseCfg(mean=0.0, std=0.75),
        )

        # IMU angular velocity (3) (rad/s)
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.1),
                bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05),
            ),
        )
        # Projected gravity (3)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoiseCfg(mean=0.0, std=0.025),
        )

        # Velocity command (3) (m/s, m/s, rad/s)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

        # Gait clock (2) (sin, cos)
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": GAIT_PERIOD_S})

        # Last Action (12)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = OBSERVATION_HISTORY_LENGTH

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = OBSERVATION_HISTORY_LENGTH

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""


    # Initialization base_link pose/velocity
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "yaw": (-3.14159, 3.14159),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    # Initialization leg joint positions
    reset_leg_joints = EventTerm(
        func=mdp.reset_closed_loop_to_default,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
        },
    )

    # Push Robot
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
            },
        },
    )

    # Randomization friction
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.4, 0.8),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Randomization joint friction
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "friction_distribution_params": (0.0, 0.02),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # Randomization mass
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (-0.3, 0.3),
            "operation": "add",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)

    # Task
    track_lin_vel_x = RewTerm(
        func=mdp.track_lin_vel_x_exp,
        weight=3.0,
        params={"std": TARGET_SPEED_STD},
    )
    feet_gait = RewTerm(
        func=mdp.feet_gait,
        weight=1.0,
        params={
            "period": GAIT_PERIOD_S,
            "offset": [0.0, 0.5],
            "threshold": GAIT_STANCE_FRACTION,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["l_foot", "r_foot"], preserve_order=True
            ),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_biped,
        weight=1.0,
        params={
            "threshold": STEP_AIR_TIME,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["l_foot", "r_foot"],
                preserve_order=True,
            ),
        },
    )

    # Posture
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2_bounded, weight=-0.5)
    base_height = RewTerm(
        func=mdp.base_height_l2_bounded,
        weight=-0.3,
        params={"target_height": BASE_HEIGHT},
    )
    lin_vel_y = RewTerm(func=mdp.lin_vel_y_l2_bounded, weight=-0.3)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2_bounded, weight=-0.1)
    ang_vel_z = RewTerm(func=mdp.ang_vel_z_l2_bounded, weight=-0.2)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2_bounded, weight=-0.1)

    # Foot placement
    feet_width = RewTerm(
        func=mdp.feet_stance_width_l2,
        weight=-0.1,
        params={
            "target_width": STANCE_WIDTH,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["l_foot", "r_foot"],
                preserve_order=True,
            ),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.feet_clearance_l2,
        weight=-0.5,
        params={
            "target_height": FOOT_ORIGIN_REST_HEIGHT + FOOT_CLEARANCE,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["l_foot", "r_foot"], preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["l_foot", "r_foot"], preserve_order=True
            ),
        },
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip_l2,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["l_foot", "r_foot"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["l_foot", "r_foot"],
                preserve_order=True,
            ),
            "threshold": 1.0,
        },
    )
    flight_phase = RewTerm(
        func=mdp.both_feet_off_ground,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["l_foot", "r_foot"],
                preserve_order=True,
            ),
        },
    )

    feet_contact_force = RewTerm(
        func=mdp.feet_contact_force_l2,
        weight=-0.1,
        params={
            "threshold": CONTACT_FORCE_LIMIT,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["l_foot", "r_foot"],
                preserve_order=True,
            ),
        },
    )

    # Regularization
    action_rate = RewTerm(func=mdp.action_rate_l2_bounded, weight=-0.2)
    joint_deviation_yaw_roll = RewTerm(
        func=mdp.joint_deviation_l1_bounded,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"],
            )
        },
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    )
    target_clip_excess = RewTerm(
        func=mdp.target_clip_excess_l2,
        weight=-0.3,
        params={"max_excess": 2.0},
    )
    energy = RewTerm(
        func=mdp.energy_l2_bounded,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2_bounded,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2_bounded,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall_down = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.6},
    )
    bad_joint_vel = DoneTerm(func=mdp.unstable_joint_vel, params={"limit": 100.0})
    invalid_state = DoneTerm(func=mdp.invalid_state)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("illegal_contacts"),
            "threshold": 1.0,
        },
    )
    invalid_contact = DoneTerm(
        func=mdp.nonfinite_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    lin_vel_cmd = CurrTerm(func=mdp.lin_vel_cmd_levels)


@configclass
class RoboNexWalkingEnvCfg(ManagerBasedRLEnvCfg):
    scene: RoboNexWalkingSceneCfg = RoboNexWalkingSceneCfg(num_envs=512, env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""

        self.sim.dt = 1.0 / 250
        self.decimation = 5
        self.episode_length_s = 20

        self.viewer.eye = (8.0, 0.0, 5.0)

        self.sim.render_interval = self.decimation

        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 1
        self.sim.physx.max_position_iteration_count = 255
        self.sim.physx.min_velocity_iteration_count = 0
        self.sim.physx.max_velocity_iteration_count = 255
