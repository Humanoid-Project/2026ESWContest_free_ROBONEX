# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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

from . import mdp
from .robot_contract import (
    ACTION_CLIPS,
    ACTION_OFFSETS,
    ACTION_SCALES,
    ACTUATOR_PARAMETERS,
    LEG_JOINTS,
    ROBOT_USD,
)

BASE_HEIGHT = 1.0789
STANCE_WIDTH = 0.321
TARGET_SPEED_X = 0.3
STEP_AIR_TIME = 0.35
FOOT_CLEARANCE = 0.06


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
        # Initial State (m, rad)
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, BASE_HEIGHT),
            joint_pos={
                "l_hip_yaw_joint": 0.0,
                "l_hip_pitch_joint": 0.0,
                "l_hip_roll_joint": 0.0,

                "l_knee_pitch_joint": 0.0,

                "l_ankle_upper_joint": 0.0,
                "l_ankle_lower_joint": 0.0,

                "r_hip_yaw_joint": 0.0,
                "r_hip_pitch_joint": 0.0,
                "r_hip_roll_joint": 0.0,

                "r_knee_pitch_joint": 0.0,

                "r_ankle_upper_joint": 0.0,
                "r_ankle_lower_joint": 0.0,
            },
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
        """Observations for policy group. (42)"""

        # Joint Position (12) (rad)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
            noise=GaussianNoiseCfg(mean=0.0, std=0.0002),
        )
        # Joint Velocity (12) (rad/s)
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
            noise=GaussianNoiseCfg(mean=0.0, std=0.1),
        )

        # IMU angular velocity (3) (rad/s)
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.005),
                bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05),
            ),
        )
        # Projected gravity (3)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
        )

        # Last Action (12)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Randomization Kp/Kd
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

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
        interval_range_s=(5.0, 8.0),
        params={
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
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

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)

    # Task
    track_lin_vel_x = RewTerm(
        func=mdp.track_lin_vel_x_exp,
        weight=3.0,
        params={"target_speed": TARGET_SPEED_X, "std": 0.25},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_biped,
        weight=2.0,
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
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2_bounded, weight=-5.0)
    base_height = RewTerm(
        func=mdp.base_height_l2_bounded,
        weight=-10.0,
        params={"target_height": BASE_HEIGHT},
    )
    lin_vel_y = RewTerm(func=mdp.lin_vel_y_l2_bounded, weight=-2.0)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2_bounded, weight=-2.0)
    ang_vel_z = RewTerm(func=mdp.ang_vel_z_l2_bounded, weight=-0.5)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2_bounded, weight=-0.05)

    # Foot placement
    feet_width = RewTerm(
        func=mdp.feet_stance_width_l2,
        weight=-10.0,
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
        weight=-20.0,
        params={
            "target_height": FOOT_CLEARANCE,
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
        },
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip_l2,
        weight=-2.0,
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
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["l_foot", "r_foot"],
                preserve_order=True,
            ),
        },
    )

    # Regularization
    action_rate = RewTerm(func=mdp.action_rate_l2_bounded, weight=-0.015)
    joint_deviation_yaw_roll = RewTerm(
        func=mdp.joint_deviation_l1_bounded,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"],
            )
        },
    )
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2_bounded,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2_bounded,
        weight=-2.5e-7,
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


@configclass
class RoboNexWalkingEnvCfg(ManagerBasedRLEnvCfg):
    scene: RoboNexWalkingSceneCfg = RoboNexWalkingSceneCfg(num_envs=512, env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

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
