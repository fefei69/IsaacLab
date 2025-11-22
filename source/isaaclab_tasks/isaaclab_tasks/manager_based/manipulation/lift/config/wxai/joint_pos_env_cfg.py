# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
# from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

WXAI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path="/workspace/external_assets/wxai_base.usd"),
    actuators={"joint_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], 
                                                 effort_limit_sim=100.0, 
                                                 velocity_limit_sim=100.0,
                                                 stiffness=10000.0,
                                                 damping=100.0, 
                                                 )},
)

WXAI_TEST_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/external_assets/wxai_base.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
    ), 
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_0": 0.0,
            "joint_1": 1.05,
            "joint_2": 0.87,
            "joint_3": -1.010,
            "joint_4": 0.0,
            "joint_5": 0.0, #-1.7
            "left_carriage_joint": 0.044,
            "right_carriage_joint": 0.044,
        },
    ),
    actuators={
        # Joints 1-3: Base joints with higher torque (27 Nm)
        "wxai_base": ImplicitActuatorCfg(
            joint_names_expr=["joint_[0-2]"],
            effort_limit_sim=27.0,  # Match actual robot limits
            stiffness=120.0,  # Based on motor_parameters position.kp
            damping=8.0,      # Based on motor_parameters velocity.kp
        ),
        # Joints 4-6: Wrist joints with lower torque (7 Nm)
        "wxai_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint_[3-5]"],
            effort_limit_sim=7.0,   # Match actual robot limits
            stiffness=50.0,   # Average of kp values (80, 40, 40)
            damping=1.0,      # Based on motor_parameters velocity.kp
        ),
        # Joint 7: Gripper
        "wxai_gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_carriage_joint", "right_carriage_joint"],
            effort_limit_sim=100.0,  # Match actual robot limits
            stiffness=20.0*100,   # Based on motor_parameters position.kp
            damping=1.0*100,
        ),
    },
)


# Reduce base gains to let wrist contribute more
WXAI_TEST_CFG.actuators["wxai_base"].stiffness = 300.0
WXAI_TEST_CFG.actuators["wxai_base"].damping = 60.0
WXAI_TEST_CFG.actuators["wxai_wrist"].stiffness = 60.0
WXAI_TEST_CFG.actuators["wxai_wrist"].damping = 8.0


@configclass
class WxaiCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Wxai as robot
        self.scene.robot = WXAI_TEST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        # self.actions.arm_action = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint_[0-5]"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_carriage_joint", "right_carriage_joint"],
            open_command_expr={"left_carriage_joint": 0.044, "right_carriage_joint": 0.044},
            close_command_expr={"left_carriage_joint": 0.0, "right_carriage_joint": 0.0},
        )

        # Set the body name for the end effector
        self.commands.object_pose.body_name = "ee_gripper_link"

        # Custom object pose command ranges for wxai
        self.commands.object_pose = mdp.UniformPoseCommandCfg(
                    asset_name="robot",
                    body_name="ee_gripper_link",  # will be set by agent env cfg
                    resampling_time_range=(5.0, 5.0),
                    debug_vis=True,
                    ranges=mdp.UniformPoseCommandCfg.Ranges(
                        pos_x=(0.25, 0.45), pos_y=(-0.25, 0.25), pos_z=(0.15, 0.4), roll=(3.14, 3.14), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
                    ),
                )

        # Custom reset for object position
        self.events.reset_object_position = EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.3, -0.1), "y": (-0.25, 0.05), "z": (0.0, 0.0)},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                },
            )
        
        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # large penalty for action rate to encourage smooth motions
        # self.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)

        # self.rewards.joint_vel = RewTerm(
        #         func=mdp.joint_vel_l2,
        #         weight=-1e-2,
        #         params={"asset_cfg": SceneEntityCfg("robot")},
        #     )

        # # add larger weight to object goal tracking
        # self.rewards.object_goal_tracking = RewTerm(
        #         func=mdp.object_goal_distance,
        #         params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        #         weight=16.0,
        #     )
        
        self.rewards.object_goal_tracking_fine_grained = RewTerm(
                func=mdp.object_goal_distance,
                params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
                weight=-5.0,
            )
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = "ee_gripper_link"
        

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/link_6", # link_6
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.1234, 0.0, 0.0], rot=(0.0, 0.707, 0.0, 0.707) # wxyz
                    ),
                ),
            ],
        )


@configclass
class WxaiCubeLiftEnvCfg_PLAY(WxaiCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

        self.scene.ee_frame.debug_vis = False

