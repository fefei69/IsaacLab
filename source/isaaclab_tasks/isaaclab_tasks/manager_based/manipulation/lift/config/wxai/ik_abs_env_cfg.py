# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import DeformableObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
# from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
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
            stiffness=20.0,   # Based on motor_parameters position.kp
            damping=1.0,
        ),
    },
)


# Reduce base gains to let wrist contribute more
WXAI_TEST_CFG.actuators["wxai_base"].stiffness = 300.0
WXAI_TEST_CFG.actuators["wxai_base"].damping = 60.0
WXAI_TEST_CFG.actuators["wxai_wrist"].stiffness = 60.0
WXAI_TEST_CFG.actuators["wxai_wrist"].damping = 8.0

##
# Rigid object lift environment.
##

@configclass
class WxaiCubeLiftEnvCfg(joint_pos_env_cfg.WxaiCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = WXAI_TEST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint_.*"],
            body_name="ee_gripper_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            # body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.05, 0.0, 0.0]),
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

