"""Configuration for the rrl Microgravity Mobile Manipulator robot"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

RRL_CYLINDER_ROBOT_USD_PATH = "/home/cpw/workspace/IsaacLab/assets/RRL_Cylinder_Robot/cylinder_robot.usd"
RRLM3_USD_PATH = "/home/cpw/workspace/IsaacLab/assets/M3_Robot_v1000/M3_Robot_v1000.usd"

# For now, treat Cylinder as a our robot base
CYLINDER_CFG = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.CylinderCfg(
            radius=0.2, # meters
            height=0.7, # meters
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0), # in kg
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.6, 0.9),
                metallic=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),  # Start above ground
            rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (w, x, y, z)
        ),
    )


CYLINDER_WXAI_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=RRL_CYLINDER_ROBOT_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        rot=(1.0, 0.0, 0.0, 0.0),  
        joint_pos={
            "joint_0": 0.0,
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 0.0,
            "joint_4": 0.0,
            "joint_5": 0.0,
            "left_carriage_joint": 0.0,
        },
    ),
    actuators={
        "wxai_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[0-5]"],
            stiffness=None,
            damping=None,
        ),
        # right_carriage_joint is a mimic joint specified in USD file
        "wxai_gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_carriage_joint"],
            stiffness=None,
            damping=None,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


RRLM3_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=RRLM3_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            fix_root_link=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.01),
        rot=(1.0, 0.0, 0.0, 0.0),  
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    # stiffness and damping are copy from their USD https://github.com/TrossenRobotics/trossen_ai_isaac/blob/main/assets/robots/wxai/wxai_follower.usd
    actuators={
        "joint0": ImplicitActuatorCfg(
            joint_names_expr=["joint_0"],
            stiffness=663.97,
            damping=5.53,
        ),
        "joint1": ImplicitActuatorCfg(
            joint_names_expr=["joint_1"],
            stiffness=734.56,
            damping=6.12,
        ),
        "joint2": ImplicitActuatorCfg(
            joint_names_expr=["joint_2"],
            stiffness=737.6,
            damping=6.14,
        ),
        "joint3": ImplicitActuatorCfg(
            joint_names_expr=["joint_3"],
            stiffness=62.3,
            damping=0.78,
        ),
        "joint4": ImplicitActuatorCfg(
            joint_names_expr=["joint_4"],
            stiffness=33.82,
            damping=0.84,
        ),
        "joint5": ImplicitActuatorCfg(
            joint_names_expr=["joint_5"],
            stiffness=33.33,
            damping=0.83,
        ),
        # right_carriage_joint is a mimic joint specified in USD file
        "wxai_gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_carriage_joint"],
            stiffness=217687.0625,
            damping=10884.34961,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

"""Configuration for the RRL Microgravity Mobile Manipulator robot."""
