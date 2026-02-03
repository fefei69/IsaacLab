from __future__ import annotations
from dataclasses import MISSING, field

import torch
from typing import Dict, Tuple

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg

from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets import RRLM3_CFG  # isort: skip
from isaaclab.markers import VisualizationMarkers  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG, RED_ARROW_X_MARKER_CFG  # isort: skip
from .thruster_layout_cfg import ThrusterLayoutCfg  # isort: skip

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy() # type: ignore
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.50, 0.50, 0.50)

FORCE_ARROW_CFG = RED_ARROW_X_MARKER_CFG.copy() # type: ignore


##
# Scene Configuration
##

@configclass
class ThrusterCylinderSceneCfg(InteractiveSceneCfg):
    """Configuration for the thruster cylinder scene."""
    
    # Ground plane with zero friction
    ground = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min",
            restitution_combine_mode="max",
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.0,
        ),
    ) 

    # Cylinder robot as a rigid object
    robot: RigidObjectCfg = RRLM3_CFG.replace(prim_path="/World/envs/env_.*/Robot") # type: ignore

    # Frame visualization for robot
    robot_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot",  # Source frame (robot root)
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/RobotFrameTransformer"),
        target_frames=[
            # Visualize the robot body frame itself
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                name="robot_body",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0, 0.0, 0.0, 0.0),  # identity quaternion (w, x, y, z)
                ),
            ),
        ],
    )



##
# Environment Configuration
##

@configclass
class ThrusterCylinderEnvCfg(DirectRLEnvCfg):
    """Configuration for the thruster cylinder RL environment."""
    
    # Environment settings
    decimation = 2  # Control frequency = sim_dt * decimation
    episode_length_s = 10.0  # 10 seconds per episode
    
    # Action space: 8 thrusters (normalized [-1, 1] mapped to [0, max_thrust])
    action_space = 8
    
    # Observation space: position(3) + orientation(4) + linear_vel(3) + angular_vel(3) = 13
    observation_space = 13
    
    # No state space for asymmetric actor-critic
    state_space = 0

    debug_vis = True  
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,  # 120 Hz simulation
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),  # Normal gravity
        physx=PhysxCfg(
            solver_type=1,  # TGS solver
            enable_stabilization=True,
        ),
    )
    
    # Scene configuration
    scene: ThrusterCylinderSceneCfg = ThrusterCylinderSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
    )
    
    # Thruster configuration
    thrusters: ThrusterLayoutCfg = ThrusterLayoutCfg()
    
    # Reward scales
    lin_vel_reward_scale: float = -0.05
    ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 15.0
    reward_action_penalty: float = -0.001  # Small penalty for using thrusters
    