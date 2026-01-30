"""Thruster Cylinder Environment for Isaac Lab."""

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
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.50, 0.50, 0.50)

##
# Thruster Layout Configuration
##

@configclass
class ThrusterLayoutCfg:
    """Configuration for thruster positions and orientations."""
    
    # Thruster positions relative to body center (x, y, z)
    # Front-Right, Front-Left, Back-Right, Back-Left (pointing along X)
    # Right-Front, Left-Front, Right-Back, Left-Back (pointing along Y)
    positions: Dict[str, tuple] = field(default_factory=lambda: {
        "FR": (0.137, -0.120, 0.29757938),
        "FL": (0.137, 0.120, 0.29757938),
        "BR": (-0.137, -0.120, 0.29757938),
        "BL": (-0.137, 0.120, 0.29757938),
        "RF": (0.120, -0.137, 0.30957938),
        "LF": (0.120, 0.137, 0.30957938),
        "RB": (-0.120, -0.137, 0.30957938),
        "LB": (-0.120, 0.137, 0.30957938),
    })
    
    # Thrust directions for each thruster
    directions: Dict[str, tuple] = field(default_factory=lambda: {
        "FR": (1.0, 0.0, 0.0),   # Push forward
        "FL": (1.0, 0.0, 0.0),   # Push forward
        "BR": (-1.0, 0.0, 0.0),  # Push backward
        "BL": (-1.0, 0.0, 0.0),  # Push backward
        "RF": (0.0, -1.0, 0.0),  # Push right
        "RB": (0.0, -1.0, 0.0),  # Push right
        "LF": (0.0, 1.0, 0.0),   # Push left
        "LB": (0.0, 1.0, 0.0),   # Push left
    })
    
    # Maximum thrust force per thruster (Newtons)
    max_thrust: float = 10.0


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
    robot: RigidObjectCfg = RRLM3_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Frame visualization for robot
    robot_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot",  # Source frame (robot root)
        debug_vis=True,
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
        env_spacing=4.0,
    )
    
    # Thruster configuration
    thrusters: ThrusterLayoutCfg = ThrusterLayoutCfg()
    
    # Reward scales
    reward_forward_velocity: float = 1.0  # Reward for moving forward (+X)
    reward_lateral_penalty: float = -0.1  # Penalty for lateral velocity
    reward_angular_penalty: float = -0.05  # Penalty for rotation
    reward_action_penalty: float = -0.001  # Small penalty for using thrusters
    
    # Target forward velocity (m/s)
    target_velocity: float = 2.0




class ThrusterCylinderEnv(DirectRLEnv):
    """
    A simple environment with a cylinder robot controlled by 8 thrusters.
    
    The robot has 8 thrusters arranged around the cylinder body:
    - 4 thrusters for forward/backward motion (FR, FL, BR, BL)
    - 4 thrusters for lateral motion (RF, LF, RB, LB)
    
    Action space: 8 continuous values in [-1, 1], mapped to thrust forces
    Observation space: [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13 dims
    
    Goal: Move forward (+X direction) as fast as possible.
    """
    
    cfg: ThrusterCylinderEnvCfg
    
    def __init__(self, cfg: ThrusterCylinderEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Cache thruster positions and directions as tensors
        self._setup_thrusters()
        
        # Track episode statistics
        self._episode_sums = {
            "forward_velocity": torch.zeros(self.num_envs, device=self.device),
            "lateral_velocity": torch.zeros(self.num_envs, device=self.device),
            "distance_traveled": torch.zeros(self.num_envs, device=self.device),
        }
    
    def _setup_thrusters(self):
        """Pre-compute thruster positions and directions as tensors."""
        thruster_names = ["FR", "FL", "BR", "BL", "RF", "LF", "RB", "LB"]
        
        # Positions: (8, 3)
        positions = []
        directions = []
        for name in thruster_names:
            positions.append(self.cfg.thrusters.positions[name])
            directions.append(self.cfg.thrusters.directions[name])
        
        self._thruster_positions = torch.tensor(
            positions, dtype=torch.float32, device=self.device
        )  # (8, 3)
        
        self._thruster_directions = torch.tensor(
            directions, dtype=torch.float32, device=self.device
        )  # (8, 3)
        
        self._max_thrust = self.cfg.thrusters.max_thrust
        self._num_thrusters = len(thruster_names)
    
    def _setup_scene(self):
        """Set up the scene with robot and ground."""
        # Add robot to scene
        self.robot = self.scene["robot"]

        # Add ground plane (handled by terrain importer in scene config)
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Process actions before physics simulation step.
        
        Actions are normalized [-1, 1] and mapped to thrust forces [0, max_thrust].
        We use (action + 1) / 2 to map [-1, 1] -> [0, 1], then scale by max_thrust.
        """
        self._actions = actions.clone()
        
        # Map actions from [-1, 1] to [0, max_thrust]
        # This ensures thrusters can only push, not pull
        thrust_magnitudes = (actions + 1.0) * 0.5 * self._max_thrust  # (num_envs, 8)
        
        # Compute forces in body frame
        # forces = magnitude * direction for each thruster
        # thrust_magnitudes: (num_envs, 8) -> (num_envs, 8, 1)
        # directions: (8, 3) -> (1, 8, 3)
        forces_body = thrust_magnitudes.unsqueeze(-1) * self._thruster_directions.unsqueeze(0)
        # forces_body: (num_envs, 8, 3)
        
        # Sum all thruster forces to get total force
        total_force_body = forces_body.sum(dim=1)  # (num_envs, 3)
        
        # Compute torques: torque = position × force
        torques_body = torch.cross(
            self._thruster_positions.unsqueeze(0).expand(self.num_envs, -1, -1),
            forces_body,
            dim=-1
        )
        total_torque_body = torques_body.sum(dim=1)  # (num_envs, 3)
        
        # Transform forces and torques to world frame using robot orientation
        robot_quat = self.robot.data.root_quat_w  # (num_envs, 4) in (w, x, y, z) format
        
        # Rotate force and torque vectors to world frame
        total_force_world = self._quat_rotate(robot_quat, total_force_body)
        total_torque_world = self._quat_rotate(robot_quat, total_torque_body)
        
        # Apply external forces and torques
        # Isaac Lab expects forces at body positions
        self._applied_forces = total_force_world
        self._applied_torques = total_torque_world
    
    def _apply_action(self):
        """Apply the computed forces and torques to the robot."""
        # Get robot body indices
        body_ids = self.robot.find_bodies(".*")[0]
        
        # Apply external wrench (force + torque) to robot
        # Forces shape: (num_envs, num_bodies, 3)
        # Torques shape: (num_envs, num_bodies, 3)
        forces = self._applied_forces.unsqueeze(1)  # (num_envs, 1, 3)
        torques = self._applied_torques.unsqueeze(1)  # (num_envs, 1, 3)
        
        #################
        # deprecated API, check https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html#external-force-and-torque-application-wrench-composers
        #################
        # self.robot.set_external_force_and_torque(
        #     forces=forces,
        #     torques=torch.zeros_like(torques),
        #     body_ids=body_ids
        #     )
        
        self.robot.instantaneous_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torch.zeros_like(torques),
            body_ids=body_ids,
        )

        # TODO: consider applying force to specific poistion, not just center of mass, see Isaac Lab docs. 

    def _get_observations(self) -> dict:
        """
        Compute observations.
        
        Observations include:
        - Position (3): x, y, z in world frame
        - Orientation (4): quaternion (w, x, y, z)
        - Linear velocity (3): vx, vy, vz in world frame
        - Angular velocity (3): wx, wy, wz in world frame
        
        Total: 13 dimensions
        """
        obs = torch.cat([
            self.robot.data.root_pos_w,           # (num_envs, 3)
            self.robot.data.root_quat_w,          # (num_envs, 4)
            self.robot.data.root_lin_vel_w,       # (num_envs, 3)
            self.robot.data.root_ang_vel_w,       # (num_envs, 3)
        ], dim=-1)
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """
        Compute rewards for moving forward.
        
        Reward components:
        1. Forward velocity reward: positive for moving in +X direction
        2. Lateral velocity penalty: negative for moving in Y direction
        3. Angular velocity penalty: negative for rotation
        4. Action penalty: small negative for using thrust (energy efficiency)
        """
        # Get velocities in world frame
        lin_vel = self.robot.data.root_lin_vel_w  # (num_envs, 3)
        ang_vel = self.robot.data.root_ang_vel_w  # (num_envs, 3)
        
        # Forward velocity (X direction)
        forward_vel = lin_vel[:, 0]
        
        # Lateral velocity (Y direction)
        lateral_vel = torch.abs(lin_vel[:, 1])
        
        # Angular velocity magnitude
        ang_vel_mag = torch.norm(ang_vel, dim=-1)
        
        # Action magnitude (thrust usage)
        action_mag = torch.sum(torch.abs(self._actions), dim=-1)
        
        # Compute reward components
        # Reward is higher when closer to target velocity
        forward_reward = self.cfg.reward_forward_velocity * (
            forward_vel - 0.5 * torch.abs(forward_vel - self.cfg.target_velocity)
        )
        lateral_penalty = self.cfg.reward_lateral_penalty * lateral_vel
        angular_penalty = self.cfg.reward_angular_penalty * ang_vel_mag
        action_penalty = self.cfg.reward_action_penalty * action_mag
        
        # Total reward
        reward = forward_reward + lateral_penalty + angular_penalty + action_penalty
        
        # Update episode statistics
        self._episode_sums["forward_velocity"] += forward_vel
        self._episode_sums["lateral_velocity"] += lateral_vel
        self._episode_sums["distance_traveled"] += forward_vel * self.step_dt
        
        return reward
    
    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check for episode termination.
        
        Episodes terminate when:
        1. Time limit reached
        2. Robot falls below ground (z < 0)
        3. Robot tips over (large roll/pitch angle)
        """
        # Time limit
        time_out = self.episode_length_buf >= self.max_episode_length
        
        # Robot fell
        fell = self.robot.data.root_pos_w[:, 2] < 0.0
        
        # Robot tipped over (check if up vector is pointing down)
        # Get the up vector in world frame by rotating [0, 0, 1] by robot orientation
        up_world = self._quat_rotate(
            self.robot.data.root_quat_w,
            torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(self.num_envs, -1)
        )
        tipped = up_world[:, 2] < 0.3  # Cosine of ~73 degrees
        
        # Combine termination conditions
        terminated = fell | tipped
        truncated = time_out & ~terminated
        
        return terminated, truncated
    
    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specified environments."""
        super()._reset_idx(env_ids)
        
        # Reset robot state
        num_resets = len(env_ids)
        
        # Random initial positions with small variation
        default_pos = torch.tensor(
            self.cfg.scene.robot.init_state.pos,
            device=self.device
        ).unsqueeze(0).expand(num_resets, -1).clone()
        
        # Add small random offset to x, y positions
        default_pos[:, :2] += torch.randn(num_resets, 2, device=self.device) * 0.1
        
        # Default orientation (identity quaternion)
        default_quat = torch.tensor(
            self.cfg.scene.robot.init_state.rot,
            device=self.device
        ).unsqueeze(0).expand(num_resets, -1).clone()
        
        # Zero initial velocities
        default_lin_vel = torch.zeros(num_resets, 3, device=self.device)
        default_ang_vel = torch.zeros(num_resets, 3, device=self.device)
        
        # Write to simulation
        self.robot.write_root_pose_to_sim(
            torch.cat([default_pos, default_quat], dim=-1),
            env_ids
        )
        self.robot.write_root_velocity_to_sim(
            torch.cat([default_lin_vel, default_ang_vel], dim=-1),
            env_ids
        )
        
        # Reset episode statistics
        for key in self._episode_sums:
            self._episode_sums[key][env_ids] = 0.0
    
    @staticmethod
    def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """
        Rotate vectors by quaternions.
        
        Args:
            quat: Quaternions (w, x, y, z) of shape (..., 4)
            vec: Vectors of shape (..., 3)
        
        Returns:
            Rotated vectors of shape (..., 3)
        """
        # Extract quaternion components
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        
        # Extract vector components
        vx, vy, vz = vec[..., 0], vec[..., 1], vec[..., 2]
        
        # Quaternion rotation formula: q * v * q^(-1)
        # Optimized computation
        ww = w * w
        xx = x * x
        yy = y * y
        zz = z * z
        wx = w * x
        wy = w * y
        wz = w * z
        xy = x * y
        xz = x * z
        yz = y * z
        
        # Rotation matrix elements
        r00 = ww + xx - yy - zz
        r01 = 2 * (xy - wz)
        r02 = 2 * (xz + wy)
        r10 = 2 * (xy + wz)
        r11 = ww - xx + yy - zz
        r12 = 2 * (yz - wx)
        r20 = 2 * (xz - wy)
        r21 = 2 * (yz + wx)
        r22 = ww - xx - yy + zz
        
        # Apply rotation
        out_x = r00 * vx + r01 * vy + r02 * vz
        out_y = r10 * vx + r11 * vy + r12 * vz
        out_z = r20 * vx + r21 * vy + r22 * vz
        
        return torch.stack([out_x, out_y, out_z], dim=-1)