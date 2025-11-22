# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from html import entities

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch


from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

# from isaaclab_tasks.manager_based.wxai_custom.wxai_custom_env_cfg import WxaiCustomEnvCfg 
from isaaclab_tasks.manager_based.manipulation.lift.config.wxai.joint_pos_env_cfg import WxaiCubeLiftEnvCfg 

from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers

import numpy as np
import time


def main():
    """Main function."""
    # create environment configuration
    env_cfg = WxaiCubeLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)


    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    
    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=env.num_envs, device=env.device)

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint_.*"], body_names=["ee_gripper_link"])
    robot = env.scene["robot"]
    # initialize ee goals
    ee_goals = torch.zeros(env.num_envs, diff_ik_controller.action_dim, device=robot.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(env.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    # Resolving the scene entities
    robot_entity_cfg.resolve(env.scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]
    # simulate physics
    count = 0
    sim_dt = env.sim.get_physics_dt()
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 150 == 0:
                count = 0
                obs, _  = env.reset()
                obj_pos = obs["policy"]["object_position"]  # n, 3
                # z goal above object
                obj_pos = obj_pos + torch.tensor([0.0, 0.0, 0.2], device=obj_pos.device)
                ee_goals = torch.cat([
                                        obj_pos,  # n, 3
                                        torch.tensor([0.707, 0, 0.707, 0.0], device=obj_pos.device).expand(obj_pos.shape[0], 4)
                                     ], dim=-1)  
                ee_goals = [
                    [0.2, 0.4, 0.3, 1.0, 0, 0, 0],
                    [0.2, -0.3, 0.5, 0.707, 0.707, 0.0, 0.0],
                    [0.2, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
                ]
                ee_goals = torch.tensor(ee_goals, device=env.sim.device)
                # reset joint state
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.reset()
                # reset actions
                # ik_commands[:] = ee_goals
                ik_commands[:] = ee_goals[current_goal_idx]
                joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
                # reset controller
                diff_ik_controller.reset()
                diff_ik_controller.set_command(ik_commands)
                # change goal
                current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            else:
                # obtain quantities from simulation
                jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
                ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
                root_pose_w = robot.data.root_pose_w
                joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
                # compute frame in root frame
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )
                # compute the joint commands
                joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            # joint pos and gripper
            action = torch.concat([joint_pos_des, 0.044 * torch.ones(env.num_envs, 1, device=env.device)], dim=-1)
            # step the environment
            obs, reward, terminated, truncated, info  = env.step(action)
            print("-" * 80)
            # Check if you're hitting limits - add this to your IK script
            print(f"Applied efforts: {robot.data.applied_torque}")
            print(f"Effort limits: {[27, 27, 27, 7, 7, 7]}")  # Check if torques are maxed out
            print(f"[INFO]: Step {count}, Reward: {reward.mean().item():.3f}")
            # update counter
            count += 1
            # obtain quantities from simulation
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            # update marker positions
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(ik_commands[:, 0:3] + env.scene.env_origins, ik_commands[:, 3:7])

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
