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

# from isaaclab_tasks.manager_based.wxai_custom.wxai_custom_env_cfg import WxaiCustomEnvCfg 
from isaaclab_tasks.manager_based.manipulation.lift.config.wxai.joint_pos_env_cfg import WxaiCubeLiftEnvCfg 

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

    # simulate physics
    count = 0
    sim_time = 0.0
    sim_dt = env.sim.get_physics_dt()
    while simulation_app.is_running():
        with torch.inference_mode():
            # # reset
            # if count % 300 == 0:
            #     count = 0
            #     env.reset()
            #     print("-" * 80)
            #     print("[INFO]: Resetting environment...")
            # sample random actions
            # Apply wave
            wave_action = env.action_manager.action.clone() # n ,7
            # joint_efforts = torch.randn_like(env.action_manager.action)
            wave_action[:, :-1] = 0.35 * np.sin(2 * np.pi * 0.5 * sim_time)
            wave_action[:, -1] = 0.044 * np.sin(2 * np.pi * 0.5 * sim_time)
            # import pdb; pdb.set_trace()
            # step the environment
            obs, reward, terminated, truncated, info  = env.step(wave_action)
            obj_pos = obs['policy']['object_position'] #  n, 3
            target_pos = obs['policy']['target_object_position'] # n, 7
            print("-" * 80)
            print(f"[INFO]: Step {count}, Reward: {reward.mean().item():.3f}")
            # update counter
            count += 1
            sim_time += sim_dt # 100Hz

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
