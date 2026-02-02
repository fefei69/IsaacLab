"""Script to test the control of rrl m3 robot."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=5, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.rrl_mobile_manipulation.rrl_mobile_manipulation_env_cfg import ThrusterCylinderEnvCfg
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg: ThrusterCylinderEnvCfg = parse_env_cfg(
            "RRL-M3-Direct-v0",
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        ) #type: ignore
    
    env_cfg.episode_length_s = 100.0 # 100 seconds per episode for testing

    # create environment
    env = gym.make("RRL-M3-Direct-v0", cfg=env_cfg)
    # reset environment at start
    env.reset()

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # goal_position = torch.tensor([[0.5, 0.2]], device=env.unwrapped.device).expand(env.unwrapped.num_envs, -1)
    goal_position = torch.tensor([[0.5, 0.0]], device=env.unwrapped.device).expand(args_cli.num_envs, -1)
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            ## P positional control towards goal position ##
            xy_pos_error = goal_position - env.unwrapped.robot.data.root_pos_w[:, :2]
            u_pred = xy_pos_error * 5.0  # P gain = 5.0
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            print(f"[INFO]: XY Pos Error: {xy_pos_error.cpu().numpy()}")
            
            # apply actions
            obs = env.step(actions)[0]
            current_step += 1

            print(f"[INFO]: Lin vel: {obs['policy'][0, 7:10].cpu().numpy()}, "
                  f"Ang vel: {obs['policy'][0, 10:13].cpu().numpy()}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
