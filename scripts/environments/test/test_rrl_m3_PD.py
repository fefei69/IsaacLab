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
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.direct.rrl_mobile_manipulation.rrl_m3.rrl_m3_env_cfg import M3EnvCfg
from isaaclab_tasks.utils import parse_env_cfg


from isaaclab_tasks.direct.rrl_mobile_manipulation.thruster_layout_cfg import ThrusterLayoutCfg
thruster_cfg = ThrusterLayoutCfg()



def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg: M3EnvCfg = parse_env_cfg(
            "RRL-M3-Direct-v0",
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        ) #type: ignore
    
    env_cfg.episode_length_s = 30.0 # 100 seconds per episode for testing
    env_cfg.thrusters.max_thrust = 1.7

    # create environment
    env = gym.make("RRL-M3-Direct-v0", cfg=env_cfg)
    # reset environment at start
    env.reset()

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    current_step = 0
    # simulate environment
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    KP_POS = 1.5
    KD_POS = 0.1
    KP_ORI = 0.5
    KD_ORI = 0.1
    B = torch.tensor([[1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],   # Fx contribution from each thruster
                      [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0],   # Fy contribution from each thruster
                      [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0]], device=env.unwrapped.device) # (3, 3) control allocation matrix for surge, sway, yaw
    M = torch.linalg.pinv(B) # (3, 8) pseudo-inverse for control allocation
    robot = env.unwrapped.scene["robot"]
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # apply actions
            obs, rews, _, _, _ = env.step(actions)
            robot_states = obs['policy']
            lin_vel_b = robot_states[:, :3]
            ang_vel_b = robot_states[:, 3:6]
            desired_pos_b = robot_states[:, 6:9]
            desired_ori_b = robot_states[:, 9:13]
            F_x_des_b = KP_POS * desired_pos_b[:, 0] - KD_POS * lin_vel_b[:, 0] # (num_envs, )
            F_y_des_b = KP_POS * desired_pos_b[:, 1] - KD_POS * lin_vel_b[:, 1]
            roll, pitch, yaw = euler_xyz_from_quat(desired_ori_b)
            tau_z_des_b = KP_ORI * yaw - KD_ORI * ang_vel_b[:, 2]  # yaw control
            actions[:, :8] = torch.matmul(torch.stack([F_x_des_b, F_y_des_b, tau_z_des_b], dim=1), M.T) # (num_envs, 8)
            current_step += 1
            print(f"[INFO]: Step: {current_step}, yaw difference: {torch.rad2deg(yaw).tolist()} deg, actions: {actions.cpu().numpy()}")


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
