"""
RRL Microgravity Mobile Manipulator (M3) in lab env 
"""

import gymnasium as gym

# RL config has yet configuerd
from . import agents 

gym.register(
    id="RRL-CylinderArm-Direct-v0",
    entry_point=f"{__name__}.cylinder_arm_env:M3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cylinder_arm_env_cfg:CylinderArmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RRLM3PPORunnerCfg",
    },
)


