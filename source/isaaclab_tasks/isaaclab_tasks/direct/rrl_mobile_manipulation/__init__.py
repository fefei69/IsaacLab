"""
RRL Microgravity Mobile Manipulator (M3) in lab env 
"""

import gymnasium as gym

# RL config has yet configuerd
# from . import agents 

gym.register(
    id="RRL-M3-Direct-v0",
    entry_point=f"{__name__}.rrl_mobile_manipulation_env:ThrusterCylinderEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rrl_mobile_manipulation_env:ThrusterCylinderEnvCfg",
    },
)



