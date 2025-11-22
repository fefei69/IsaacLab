
export CUDA_VISIBLE_DEVICES=0
# LIVESTREAM=2 ./isaaclab.sh -p scripts/tools/convert_urdf.py /workspace/isaaclab/external_assets/trossen_arm_description/urdf/generated/wxai/wxai_base.urdf /workspace/external_assets/wxai_base.usd \
#     --fix-base \
#     --joint-stiffness 200.0 \
#     --joint-damping 2.0 

# LIVESTREAM=2 ./isaaclab.sh -p scripts/tools/convert_mjcf.py /workspace/isaaclab/external_assets/rrl_atmos_mjcf/atmos_rrl.xml /workspace/isaaclab/external_assets/atmos_v0.usd 

# LIVESTREAM=2 ./isaaclab.sh -p scripts/tools/convert_mjcf.py /workspace/isaaclab/external_assets/atmos/atmos_rrl.xml /workspace/isaaclab/external_assets/atmos.usd 

# Isaac-Lift-Cube-Wxai-v0
#  LIVESTREAM=2 ./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Wxai-Custom-v0

# LIVESTREAM=2 ./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Franka-Cabinet-Direct-v0 --num_env 5

# LIVESTREAM=2 ./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Lift-Cube-Franka-v0 --num_env 5

# LIVESTREAM=2 ./isaaclab.sh -p scripts/tutorials/05_controllers/run_osc.py --num_env 5 --rendering_mode performance
# LIVESTREAM=2 ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py --num_env 5 --rendering_mode performance
# LIVESTREAM=2 ./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Lift-Cube-Wxai-v0 --num_env 2 --rendering_mode performance
# LIVESTREAM=2 ./isaaclab.sh -p scripts/tutorials/03_envs/test_rl_env_ik.py --num_env 3 
# LIVESTREAM=2 ./isaaclab.sh -p scripts/tutorials/03_envs/test_rl_env.py 

# LIVESTREAM=2 ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik_wxai.py 
# LIVESTREAM=2 ./isaaclab.sh -p scripts/environments/state_machine/lift_cube_sm.py --num_envs 5

# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Cube-Wxai-v0 --headless #--resume
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Cube-Wxai-IK-Abs-v0
LIVESTREAM=2 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Lift-Cube-Wxai-Play-v0 

# Open Drawer with Franka
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Open-Drawer-Franka-v0 --headless --resume
# LIVESTREAM=2 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Open-Drawer-Franka-v0 

# Zero Agent Test
# LIVESTREAM=2 ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Lift-Cube-Wxai-Play-v0 --num_envs 32
# LIVESTREAM=2 ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Deploy-Reach-UR10e-v0
# LIVESTREAM=2 ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 32

# LIVESTREAM=2 ./isaaclab.sh -p scripts/tutorials/01_assets/test_articulation_atmos.py  

# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Cube-Franka-v0 --headless #--resume
# LIVESTREAM=2 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Lift-Cube-Franka-Play-v0 