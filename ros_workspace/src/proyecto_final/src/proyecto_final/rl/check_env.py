from stable_baselines3.common.env_checker import check_env
import sys
sys.path.append("/home/laboratorio/ros_workspace/src/proyecto_final/scripts/rl")
from proyecto_final.rl.env_rob_train_discrete import ROSEnv

env = ROSEnv(2)
# It will check your custom environment and output additional warnings if needed
check_env(env)