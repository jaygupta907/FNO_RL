import neuralop.models.fno as fno
from jaxfluids_rl.envs.cylinder2D import Cylinder2DEnv
import torch.functional as F
import torch.nn as nn
import torch
import numpy as np
from stable_baselines3 import PPO,SAC,TD3,DDPG
from stable_baselines3.common.policies import ActorCriticPolicy
import os
from stable_baselines3.common.callbacks import CheckpointCallback
import time
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from config import get_args
from callbacks import RenderCallback, LoggingCallback
from policy import CustomPolicy
args = get_args()

project_name = "JAX_FLUIDS_Cylinder2D"
run_name = f"{args.algorithm}_{args.learning_rate}_{args.batch_size}"

save_dir = os.path.join("models", run_name)
os.makedirs(save_dir, exist_ok=True)
checkpoint_callback = CheckpointCallback(
    save_freq=1000,  
    save_path=save_dir,
    name_prefix="model_steps"
)
wandb.init(project=project_name, name=run_name, sync_tensorboard=True)


callback_list = [checkpoint_callback, RenderCallback(),LoggingCallback()]

env = Cylinder2DEnv(render_mode='save',episode_length=100,action_length=0.05)

algorithm = None
if args.algorithm == 'sac':
    algorithm = SAC
elif args.algorithm == 'ppo':
    algorithm = PPO
elif args.algorithm == 'td3':
    algorithm = TD3
elif args.algorithm == 'ddpg':
    algorithm = DDPG
else:
    raise ValueError(f"Algorithm {args.algorithm} not supported.")

if args.policy == 'custom':
    model = algorithm(policy=CustomPolicy,env=env,verbose=1,batch_size=16,learning_rate=0.001,device='cuda')

else:
    model = algorithm(policy=args.policy,env=env,verbose=1,batch_size=args.batch_size,learning_rate=args.learning_rate,device='cuda')
model.learn(total_timesteps=100000,callback=callback_list)