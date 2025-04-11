import neuralop.models.fno as fno
from jaxfluids_rl.envs.cylinder2D import Cylinder2DEnv
import torch.functional as F
import torch.nn as nn
import torch
import numpy as np
from stable_baselines3 import PPO,SAC,TD3,DDPG,A2C
from stable_baselines3.common.policies import ActorCriticPolicy
import os
from stable_baselines3.common.callbacks import CheckpointCallback
import time
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from config import get_args
from callbacks import RenderCallback, LoggingCallback
from policy import CustomPolicy


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
args = get_args()

project_name = "JAX_FLUIDS_Cylinder2D"
run_name = f"{args.algorithm}_{args.learning_rate}_{args.batch_size}"
render_dir = os.path.join("render", run_name)

save_dir = os.path.join("models", run_name)
os.makedirs(save_dir, exist_ok=True)
checkpoint_callback = CheckpointCallback(
    save_freq=1000,  
    save_path=save_dir,
    name_prefix="model_steps"
)
wandb.init(project=project_name, name=run_name, sync_tensorboard=True)


callback_list = [checkpoint_callback, RenderCallback(),LoggingCallback()]

env = Cylinder2DEnv(render_mode='save',episode_length=100,action_length=0.05,render_dir=render_dir)
env.reset()

policy = None
if args.policy == 'custom':
    policy = CustomPolicy
else:
    policy = args.policy


model = None
# Define the algorithm based on user input
if args.algorithm == 'sac':
    model = SAC(policy=policy,
                env=env,verbose=1,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                learning_rate=args.learning_rate,
                device='cuda')
elif args.algorithm == 'ppo':
    model = PPO(policy=policy,
                env=env,verbose=1,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device='cuda')
elif args.algorithm == 'td3':
    model = TD3(policy=policy,
                env=env,verbose=1,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                learning_rate=args.learning_rate,
                device='cuda')
elif args.algorithm == 'ddpg':
    model = DDPG(policy=policy,
                 env=env,verbose=1,
                 batch_size=args.batch_size,
                 buffer_size=args.buffer_size,
                 learning_rate=args.learning_rate,
                 device='cuda')
elif args.algorithm == 'a2c':
    model = A2C(policy=policy,
                env=env,
                verbose=1,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device='cuda')
else:
    raise ValueError(f"Algorithm {args.algorithm} not supported.")

model.learn(total_timesteps=args.training_steps,callback=callback_list)