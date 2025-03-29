import neuralop.models.fno as fno
from jaxfluids_rl.envs.cylinder2D import Cylinder2DEnv
import time
import torch.functional as F
import torch.nn as nn
import torch
import numpy as np
from stable_baselines3 import PPO,SAC
from stable_baselines3.common.policies import ActorCriticPolicy
import os
from stable_baselines3.common.callbacks import CheckpointCallback
import time
from stable_baselines3.common.callbacks import BaseCallback
import wandb



class CustomActorCritic(nn.Module):
    def __init__(self, 
                 in_channels=2, 
                 out_channels=1, 
                 hidden_channels=32, 
                 n_modes_height=10, 
                 n_layers=3, 
                 action_dim=1, 
                 obs_dim=75):
        super(CustomActorCritic, self).__init__()

        self.fno = fno.FNO1d(in_channels=in_channels,
                             out_channels=out_channels,
                             hidden_channels=hidden_channels,
                             n_modes_height=n_modes_height,
                             n_layers=n_layers)
        
        self.value = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.action_mean = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # Output mean for normal distribution
        )

        self.action_log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable std dev

    def forward(self, x):
        x = x.float()
        x = self.fno(x)
        x = x.view(x.size(0), -1)  # Flatten

        value = self.value(x)
        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)  # Expand to match mean
        action_std = torch.exp(action_log_std)  # Ensure positivity

        return value, action_mean, action_std  # Now returning mean & std for a normal distribution


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

        obs_dim = observation_space.shape[1]  # Correct observation space shape
        action_dim = action_space.shape[0]  # Continuous action space

        self.ac_network = CustomActorCritic(action_dim=action_dim, obs_dim=obs_dim)

    def forward(self, obs, deterministic=False):
        value, action_mean, action_std = self.ac_network(obs)

        # Sample from a normal distribution
        action_dist = torch.distributions.Normal(action_mean, action_std)
        actions = action_dist.sample() if not deterministic else action_mean
        log_probs = action_dist.log_prob(actions).sum(dim=-1)  # Sum over action dimensions

        # Ensure actions stay in valid range
        actions = torch.tanh(actions)  

        return actions, value, log_probs



save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=1000,  
    save_path=save_dir,
    name_prefix="ppo_cylinder2D"
)

class RenderCallback(BaseCallback):
    def _on_step(self) -> bool:
        self.training_env.render()  
        return True  
    
class LoggingCallback(BaseCallback):
    def __init__(self, project_name="FNO_JAX_FLUIDS_Cylidner2D", run_name="PPO_Experiment_1"):
        super().__init__()
        wandb.init(project=project_name, name=run_name, sync_tensorboard=True)

    def _on_step(self) -> bool:
        
        action = self.locals["actions"]
        reward = self.locals["rewards"]

        
        wandb.log({
            "action_mean": np.mean(action),  
            "reward_mean": np.mean(reward),  
            "step": self.num_timesteps
        })

        return True 

    def _on_training_end(self) -> None:
        wandb.finish()

callback_list = [checkpoint_callback, RenderCallback(),LoggingCallback()]

env = Cylinder2DEnv(render_mode='save',episode_length=100,action_length=0.05)
model = PPO(policy=CustomPolicy,env=env,verbose=1,batch_size=16,learning_rate=0.001,device='cuda')
model.learn(total_timesteps=100000,callback=callback_list)




# model =  fno.FNO1d(in_channels=2, out_channels=1,hidden_channels=32,n_modes_height=10,n_layers=3)

# model = model.to('cuda')
# env = Cylinder2DEnv(render_mode='save',episode_length=100,action_length=0.05)

# observation, info = env.reset()
# env.render()
# for i in range(1000):
#     t1 = time.time()
#     action = env.action_space.sample()  
#     observation, reward, terminated, truncated, info = env.step(0.0)
#     observation = observation.copy()
#     observation = torch.from_numpy(np.array(observation)).unsqueeze(0)
#     observation = observation.to(torch.float32).to('cuda')
#     out = model(observation)
#     print(out.shape)
#     t2 = time.time()
#     print(f"EPISODE {i:04d}, reward {reward:.3f}, terminated {terminated}, WCT {t2-t1:.2f}s")
#     env.render()
#     if terminated or truncated:
#         observation, info = env.reset()