import neuralop.models.fno as fno
import torch.functional as F
import torch.nn as nn
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from neuralop.layers.spectral_convolution import SpectralConv1d

class CustomActorCritic(nn.Module):
    def __init__(self,
                in_channels=2,
                out_channels=1,
                hidden_channels=64,
                n_modes_height=10,
                n_layers=3,
                action_dim=1,
                obs_dim=75):
        super(CustomActorCritic, self).__init__()

        # FNO Backbone
        self.fno = fno.FNO1d(in_channels=in_channels,
                            out_channels=out_channels,
                            hidden_channels=hidden_channels,
                            n_modes_height=n_modes_height,
                            n_layers=n_layers)

        # Value Function
        self.value = nn.Sequential(
            fno.FNO1d(in_channels=1,out_channels=1,n_layers=3,n_modes_height=16,hidden_channels=64),
            nn.Flatten(),
            nn.Linear(75, 1)
        )

        # Mean Action Output
        self.action_mean = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # Learnable Standard Deviation
        self.action_std_layer = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        x = x.float()
        x = self.fno(x)
        value = self.value(x)
        x = x.view(x.size(0), -1)
        action_mean = self.action_mean(x)
        action_log_std = self.action_std_layer(x)  
        action_std = torch.exp(action_log_std)  

        return value, action_mean, action_std


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

