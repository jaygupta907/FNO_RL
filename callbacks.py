from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
import wandb
import numpy as np


class RenderCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.render_freq = 500
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.render_freq == 0:
            self.training_env.render()
        return True  
    
class LoggingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        
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