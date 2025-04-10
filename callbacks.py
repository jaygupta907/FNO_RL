from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
import wandb
import numpy as np


class RenderCallback(BaseCallback):
    def _on_step(self) -> bool:
        self.training_env.render()  
        return True  
    
class LoggingCallback(BaseCallback):
    def __init__(self, project_name="FNO_JAX_FLUIDS_Cylinder2D", run_name="PPO_Experiment_2"):
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