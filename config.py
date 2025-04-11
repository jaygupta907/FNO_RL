import argparse 

def get_args():
    """
    Parses command-line arguments and returns them.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='JAX Fluids RL Training Configuration')

    # Adding various arguments for training configuration
    parser.add_argument('--algorithm', type=str, default='ddpg', help='The RL algorithm to use (e.g., sac, ppo)')
    parser.add_argument('--policy', type=str, default='MlpPolicy', help='The policy architecture to use (e.g., MlpPolicy, CustomPolicy)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--buffer_size', type=int, default=50000, help='Buffer size for experience replay (if applicable)')
    parser.add_argument('--training_steps', type=int, default=100000, help='Total number of timesteps for training')


    return parser.parse_args()