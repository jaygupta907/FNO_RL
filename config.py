import argparse 

def get_args():
    """
    Parses command-line arguments and returns them.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Multilayer Feedforward Neural Network')

    # Adding various arguments for training configuration
    parser.add_argument('--algorithm', type=str, default='sac', help='The RL algorithm to use (e.g., sac, ppo)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    

    return parser.parse_args()