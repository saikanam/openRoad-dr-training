import argparse
import os
import d3rlpy # Import top-level d3rlpy for seed setting
from d3rlpy.dataset import MDPDataset, ReplayBuffer, InfiniteBuffer # Use MDPDataset, ReplayBuffer, InfiniteBuffer
from datetime import datetime # Add datetime import
# Import Decision Transformer components
from d3rlpy.algos import DecisionTransformerConfig, DiscreteDecisionTransformerConfig
from d3rlpy.algos.transformer.action_samplers import IdentityTransformerActionSampler # Import action sampler for eval
from d3rlpy.optimizers import GPTAdamWFactory # Optimizer often used with transformers
from d3rlpy.metrics import InitialStateValueEstimationEvaluator # Keep this evaluator
from d3rlpy.metrics import ContinuousActionDiffEvaluator # To track action prediction error
from d3rlpy.metrics import EnvironmentEvaluator # To track online performance (requires env)
from d3rlpy.preprocessing import StandardRewardScaler # Import reward scaler
from d3rlpy.preprocessing import MinMaxRewardScaler # Import MinMaxRewardScaler
# TDErrorEvaluator is not suitable for DT
# from d3rlpy.metrics import TDErrorEvaluator
# --- Add Logger Imports ---
from d3rlpy.logging import TensorboardAdapterFactory, FileAdapterFactory, CombineAdapterFactory
# --- End Logger Imports ---
import torch # Check PyTorch availability for GPU
import numpy as np # Import numpy

# --- Constants --- 
DEFAULT_DATASET_DIR = "data/dt_dataset" # New default directory
DEFAULT_DATASET_FILENAME = "routing_dataset.h5" # Fixed filename within dir
# DEFAULT_STATS_FILENAME = "action_reward_stats.npz" # Old name
# DEFAULT_STATS_FILENAME = "reward_timestep_stats.npz" # New name: Only reward and timestep stats
DEFAULT_STATS_FILENAME = "timestep_stats.npz" # New name: Only timestep stats
DEFAULT_LOGDIR_BASE = "d3rlpy_logs"
DEFAULT_EXPERIMENT_NAME = "DT_Initial_Train" # Changed experiment name
DEFAULT_EPOCHS = 10 # Start with a small number for initial test
# DT specific hyperparameters
DEFAULT_LEARNING_RATE = 1e-5 # Reduced substantially to prevent NaN loss
DEFAULT_CONTEXT_SIZE = 64 # Temporarily reduced for debugging sequence error
DEFAULT_BATCH_SIZE = 128 # Default batch size for DiscreteDT
DEFAULT_NUM_HEADS = 8 # Default heads
DEFAULT_NUM_LAYERS = 6 # Default layers
# Regularization parameters
DEFAULT_ATTN_DROPOUT = 0.3  # Increased from 0.1 default
DEFAULT_RESID_DROPOUT = 0.3  # Increased from 0.1 default
DEFAULT_EMBED_DROPOUT = 0.3  # Increased from 0.1 default
# Common hyperparameters
DEFAULT_USE_REWARD_SCALER = True # Default to using reward scaler as planned

# --- Argument Parsing --- 
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Decision Transformer agent on the routing dataset.")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR,
                        help=f"Directory containing dataset (.h5) and stats (.npz) files (default: {DEFAULT_DATASET_DIR})")
    parser.add_argument("--logdir", type=str, default=None,
                        help=f"Directory to save logs and model files (default: {DEFAULT_LOGDIR_BASE}/<experiment_name>)")
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME,
                        help=f"Name for the training run (default: {DEFAULT_EXPERIMENT_NAME})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--gpu", type=int, default=None, metavar='ID',
                        help="GPU ID to use (e.g., 0). If None, use CPU.")
    # --- Add DT Hyperparameters --- 
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
                        help=f"Base learning rate (default: {DEFAULT_LEARNING_RATE})")
    parser.add_argument("--context_size", type=int, default=DEFAULT_CONTEXT_SIZE,
                        help=f"Sequence context size (default: {DEFAULT_CONTEXT_SIZE})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                         help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--num_heads", type=int, default=DEFAULT_NUM_HEADS,
                         help=f"Number of attention heads (default: {DEFAULT_NUM_HEADS})")
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS,
                         help=f"Number of transformer layers (default: {DEFAULT_NUM_LAYERS})")
    # --- Common Hyperparameters --- 
    # parser.add_argument("--use_reward_scaler", action=argparse.BooleanOptionalAction, default=DEFAULT_USE_REWARD_SCALER,
    #                     help=f"Use reward scaler (MinMaxRewardScaler will be used) (default: {DEFAULT_USE_REWARD_SCALER})") # Removed: Rewards are always pre-scaled now
    # parser.add_argument("--action_stats_path", type=str, default="",
    #                     help=f"Path to action min/max statistics file (.npz) (default: \"\")") # Removed
    
    args = parser.parse_args()
    
    # Determine log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.logdir is None:
        args.logdir = os.path.join(DEFAULT_LOGDIR_BASE, f"{args.experiment_name}_{timestamp}")
    else:
        # If custom logdir provided, still append timestamp
        args.logdir = f"{args.logdir}_{timestamp}"
        
    # Overwrite output filename if needed for DT
    args.dt_dataset_filename = args.dataset_dir.replace(".h5", ".h5")

    return args

def load_dataset(dataset_path):
    """Helper function to load ReplayBuffer for DT"""
    print(f"Loading ReplayBuffer for DT from {dataset_path}...")
    try:
        # ReplayBuffer requires buffer argument on load
        with open(dataset_path, "rb") as f:
            replay_buffer = ReplayBuffer.load(f, buffer=InfiniteBuffer())
        print("Successfully loaded ReplayBuffer")
        n_transitions = replay_buffer.transition_count
        print(f"  Dataset contains {n_transitions} transitions across {len(replay_buffer.episodes)} episodes.")
        return replay_buffer, n_transitions # Return ReplayBuffer directly
        
    except FileNotFoundError:
        print(f"Error: ReplayBuffer file not found at {dataset_path}")
        print(f"Please generate it first using: python src/data_processing/create_dataset.py --algo_type dt --output_filename {os.path.basename(dataset_path)}")
        return None, 0
    except Exception as e:
        print(f"Error loading ReplayBuffer: {e}")
        return None, 0

class MixedActionWrapper:
    """Wrapper to handle mixed discrete/continuous action space for routing weights."""
    
    def __init__(self, dt_policy):
        self.dt_policy = dt_policy
        # Define discrete values for the first three weights
        self.discrete_values = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    
    def predict(self, x):
        # Get continuous predictions from DT
        actions = self.dt_policy.predict(x)
        
        # Discretize first three weights to nearest valid value
        for i in range(3):  # drc_weight, marker_weight, fixed_weight
            actions[:, i] = self.discretize(actions[:, i])
            
        # decay_weight (index 3) remains continuous
        return actions
    
    def discretize(self, values):
        """Map continuous values to nearest discrete value."""
        indices = np.abs(values.reshape(-1, 1) - self.discrete_values).argmin(axis=1)
        return self.discrete_values[indices]

# --- Main Function --- 
def main():
    args = parse_args()

    # --- Construct paths based on dataset_dir --- 
    dataset_file_path = os.path.join(args.dataset_dir, DEFAULT_DATASET_FILENAME)
    stats_file_path = os.path.join(args.dataset_dir, DEFAULT_STATS_FILENAME) # Use new stats filename

    print("--- Starting Decision Transformer Training --- ")
    print(f"Dataset Directory: {args.dataset_dir}")
    print(f"Dataset File: {dataset_file_path}")
    print(f"Stats File: {stats_file_path}")
    print(f"Log Directory: {args.logdir}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")
    
    # Default to GPU if available, otherwise CPU
    if torch.cuda.is_available():
        device = "cuda:0" 
    else:
        device = "cpu"
    print(f"Device: {device}")
    print(f"Hyperparameters: LR={args.learning_rate}, Context={args.context_size}, Batch={args.batch_size}")
    print(f"Transformer Arch: Heads={args.num_heads}, Layers={args.num_layers}")
    # print(f"Using Reward Scaler: {args.use_reward_scaler}") # Removed: Rewards are always pre-scaled

    d3rlpy.seed(args.seed)
    print(f"Set d3rlpy random seed to {args.seed}")

    # Load ReplayBuffer dataset specifically for DT
    dataset, n_transitions = load_dataset(dataset_file_path) # Use constructed path
    if dataset is None:
        return

    print("ReplayBuffer loaded successfully.")
    
    # Load action and max_timestep stats
    max_timestep = None # Initialize max_timestep
    print(f"Loading action/reward stats from: {stats_file_path}")
    try:
        stats = np.load(stats_file_path)
        # Reward stats are no longer loaded
        # reward_min = stats['reward_minimum'] # Removed
        # reward_max = stats['reward_maximum'] # Removed
        max_timestep = int(stats['max_timestep']) # Load and cast max_timestep
        # print(f"Loaded action stats: Min={action_min}, Max={action_max}") # Removed
        # print(f"Loaded reward stats: Min={reward_min:.4f}, Max={reward_max:.4f}") # Removed
        print(f"Loaded max_timestep: {max_timestep}")
        # action_scaler = MinMaxActionScaler(minimum=action_min, maximum=action_max) # Removed
        # print("MinMaxActionScaler configured.") # Removed
    except FileNotFoundError:
        print(f"Warning: Timestep stats file not found at {stats_file_path}. Cannot get max_timestep.") # Updated print
        print("Exiting due to missing stats file.")
        return
    except KeyError as e:
        print(f"Warning: Key {e} not found in stats file {stats_file_path}. Cannot get max_timestep.") # Updated print
        print("Exiting due to incomplete stats file.")
        return
    except Exception as e:
        print(f"Error loading stats from {stats_file_path}: {e}")
        print("Exiting.")
        return

    # Check if max_timestep was loaded successfully
    if max_timestep is None:
        print("Error: max_timestep could not be determined from reward stats file.")
        print("Please ensure create_dataset.py ran successfully and saved 'max_timestep' in the .npz file.")
        return
    
    # Rewards are now pre-scaled in the dataset
    reward_scaler = None 
    print("Reward scaler disabled in config (rewards are pre-scaled in dataset).")

    # Configure Decision Transformer
    print("Configuring ContinuousDecisionTransformer...")
    dt_config = DecisionTransformerConfig(
        observation_scaler=None, # Assuming states are already scaled
        action_scaler=None, # Action scaler explicitly set to None
        reward_scaler=None, # MUST be None as rewards are pre-scaled
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        context_size=args.context_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        attn_dropout=DEFAULT_ATTN_DROPOUT,  # Higher dropout for stability
        resid_dropout=DEFAULT_RESID_DROPOUT,  # Higher dropout for stability
        embed_dropout=DEFAULT_EMBED_DROPOUT,  # Higher dropout for stability
        max_timestep=max_timestep, # CORRECT - Use value from stats file
        optim_factory=GPTAdamWFactory(clip_grad_norm=0.1, weight_decay=1e-4),  # More aggressive gradient clipping + weight decay
        compile_graph=False # Disable graph compilation to fix export error
    )
    
    dt = dt_config.create(device=device)

    # Build and train model
    print("Building model with dataset structure...")
    try:
        # Build with ReplayBuffer
        dt.build_with_dataset(dataset)
        print("Model built successfully.")
        
        print(f"Starting training for {args.epochs} epochs...")
        # Calculate n_steps based on epochs and a chosen steps_per_epoch
        n_steps_per_epoch = 1000 # Or adjust based on dataset size/preference
        n_steps = args.epochs * n_steps_per_epoch

        # Define evaluation parameters for DT fit method
        # Note: eval_env must be a gym.Env instance for evaluation/return metric
        eval_env = None # Placeholder - provide an actual environment instance if available
        eval_target_return = 0.0 # Placeholder - adjust based on expected returns
        eval_action_sampler = IdentityTransformerActionSampler()

        # --- Configure Logging --- 
        # The experiment subdirectory will be created automatically based on experiment_name
        # logger_adapter_factory = TensorboardAdapterFactory(root_dir=DEFAULT_LOGDIR_BASE) # Old: Only TensorBoard
        # Combine TensorBoard and CSV/model file logging
        logger_adapter_factory = CombineAdapterFactory([
            TensorboardAdapterFactory(root_dir=DEFAULT_LOGDIR_BASE), 
            FileAdapterFactory(root_dir=DEFAULT_LOGDIR_BASE) # Add FileAdapter for CSVs/models
        ])
        print(f"Tensorboard and CSV logs will be saved under {DEFAULT_LOGDIR_BASE}/{args.experiment_name}")
        # --- End Configure Logging ---

        dt.fit(
            dataset, # Pass the ReplayBuffer
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            experiment_name=args.experiment_name, # Controls logging directory name
            # Use DT's specific evaluation args:
            eval_env=eval_env, # Pass the environment instance here
            eval_target_return=eval_target_return, # Target return for evaluation rollouts
            eval_action_sampler=eval_action_sampler, # How actions are sampled during eval
            # Removed incompatible 'evaluators' argument
            # Training loss components (state, action, return errors)
            # should be automatically logged to TensorBoard by the fit method
            # if returned by the algorithm's update step.
            logger_adapter=logger_adapter_factory, # Correct way to pass logger
            save_interval=5 # Save model every 5 epochs (increased from 1 to reduce frequency)
        )
        
        print("Training completed successfully.")
        
        # Save model and export policies
        # The logdir is determined by the experiment_name and root_dir in the logger adapter
        # IMPORTANT: Logger adapter saves to {DEFAULT_LOGDIR_BASE}/{args.experiment_name}
        # Without the timestamp that was added to args.logdir
        # So we need to adjust our path accordingly
        experiment_dir = os.path.join(DEFAULT_LOGDIR_BASE, args.experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        final_model_path = os.path.join(experiment_dir, "model_final.d3")
        dt.save_model(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        final_policy_pt_path = os.path.join(experiment_dir, "policy.pt")
        dt.save_policy(final_policy_pt_path)
        print(f"Policy exported to TorchScript: {final_policy_pt_path}")
        
        try:
            final_policy_onnx_path = os.path.join(experiment_dir, "policy.onnx")
            dt.save_policy(final_policy_onnx_path)
            print(f"Policy exported to ONNX: {final_policy_onnx_path}")
            
            # Create and save wrapped policy
            wrapped_policy = MixedActionWrapper(dt)
            print("Created MixedActionWrapper for discrete/continuous action handling")
            
        except Exception as e:
            print(f"ONNX export failed (this is optional): {e}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        # Print traceback for unexpected errors
        import traceback
        traceback.print_exc()
        return

    print("\n--- Decision Transformer Training Script Finished ---")

if __name__ == "__main__":
    main() 