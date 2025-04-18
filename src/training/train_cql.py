import argparse
import os
import d3rlpy # Import top-level d3rlpy for seed setting
from d3rlpy.dataset import MDPDataset
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer # Keep ReplayBuffer components
from d3rlpy.algos import CQLConfig
from d3rlpy.optimizers import AdamFactory # Try importing directly from optimizers
from d3rlpy.metrics import InitialStateValueEstimationEvaluator
from d3rlpy.preprocessing import StandardRewardScaler # Import reward scaler
from d3rlpy.preprocessing import MinMaxActionScaler # Import action scaler
# from d3rlpy.metrics import SoftOfflinePolicyEvaluationEvaluator
# from d3rlpy.metrics.ope import SoftOfflinePolicyEvaluation # Try importing from ope submodule
from d3rlpy.metrics import TDErrorEvaluator
from d3rlpy.metrics import EnvironmentEvaluator # If we have a simulator later
# --- Add new evaluators ---
from d3rlpy.metrics import ContinuousActionDiffEvaluator
from d3rlpy.metrics import AverageValueEstimationEvaluator
from d3rlpy.metrics import SoftOPCEvaluator
# --- End Add new evaluators ---
# --- Add Logger Imports ---
from d3rlpy.logging import TensorboardAdapterFactory, FileAdapterFactory, CombineAdapterFactory
# --- End Logger Imports ---
import torch # Check PyTorch availability for GPU
import numpy as np # Import numpy
from datetime import datetime # Add datetime import

# --- Constants ---
# DEFAULT_DATASET_PATH = "data/routing_dataset.h5" # Removed
DEFAULT_DATASET_DIR = "data/cql_dataset" # New default directory for CQL data
DEFAULT_DATASET_FILENAME = "routing_dataset.h5" # Fixed filename within dir
DEFAULT_STATS_FILENAME = "action_reward_stats.npz" # Fixed filename within dir
DEFAULT_LOGDIR_BASE = "d3rlpy_logs"
DEFAULT_EXPERIMENT_NAME = "CQL_Long_Train_RewardScaled" # Updated name for experiment
DEFAULT_EPOCHS = 50 # Increased epochs for longer run
# Lowered default LRs based on instability
DEFAULT_ACTOR_LR = 1e-5
DEFAULT_CRITIC_LR = 1e-5
# Reset conservative weight to d3rlpy default, tune via CLI
DEFAULT_CONSERVATIVE_WEIGHT = 5.0
DEFAULT_BATCH_SIZE = 256 # Common batch size
# DEFAULT_USE_REWARD_SCALER = True # Default to using reward scaler as planned # Removed, now added directly
# DEFAULT_ACTION_STATS_PATH = "data/action_min_max.npz" # Removed
DEFAULT_GRAD_CLIP = 10.0 # Default gradient clipping norm
DEFAULT_INITIAL_TEMP = 1.0
DEFAULT_TEMP_LR = 1e-4 # Default temperature learning rate from CQLConfig

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a CQL agent on the routing dataset.")
    # parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH,
    #                     help=f"Path to the MDPDataset HDF5 file (default: {DEFAULT_DATASET_PATH})")
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
    # --- Add Hyperparameters ---
    parser.add_argument("--actor_lr", type=float, default=DEFAULT_ACTOR_LR,
                        help=f"Actor learning rate (default: {DEFAULT_ACTOR_LR})")
    parser.add_argument("--critic_lr", type=float, default=DEFAULT_CRITIC_LR,
                        help=f"Critic learning rate (default: {DEFAULT_CRITIC_LR})")
    parser.add_argument("--conservative_weight", type=float, default=DEFAULT_CONSERVATIVE_WEIGHT,
                         help=f"CQL conservative weight (default: {DEFAULT_CONSERVATIVE_WEIGHT})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                         help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})")
    # parser.add_argument("--use_reward_scaler", action=argparse.BooleanOptionalAction, default=DEFAULT_USE_REWARD_SCALER,
    #                     help=f"Use StandardScaler for rewards (default: {DEFAULT_USE_REWARD_SCALER})") # Removed, now added directly
    # parser.add_argument("--action_stats_path", type=str, default=DEFAULT_ACTION_STATS_PATH,
    #                     help=f"Path to action min/max statistics file (.npz) (default: {DEFAULT_ACTION_STATS_PATH})") # Removed
    parser.add_argument("--grad_clip", type=float, default=DEFAULT_GRAD_CLIP,
                        help=f"Gradient clipping norm (applied internally, default: {DEFAULT_GRAD_CLIP})") # Note: Clipping might need specific setup
    parser.add_argument("--initial_temperature", type=float, default=DEFAULT_INITIAL_TEMP,
                         help=f"Initial temperature for SAC entropy (default: {DEFAULT_INITIAL_TEMP})")
    parser.add_argument("--temp_learning_rate", type=float, default=DEFAULT_TEMP_LR,
                         help=f"Learning rate for temperature (default: {DEFAULT_TEMP_LR}, 0 to fix temperature)")
    # Add more arguments for CQL hyperparameters later (e.g., target update interval, n_critics)
    args = parser.parse_args()

    # Determine log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.logdir is None:
        args.logdir = os.path.join(DEFAULT_LOGDIR_BASE, f"{args.experiment_name}_{timestamp}")
    else:
        # If custom logdir provided, still append timestamp
        args.logdir = f"{args.logdir}_{timestamp}"

    return args

# --- Main Function ---
def main():
    args = parse_args()

    # --- Construct paths based on dataset_dir ---
    dataset_file_path = os.path.join(args.dataset_dir, DEFAULT_DATASET_FILENAME)
    stats_file_path = os.path.join(args.dataset_dir, DEFAULT_STATS_FILENAME)

    print("--- Starting CQL Training --- ")
    # print(f"Dataset: {args.dataset}") # Old
    print(f"Dataset Directory: {args.dataset_dir}")
    print(f"Dataset File: {dataset_file_path}")
    print(f"Stats File: {stats_file_path}")
    print(f"Log Directory: {args.logdir}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Hyperparameters: ActorLR={args.actor_lr}, CriticLR={args.critic_lr}, ConservWeight={args.conservative_weight}, Batch={args.batch_size}")
    print(f"Gradient Clipping Norm: {args.grad_clip}")
    print(f"Initial Temperature: {args.initial_temperature}, Temp LR: {args.temp_learning_rate}")
    # print(f"Action Stats Path: {args.action_stats_path}") # Old

    # --- Set Random Seed ---
    d3rlpy.seed(args.seed)
    print(f"Set d3rlpy random seed to {args.seed}")

    # --- Load data directly into ReplayBuffer ---
    # print(f"Loading dataset from {args.dataset} into ReplayBuffer...") # Old
    print(f"Loading dataset from {dataset_file_path} into ReplayBuffer...")
    try:
        # with open(args.dataset, "rb") as f: # Old
        with open(dataset_file_path, "rb") as f:
            # Specify buffer type during load
            replay_buffer = ReplayBuffer.load(f, buffer=InfiniteBuffer())
        print("Dataset loaded successfully into ReplayBuffer.")
        # Can optionally check replay_buffer.buffer here if needed
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_file_path}")
        print(f"Please generate it first using: python src/data_processing/create_dataset.py --algo_type cql --output_dir {args.dataset_dir} --output_filename {DEFAULT_DATASET_FILENAME}")
        return
    except Exception as e:
        # print(f"Error loading dataset from {args.dataset} into ReplayBuffer: {e}") # Old
        print(f"Error loading dataset from {dataset_file_path} into ReplayBuffer: {e}")
        return

    # --- Setup CQL Algorithm ---
    # Action scaler might be useful if actions have very different ranges, but weights are somewhat similar
    # Reward scaler can help stabilize training, especially if rewards have high variance
    observation_scaler = None # Scaler is applied during data prep

    # Load action stats and configure action scaler
    action_scaler = None
    # Update stats file path reference
    # stats_path = args.action_stats_path.replace("action_min_max.npz", "action_reward_stats.npz") # Old
    stats_path = stats_file_path # Use the constructed path
    print(f"Loading action stats from: {stats_path}")
    try:
        stats = np.load(stats_path)
        # Update keys for loading
        min_vals = stats['action_minimum']
        max_vals = stats['action_maximum']
        # reward_min = stats['reward_minimum'] # Load but not needed for scaler config
        # reward_max = stats['reward_maximum']
        # max_timestep = int(stats['max_timestep']) # Not needed for CQL config
        print(f"Loaded action stats: Min={min_vals}, Max={max_vals}")

        # Ensure min and max are distinct to avoid NaN/Inf issues
        if np.any(min_vals == max_vals):
            print("Warning: Min and Max values are identical for some actions. Adjusting slightly.")
            # Add a small epsilon to max where min == max
            max_vals[min_vals == max_vals] += 1e-6

        action_scaler = MinMaxActionScaler(minimum=min_vals, maximum=max_vals)
        print("MinMaxActionScaler configured.")
    except FileNotFoundError:
        print(f"Warning: Action stats file not found at {stats_path}. Action scaling will NOT be used.")
    except KeyError as e:
        print(f"Warning: Key {e} not found in stats file {stats_path}. Action scaling will NOT be used.")
    except Exception as e:
        print(f"Error loading action stats from {stats_path}: {e}. Action scaling will NOT be used.")

    # Rewards are pre-scaled, disable scaler in config
    reward_scaler = None
    print("Reward scaler disabled in config (rewards are pre-scaled in dataset).")

    # Configure optimizers with potential clipping
    actor_optim_factory = AdamFactory(clip_grad_norm=DEFAULT_GRAD_CLIP) # Corrected parameter name
    critic_optim_factory = AdamFactory(clip_grad_norm=DEFAULT_GRAD_CLIP) # Corrected parameter name
    temp_optim_factory = AdamFactory(weight_decay=0) # No clipping needed/supported here
    alpha_optim_factory = AdamFactory(weight_decay=0) # No clipping needed/supported here

    cql_config = CQLConfig(
        actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr,
        temp_learning_rate=args.temp_learning_rate,
        alpha_learning_rate=1e-4, # Keep alpha LR default for now
        actor_optim_factory=actor_optim_factory,
        critic_optim_factory=critic_optim_factory,
        temp_optim_factory=temp_optim_factory,
        alpha_optim_factory=alpha_optim_factory,
        conservative_weight=args.conservative_weight,
        initial_temperature=args.initial_temperature,
        batch_size=args.batch_size,
        observation_scaler=observation_scaler, 
        action_scaler=action_scaler,      
        reward_scaler=None, # REMOVED StandardRewardScaler - rewards are now pre-scaled in dataset
        compile_graph = False # Keep disabled
        # Add other configured hyperparameters here
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cql = cql_config.create(device= device.type)

    # --- Setup Evaluators (Optional but Recommended) ---
    # These run periodically during training to provide insights
    evaluators = {
        'td_error': TDErrorEvaluator(),
        'soft_ope': SoftOPCEvaluator(return_threshold=0.0), # Temporarily removed due to import issues
        'initial_state_value': InitialStateValueEstimationEvaluator(),
        # --- Add new evaluators to the dict ---
        'action_diff': ContinuousActionDiffEvaluator(),
        'avg_value': AverageValueEstimationEvaluator()
        # --- End Add new evaluators ---
        # Add 'environment' evaluator if an environment simulator becomes available
    }
    
    # --- Train Agent ---
    print(f"\nStarting training for {args.epochs} epochs...")
    try:
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

        cql.fit(
            replay_buffer, # Pass the replay_buffer object
            n_steps=args.epochs * 1000, # d3rlpy uses steps, convert epochs approximately
            n_steps_per_epoch=1000,
            experiment_name=args.experiment_name,
            evaluators=evaluators,
            logger_adapter=logger_adapter_factory, # Correct way to pass logger
            save_interval=10, # Save model every N epochs
            with_timestamp=False # Don't add timestamp to logdir
        )
        print("Training finished.")
    except Exception as e:
        print(f"Error during training: {e}")
        # Potentially save partially trained model here if needed
        # Print traceback for unexpected errors
        import traceback
        traceback.print_exc()
        return

    # --- Save Final Model --- 
    # d3rlpy automatically saves the best model during fit if logger_adapter is used
    # and save_interval is set. Manual saving might be redundant.
    # final_model_path = os.path.join(args.logdir, "model_final.d3") # Path needs adjustment
    # try:
    #     cql.save_model(final_model_path)
    #     print(f"Final model saved to {final_model_path}")
    # except Exception as e:
    #     print(f"Error saving final model: {e}")

    # --- Export Policy for Inference (TorchScript and ONNX) ---
    # Exporting the policy is often better for deployment than saving the whole model
    # final_policy_pt_path = os.path.join(args.logdir, "policy.pt") # Path needs adjustment
    # final_policy_onnx_path = os.path.join(args.logdir, "policy.onnx") # Path needs adjustment

    # print(f"\nExporting final policy to TorchScript: {final_policy_pt_path}...")
    # try:
    #     cql.save_policy(final_policy_pt_path)
    #     print("Policy exported successfully to TorchScript.")
    # except Exception as e:
    #     print(f"Error exporting policy to TorchScript: {e}")

    # print(f"\nExporting final policy to ONNX: {final_policy_onnx_path}...")
    # try:
    #     cql.save_policy(final_policy_onnx_path)
    #     print("Policy exported successfully to ONNX.")
    # except Exception as e:
    #     # ONNX export might fail if ops are not supported, e.g., with certain observation types
    #     print(f"Error exporting policy to ONNX: {e}")
    #     print("Note: ONNX export might require specific libraries or model architectures.")

    print("\n--- CQL Training Script Finished ---")

if __name__ == "__main__":
    main() 