import argparse
import os
import d3rlpy # Import top-level d3rlpy for seed setting
from d3rlpy.dataset import MDPDataset
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer # Keep ReplayBuffer components
from d3rlpy.algos import CQLConfig
from d3rlpy.optimizers import AdamFactory # Try importing directly from optimizers
from d3rlpy.metrics import InitialStateValueEstimationEvaluator
from d3rlpy.preprocessing import StandardRewardScaler # Import reward scaler
# from d3rlpy.metrics import SoftOfflinePolicyEvaluationEvaluator
# from d3rlpy.metrics.ope import SoftOfflinePolicyEvaluation # Try importing from ope submodule
from d3rlpy.metrics import TDErrorEvaluator
from d3rlpy.metrics import EnvironmentEvaluator # If we have a simulator later
import torch # Check PyTorch availability for GPU

# --- Constants ---
DEFAULT_DATASET_PATH = "data/routing_dataset.h5"
DEFAULT_LOGDIR_BASE = "d3rlpy_logs"
DEFAULT_EXPERIMENT_NAME = "CQL_Initial_Train"
DEFAULT_EPOCHS = 10 # Start with a small number for initial test
# Lowered default LRs based on instability
DEFAULT_ACTOR_LR = 1e-5 
DEFAULT_CRITIC_LR = 1e-5
# Reset conservative weight to d3rlpy default, tune via CLI
DEFAULT_CONSERVATIVE_WEIGHT = 5.0 
DEFAULT_BATCH_SIZE = 256 # Common batch size
DEFAULT_USE_REWARD_SCALER = True # Default to using reward scaler as planned
DEFAULT_GRAD_CLIP = 10.0 # Default gradient clipping norm
DEFAULT_INITIAL_TEMP = 1.0
DEFAULT_TEMP_LR = 1e-4 # Default temperature learning rate from CQLConfig

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a CQL agent on the routing dataset.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH,
                        help=f"Path to the MDPDataset HDF5 file (default: {DEFAULT_DATASET_PATH})")
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
    parser.add_argument("--use_reward_scaler", action=argparse.BooleanOptionalAction, default=DEFAULT_USE_REWARD_SCALER,
                        help=f"Use StandardScaler for rewards (default: {DEFAULT_USE_REWARD_SCALER})")
    parser.add_argument("--grad_clip", type=float, default=DEFAULT_GRAD_CLIP,
                        help=f"Gradient clipping norm (applied internally, default: {DEFAULT_GRAD_CLIP})") # Note: Clipping might need specific setup
    parser.add_argument("--initial_temperature", type=float, default=DEFAULT_INITIAL_TEMP,
                         help=f"Initial temperature for SAC entropy (default: {DEFAULT_INITIAL_TEMP})")
    parser.add_argument("--temp_learning_rate", type=float, default=DEFAULT_TEMP_LR,
                         help=f"Learning rate for temperature (default: {DEFAULT_TEMP_LR}, 0 to fix temperature)")
    # Add more arguments for CQL hyperparameters later (e.g., target update interval, n_critics)
    args = parser.parse_args()
    
    # Determine log directory
    if args.logdir is None:
        args.logdir = os.path.join(DEFAULT_LOGDIR_BASE, args.experiment_name)
        
    
    return args

# --- Main Function ---
def main():
    args = parse_args()

    print("--- Starting CQL Training --- ")
    print(f"Dataset: {args.dataset}")
    print(f"Log Directory: {args.logdir}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Hyperparameters: ActorLR={args.actor_lr}, CriticLR={args.critic_lr}, ConservWeight={args.conservative_weight}, Batch={args.batch_size}")
    print(f"Gradient Clipping Norm: {args.grad_clip}")
    print(f"Initial Temperature: {args.initial_temperature}, Temp LR: {args.temp_learning_rate}")
    print(f"Using Reward Scaler: {args.use_reward_scaler}")

    # --- Set Random Seed --- 
    d3rlpy.seed(args.seed)
    print(f"Set d3rlpy random seed to {args.seed}")

    # --- Load data directly into ReplayBuffer ---
    print(f"Loading dataset from {args.dataset} into ReplayBuffer...")
    try:
        with open(args.dataset, "rb") as f:
            # Specify buffer type during load
            replay_buffer = ReplayBuffer.load(f, buffer=InfiniteBuffer())
        print("Dataset loaded successfully into ReplayBuffer.")
        # Can optionally check replay_buffer.buffer here if needed
    except Exception as e:
        print(f"Error loading dataset from {args.dataset} into ReplayBuffer: {e}")
        return

    # --- Setup CQL Algorithm ---
    # Action scaler might be useful if actions have very different ranges, but weights are somewhat similar
    # Reward scaler can help stabilize training, especially if rewards have high variance
    observation_scaler = None # Scaler is applied during data prep
    action_scaler = None # Optional: MinMaxActionScaler() or StandardScaler()
    # Conditionally create reward scaler
    reward_scaler = StandardRewardScaler() if args.use_reward_scaler else None
    
    # Configure optimizers with potential clipping (fix: remove clip_norm from factory)
    actor_optim_factory = AdamFactory(weight_decay=0) # Removed clip_norm
    critic_optim_factory = AdamFactory(weight_decay=0) # Removed clip_norm
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
        reward_scaler=reward_scaler # Add reward scaler to config
        # Add other configured hyperparameters here
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cql = cql_config.create(device= device.type)

    # --- Setup Evaluators (Optional but Recommended) ---
    # These run periodically during training to provide insights
    evaluators = {
        'td_error': TDErrorEvaluator(),
        # 'soft_ope': SoftOfflinePolicyEvaluation(), # Temporarily removed due to import issues
        'initial_state_value': InitialStateValueEstimationEvaluator()
        # Add 'environment' evaluator if an environment simulator becomes available
    }
    
    # --- Train Agent ---
    print(f"\nStarting training for {args.epochs} epochs...")
    try:
        cql.fit(
            replay_buffer, # Pass the replay_buffer object
            n_steps=args.epochs * 1000, # d3rlpy uses steps, convert epochs approximately
            n_steps_per_epoch=1000,
            experiment_name=args.experiment_name,
            evaluators=evaluators,
            # tensorboard_dir="runs", # if you want tensorboard logs
            save_interval=10, # Save model every N epochs
            with_timestamp=False # Don't add timestamp to logdir
        )
        print("Training finished.")
    except Exception as e:
        print(f"Error during training: {e}")
        # Potentially save partially trained model here if needed
        return

    # --- Save Final Model --- 
    # d3rlpy automatically saves the best model during fit if evaluators are used
    # But we can explicitly save the final one too
    final_model_path = os.path.join(args.logdir, "model_final.d3")
    try:
        cql.save_model(final_model_path)
        print(f"Final model saved to {final_model_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")

    print("\n--- CQL Training Script Finished ---")

if __name__ == "__main__":
    main() 