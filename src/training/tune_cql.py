'''Optuna hyperparameter tuning script for CQL.

Based on train_cql.py, but structured for Optuna optimization.
'''
import argparse
import os
import json
import d3rlpy # Import top-level d3rlpy for seed setting
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer # Keep ReplayBuffer components
from d3rlpy.algos import CQLConfig
from d3rlpy.optimizers import AdamFactory # Try importing directly from optimizers
from d3rlpy.metrics import InitialStateValueEstimationEvaluator
from d3rlpy.preprocessing import MinMaxActionScaler # Import action scaler
from d3rlpy.metrics import TDErrorEvaluator
from d3rlpy.metrics import ContinuousActionDiffEvaluator
from d3rlpy.metrics import AverageValueEstimationEvaluator
# --- Add Soft OPC Evaluator ---
try:
    from d3rlpy.metrics import SoftOPCEvaluator # For newer d3rlpy versions
except ImportError:
    try:
        from d3rlpy.metrics.ope import SoftOfflinePolicyEvaluation as SoftOPCEvaluator # For older versions
    except ImportError:
        print("Warning: SoftOPCEvaluator not available. Continuing without it.")
        SoftOPCEvaluator = None
# --- End Add Soft OPC Evaluator ---
# --- Add Logger Imports ---
from d3rlpy.logging import TensorboardAdapterFactory, FileAdapterFactory, CombineAdapterFactory
# --- End Logger Imports ---
# --- Add Optuna Import ---
import optuna
# --- End Optuna Import ---
import torch # Check PyTorch availability for GPU
import numpy as np # Import numpy
from datetime import datetime # Add datetime import
from typing import Optional
# --- Import EncoderFactory --- 
from d3rlpy.models.encoders import VectorEncoderFactory
# --- End EncoderFactory Import --- 
# --- Import Q-Function Factory --- 
# Try QRQFunctionFactory as an alternative distributional method
from d3rlpy.models.q_functions import QRQFunctionFactory 
# from d3rlpy.models.q_functions import IQNQFunctionFactory # Keep commented out for now
# --- End Q-Function Factory Import --- 

# --- Constants (Copied/adapted from train_cql.py) ---
DEFAULT_DATASET_DIR = "data/cql_dataset" # Default directory for CQL data
DEFAULT_DATASET_FILENAME = "routing_dataset.h5" # Fixed filename within dir
DEFAULT_STATS_FILENAME = "action_reward_stats.npz" # Fixed filename within dir
DEFAULT_LOGDIR_BASE = "optuna_logs/cql" # Separate log dir for Optuna runs
DEFAULT_EPOCHS = 10 # Default epochs PER TRIAL - keep relatively low for tuning
DEFAULT_N_TRIALS = 50 # Default number of Optuna trials
DEFAULT_GRAD_CLIP = 10.0 # Default gradient clipping norm (keep fixed for now)

# --- Objective Function for Optuna ---
def objective(trial: optuna.Trial, dataset_dir: str, logdir_base: str, base_experiment_name: str, fixed_epochs: int, gpu_id: Optional[int], seed: int) -> float:
    """Runs a single CQL training trial with Optuna suggested hyperparameters."""

    trial_num = trial.number
    print(f"\n--- Starting Optuna Trial {trial_num} ---")

    # --- Hyperparameter Suggestion ---
    actor_lr = trial.suggest_float("actor_lr", 1e-6, 1e-3, log=True)
    critic_lr = trial.suggest_float("critic_lr", 1e-6, 1e-4, log=True) # Reduced upper bound
    conservative_weight = trial.suggest_float("conservative_weight", 1.0, 20.0, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    initial_temperature = trial.suggest_float("initial_temperature", 0.1, 5.0)
    temp_lr_choice = trial.suggest_categorical("temp_lr_choice", ["tune", "fixed"])
    temp_learning_rate = 0.0 if temp_lr_choice == "fixed" else trial.suggest_float("temp_learning_rate", 1e-5, 1e-3, log=True)
    # Add tau suggestion
    tau = trial.suggest_float("tau", 1e-4, 1e-1, log=True)

    print(f"  Trial {trial_num} Parameters:")
    print(f"    actor_lr: {actor_lr:.2e}")
    print(f"    critic_lr: {critic_lr:.2e} (New Range: [1e-6, 1e-4])")
    print(f"    conservative_weight: {conservative_weight:.3f}")
    print(f"    batch_size: {batch_size}")
    print(f"    initial_temperature: {initial_temperature:.3f}")
    print(f"    temp_learning_rate: {temp_learning_rate:.2e} (Choice: {temp_lr_choice})")
    print(f"    tau: {tau:.2e}") # Print new parameter
    print(f"    epochs: {fixed_epochs}")

    # --- Setup ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{base_experiment_name}_trial_{trial_num}_{timestamp}" # Unique name per trial
    logdir = os.path.join(logdir_base, experiment_name)
    os.makedirs(logdir, exist_ok=True)

    # Seed is set globally in main, no need to seed per trial here
    # d3rlpy.seed(seed + trial_num)

    device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None and torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # --- Load data ---
    dataset_file_path = os.path.join(dataset_dir, DEFAULT_DATASET_FILENAME)
    stats_file_path = os.path.join(dataset_dir, DEFAULT_STATS_FILENAME)

    print(f"  Loading dataset from {dataset_file_path}...")
    try:
        with open(dataset_file_path, "rb") as f:
            replay_buffer = ReplayBuffer.load(f, buffer=InfiniteBuffer())
        print("  Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_file_path}")
        raise optuna.exceptions.TrialPruned(f"Dataset not found at {dataset_file_path}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise optuna.exceptions.TrialPruned(f"Failed to load dataset: {e}")

    # --- Setup Scalers ---
    action_scaler = None
    try:
        print(f"  Loading action stats from: {stats_file_path}")
        stats = np.load(stats_file_path)
        min_vals = stats['action_minimum']
        max_vals = stats['action_maximum']
        print(f"  Loaded action stats: Min={min_vals}, Max={max_vals}")
        if np.any(min_vals == max_vals):
            print("  Warning: Adjusting identical min/max action values.")
            max_vals[min_vals == max_vals] += 1e-6
        action_scaler = MinMaxActionScaler(minimum=min_vals, maximum=max_vals)
        print("  MinMaxActionScaler configured.")
    except Exception as e:
        print(f"  Warning: Failed to load or configure action scaler from {stats_file_path}: {e}. Proceeding without action scaling.")
        action_scaler = None

    observation_scaler = None
    reward_scaler = None

    # --- Configure Critic Network Architecture ---
    critic_encoder_factory = VectorEncoderFactory(hidden_units=[256, 256, 256]) 
    print(f"    Using Critic Encoder Factory: {critic_encoder_factory}")
    
    # --- Configure Q-Function Factory --- 
    # Use Quantile Regression Q-Function Factory
    q_func_factory = QRQFunctionFactory() 
    print(f"    Using Q Function Factory: {q_func_factory}") # Updated print

    # --- Configure Algorithm ---
    actor_optim_factory = AdamFactory(clip_grad_norm=DEFAULT_GRAD_CLIP) # Corrected parameter name
    critic_optim_factory = AdamFactory(clip_grad_norm=DEFAULT_GRAD_CLIP) # Corrected parameter name
    temp_optim_factory = AdamFactory() # No clipping needed/supported here
    alpha_optim_factory = AdamFactory() # No clipping needed/supported here

    cql_config = CQLConfig(
        # --- Pass the Q function factory --- 
        q_func_factory=q_func_factory, 
        # --- End Pass Q function factory --- 
        actor_learning_rate=actor_lr,
        critic_learning_rate=critic_lr,
        temp_learning_rate=temp_learning_rate,
        alpha_learning_rate=1e-4, # Keep default
        actor_optim_factory=actor_optim_factory,
        critic_optim_factory=critic_optim_factory,
        temp_optim_factory=temp_optim_factory,
        alpha_optim_factory=alpha_optim_factory,
        conservative_weight=conservative_weight,
        initial_temperature=initial_temperature,
        batch_size=batch_size,
        tau=tau, # Add tuned tau
        critic_encoder_factory=critic_encoder_factory,
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=None, # rewards are pre-scaled in dataset
        compile_graph=False  # Keep disabled
    )
    cql = cql_config.create(device=device.type)

    # --- Setup Evaluators ---
    evaluators = {
        'td_error': TDErrorEvaluator(),
        'initial_state_value': InitialStateValueEstimationEvaluator(), # Our optimization target
        'action_diff': ContinuousActionDiffEvaluator(),
        'avg_value': AverageValueEstimationEvaluator()
    }
    
    # Add Soft OPC Evaluator if available
    if SoftOPCEvaluator is not None:
        try:
            # Configure with a return threshold (e.g., 0.0)
            # Episodes with reward >= threshold are considered 'successful'
            evaluators['soft_ope'] = SoftOPCEvaluator(return_threshold=0.0)
            print("  Added SoftOPCEvaluator to metrics (threshold=0.0).")
        except Exception as e:
            print(f"  Warning: Failed to initialize SoftOPCEvaluator: {e}")

    # --- Configure Logging for this Trial ---
    # Loggers write directly into the trial-specific logdir
    logger_adapter_factory = CombineAdapterFactory([
        TensorboardAdapterFactory(root_dir=logdir), # Use trial-specific logdir as root
        FileAdapterFactory(root_dir=logdir)       # Use trial-specific logdir as root
    ])
    print(f"  Logging for trial {trial_num} configured at: {logdir}") # Path is correct

    # --- Train Agent ---
    print(f"  Starting training for trial {trial_num}...")
    final_metrics = {}
    try:
        cql.fit(
            replay_buffer,
            n_steps=fixed_epochs * 1000, # Use fixed steps based on epochs
            n_steps_per_epoch=1000,
            # experiment_name should NOT be passed here when using CombineAdapterFactory with specific root_dirs
            evaluators=evaluators,
            logger_adapter=logger_adapter_factory,
            save_interval=fixed_epochs, # Save logs/model at the end of the last epoch
            with_timestamp=False # Timestamp already in logdir path
        )
        print(f"  Training finished for trial {trial_num}.")

        # --- Retrieve Final Metric ---
        # Check multiple possible locations for the metric file
        # 1. Directly in trial directory
        metric_file = os.path.join(logdir, "initial_state_value.csv")
        
        # 2. Inside algorithm subdirectory (default is algorithm name 'CQL')
        if not os.path.exists(metric_file):
            algo_subdir_metric_file = os.path.join(logdir, "CQL", "initial_state_value.csv")
            if os.path.exists(algo_subdir_metric_file):
                metric_file = algo_subdir_metric_file
                print(f"  Found metric file in algorithm subdirectory: {metric_file}")
            else:
                print(f"  Warning: Metric file not found at {metric_file} or {algo_subdir_metric_file}")
                raise optuna.exceptions.TrialPruned("Metric file not found.")

        try:
            with open(metric_file, 'r') as f:
                lines = [line for line in f if line.strip() and not line.startswith('step')]
                if not lines:
                    raise ValueError("Metric file is empty or contains only headers.")
                last_line = lines[-1]
            metric_value = float(last_line.strip().split(',')[-1])
            print(f"  Trial {trial_num} final initial_state_value: {metric_value:.4f}")
            final_metrics['initial_state_value'] = metric_value
        except Exception as e:
            print(f"  Warning: Could not read final metric from {metric_file}: {e}")
            raise optuna.exceptions.TrialPruned("Failed to read final metric.")

    except Exception as e:
        print(f"Error during training for trial {trial_num}: {e}")
        import traceback
        traceback.print_exc()
        raise optuna.exceptions.TrialPruned(f"Training failed: {e}")

    # --- Return the value to be maximized by Optuna ---
    return final_metrics.get('initial_state_value', -float('inf'))


# --- Main Optuna Orchestration ---
def main():
    parser = argparse.ArgumentParser(description="Tune CQL hyperparameters using Optuna.")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR,
                        help=f"Directory containing dataset and stats files (default: {DEFAULT_DATASET_DIR})")
    parser.add_argument("--logdir_base", type=str, default=DEFAULT_LOGDIR_BASE,
                        help=f"Base directory to save Optuna trial logs (default: {DEFAULT_LOGDIR_BASE})")
    parser.add_argument("--experiment_name", type=str, default="CQL_Optuna_Tune",
                        help="Base name for Optuna experiments (default: CQL_Optuna_Tune)")
    parser.add_argument("--epochs_per_trial", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs for each Optuna trial (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--n_trials", type=int, default=DEFAULT_N_TRIALS,
                        help=f"Number of Optuna trials to run (default: {DEFAULT_N_TRIALS})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Optuna study AND d3rlpy")
    parser.add_argument("--gpu", type=int, default=None, metavar='ID',
                        help="GPU ID to use (e.g., 0). If None, use CPU.")

    args = parser.parse_args()

    print("--- Starting Optuna Hyperparameter Tuning for CQL ---")
    print(f"Dataset Directory: {args.dataset_dir}")
    print(f"Log Directory Base: {args.logdir_base}")
    print(f"Experiment Name: {args.experiment_name}")
    print(f"Epochs per Trial: {args.epochs_per_trial}")
    print(f"Number of Trials: {args.n_trials}")
    print(f"Seed: {args.seed}")
    print(f"GPU ID: {args.gpu}")

    # Set d3rlpy seed globally before starting study
    d3rlpy.seed(args.seed)
    print(f"Set global d3rlpy seed to {args.seed}")

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed), # Use same seed for sampler
    )

    # Define the objective function with fixed arguments using lambda
    objective_func = lambda trial: objective(
        trial=trial,
        dataset_dir=args.dataset_dir,
        logdir_base=args.logdir_base,
        base_experiment_name=args.experiment_name,
        fixed_epochs=args.epochs_per_trial,
        gpu_id=args.gpu,
        seed=args.seed # Pass seed for potential future use inside objective if needed, though not currently used
    )

    # Run optimization
    try:
        study.optimize(objective_func, n_trials=args.n_trials, timeout=None)
    except KeyboardInterrupt:
        print("\nOptimization stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred during optimization: {e}")
        import traceback
        traceback.print_exc()

    # --- Print Results ---
    print("\n--- Optuna Tuning Finished ---")
    try:
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        fail_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        print(f"Study statistics: ")
        print(f"  Number of finished trials: {len(study.trials)}")
        print(f"  Number of pruned trials: {len(pruned_trials)}")
        print(f"  Number of complete trials: {len(complete_trials)}")
        print(f"  Number of fail trials: {len(fail_trials)}")

        if complete_trials:
            print("\nBest trial:")
            best_trial = study.best_trial
            print(f"  Value (Initial State Value): {best_trial.value:.5f}")
            print("  Params: ")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")
        else:
             print("\nNo trials completed successfully. Cannot show best trial.")

    except Exception as e:
         print(f"Error displaying results: {e}")


if __name__ == "__main__":
    main() 