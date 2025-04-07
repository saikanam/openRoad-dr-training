import argparse
import os
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
from d3rlpy.metrics import TDErrorEvaluator # Use evaluator classes again
from d3rlpy.metrics import InitialStateValueEstimationEvaluator
# from d3rlpy.metrics.scorer import td_error_scorer # Remove scorer function import
# from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer # Remove scorer function import
# from d3rlpy.metrics.ope import SoftOfflinePolicyEvaluation # Still need to find the correct OPE import if desired
from d3rlpy.algos import CQL # Need to import the algorithm class to load

# --- Constants ---
DEFAULT_DATASET_PATH = "data/routing_dataset.h5"
DEFAULT_MODEL_PATH = "d3rlpy_logs/CQL_Initial_Train/model_final.d3"
DEFAULT_GPU = None

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CQL agent.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH,
                        help=f"Path to the dataset HDF5 file (default: {DEFAULT_DATASET_PATH})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Path to the trained model file (.d3) (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU, metavar='ID',
                        help="GPU ID to use for evaluation (e.g., 0). If None, use CPU.")
    # Add arguments for specific evaluation metrics or configurations if needed
    args = parser.parse_args()
    return args

# --- Main Function ---
def main():
    args = parse_args()

    print("--- Starting Agent Evaluation --- ")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"GPU: {args.gpu}")

    # Determine device
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # --- Load Dataset ---
    print(f"Loading dataset from {args.dataset}...")
    try:
        with open(args.dataset, "rb") as f:
            replay_buffer = ReplayBuffer.load(f, buffer=InfiniteBuffer())
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # --- Load Trained Model ---
    print(f"Loading trained model from {args.model}...")
    try:
        # Need to instantiate the same algorithm class used for training
        # We assume CQL was used based on the path, but ideally, training script
        # could save config, or model file contains type info d3rlpy can use.
        # For now, hardcode CQL.
        # Note: .load() automatically puts model on the specified device.
        cql = CQL.from_json(os.path.join(os.path.dirname(args.model), 'params.json'), device=device)
        cql.load_model(args.model) # Load the weights
        # Alternatively, if d3rlpy > v2.1, we might use load_learnable
        # cql = d3rlpy.load_learnable(args.model, device=device)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: params.json not found in the model directory {os.path.dirname(args.model)}.")
        print("Make sure the model was saved correctly and params.json exists.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Perform Evaluation ---
    print("\nPerforming evaluation using offline metrics...")

    # 1. TD Error
    print("\nCalculating TD Error...")
    td_evaluator = TDErrorEvaluator()
    # td_error = td_error_scorer(algo=cql, episodes=replay_buffer) # Use scorer function
    td_error = td_evaluator(algo=cql, dataset=replay_buffer) # Call evaluator object
    print(f"  TD Error: {td_error:.4f}")

    # 2. Initial State Value
    print("\nCalculating Initial State Value...")
    isv_evaluator = InitialStateValueEstimationEvaluator()
    # initial_state_value = initial_state_value_estimation_scorer(algo=cql, episodes=replay_buffer) # Use scorer function
    initial_state_value = isv_evaluator(algo=cql, dataset=replay_buffer) # Call evaluator object
    print(f"  Initial State Value: {initial_state_value:.4f}")

    # 3. Soft OPE (if available and desired)
    # print("\nCalculating Soft OPE (Placeholder - requires correct import)...")
    # try:
    #    from d3rlpy.metrics.??? import ??? # Find correct import
    #    soft_ope_evaluator = ???()
    #    soft_ope_value = soft_ope_evaluator.evaluate(cql, replay_buffer)
    #    print(f"  Soft OPE Value: {soft_ope_value:.4f}")
    # except ImportError:
    #    print("  Soft OPE evaluator not found/imported.")
    # except Exception as e:
    #    print(f"  Error during Soft OPE calculation: {e}")

    # Add other offline metrics (e.g., FQE if implemented) or environment evaluation later

    print("\n--- Agent Evaluation Finished ---")

if __name__ == "__main__":
    main() 