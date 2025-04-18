#!/usr/bin/env python3

import argparse
import sys
import os
import json
import numpy as np
import joblib

try:
    import d3rlpy
    from d3rlpy.algos import CQLConfig # Explicitly import config if needed
except ImportError:
    sys.stderr.write("Error: d3rlpy package not found. Please install it: pip install d3rlpy\n")
    sys.exit(1)

# --- Constants --- 
OBSERVATION_DIM = 11 # drv, wireLength, max_box_drv, average_box_drv, num_violating, pin_count, net_count, iteration, drv_lag_1, drv_lag_2, drv_lag_3
ACTION_DIM = 4      # drc_weight, marker_weight, fixed_weight, decay_weight
# --- Hardcoded Paths --- 
DEFAULT_MODEL_PATH = "/home/pseudo/expOpenRoad/OpenROAD-flow-scripts/tools/OpenROAD/src/drt/src/dr/scripts/rl_model/model_final.d3"
DEFAULT_SCALER_PATH = "/home/pseudo/expOpenRoad/OpenROAD-flow-scripts/tools/OpenROAD/src/drt/src/dr/scripts/rl_model/state_scaler.pkl"

def parse_args():
    parser = argparse.ArgumentParser(description="Predict routing weights using trained RL model")
    # Remove model and scaler path arguments
    # parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, 
    #                   help="Path to the trained d3rlpy model parameters (.pt format, even if named .d3)")
    # parser.add_argument("--scaler_path", type=str, default=DEFAULT_SCALER_PATH,
    #                   help="Path to the saved state scaler (.pkl file)")
    
    # Required state features as positional arguments in the exact order needed
    parser.add_argument("drv", type=float, help="Current DRV (from iteration T-1)")
    parser.add_argument("wireLength", type=float, help="Current wire length (from iteration T-1)")
    # parser.add_argument("total_box_drv", type=float, help="Sum of box_drv across all boxes (from iteration T-1)") # Commented out
    parser.add_argument("max_box_drv", type=float, help="Maximum box_drv observed (from iteration T-1)")
    parser.add_argument("average_box_drv", type=float, help="Average box_drv observed (from iteration T-1)") # Added average
    parser.add_argument("num_violating", type=float, help="Count of boxes with box_drv > 0 (from iteration T-1)")
    parser.add_argument("pin_count", type=float, help="Total pin count for the design")
    parser.add_argument("net_count", type=float, help="Total net count for the design")
    parser.add_argument("iteration", type=float, help="Current iteration number (T)")
    parser.add_argument("drv_lag_1", type=float, help="DRV from iteration T-2, use -1 if not available")
    parser.add_argument("drv_lag_2", type=float, help="DRV from iteration T-3, use -1 if not available")
    parser.add_argument("drv_lag_3", type=float, help="DRV from iteration T-4, use -1 if not available")
    
    parser.add_argument("--output_format", type=str, choices=["json", "csv"], default="json",
                      help="Output format for the predicted weights (default: json)")
    return parser.parse_args()

def load_model(model_path):
    """Load the trained RL model parameters from the specified path."""
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            sys.stderr.write(f"Error: Model parameter file not found at {model_path}\n")
            sys.stderr.write(f"Current working directory: {os.getcwd()}\n")
            sys.exit(1)
            
        # Load model parameters (.pt format) using the appropriate d3rlpy method
        sys.stderr.write(f"Loading model parameters from {model_path} (saved via save_model)...\n")
        sys.stderr.write(f"d3rlpy version: {d3rlpy.__version__}\n")
        
        # 1. Setup algorithm manually (using default config for inference is usually okay)
        #    If specific hyperparameters affecting network structure were changed during training, 
        #    those might need to be reflected here.
        config = CQLConfig()
        # Set device to CPU for broader compatibility during inference
        cql_algo = config.create(device='cpu') 

        # 2. Build the model structure manually with observation shape and action size
        #    This is necessary before loading parameters.
        sys.stderr.write(f"Building model structure with observation_shape=({OBSERVATION_DIM},) and action_size={ACTION_DIM}...\n")
        cql_algo.create_impl(observation_shape=(OBSERVATION_DIM,), action_size=ACTION_DIM)
        
        # 3. Load the saved parameters into the built structure
        sys.stderr.write(f"Loading parameters using load_model...\n")
        cql_algo.load_model(model_path)
        
        sys.stderr.write(f"Model parameters loaded successfully.\n")
        return cql_algo
        
    except Exception as e:
        err_msg = str(e)
        sys.stderr.write(f"Error loading model parameters from {model_path}: {err_msg}\n")
        # Add specific hints if possible
        if "size mismatch" in err_msg:
             sys.stderr.write("Hint: Check if OBSERVATION_DIM or ACTION_DIM in the script match the trained model.\n")
        sys.exit(1)

def load_scaler(scaler_path):
    """Load the saved state scaler from the specified path."""
    try:
        # Check if file exists
        if not os.path.exists(scaler_path):
            sys.stderr.write(f"Error: Scaler file not found at {scaler_path}\n")
            sys.stderr.write(f"Current working directory: {os.getcwd()}\n")
            sys.exit(1)
            
        sys.stderr.write(f"Loading scaler from {scaler_path}\n")
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        sys.stderr.write(f"Error loading scaler from {scaler_path}: {e}\n")
        sys.exit(1)

def predict_weights(model, scaler, state_features):
    """Normalize state features and predict weights using the trained model."""
    # Convert state features to numpy array and reshape for batch dimension
    # Ensure observation dim matches expected
    if len(state_features) != OBSERVATION_DIM:
         sys.stderr.write(f"Error: Expected {OBSERVATION_DIM} state features, but got {len(state_features)}\\n")
         sys.stderr.write(f"Received features ({len(state_features)}): {state_features}\\n")
         sys.exit(1)
         
    state = np.array(state_features, dtype=np.float32).reshape(1, -1)
    
    # Log input state for debugging
    sys.stderr.write(f"Input state: {state}\n")
    
    # Apply scaler to normalize the state
    try:
        normalized_state = scaler.transform(state)
        sys.stderr.write(f"Normalized state: {normalized_state}\n")
    except Exception as e:
        sys.stderr.write(f"Error applying scaler: {e}\n")
        sys.exit(1)
    
    # Predict weights using the model
    try:
        # Use predict method of the loaded algorithm object
        # If the model was trained with an action_scaler, predict() automatically denormalizes.
        weights = model.predict(normalized_state)[0]
        sys.stderr.write(f"Denormalized weights (from model.predict): {weights}\n")
        
        # Ensure we always have 4 weights (matching ACTION_DIM)
        if len(weights) != ACTION_DIM:
            # This shouldn't happen if model loading is correct, treat as error.
            sys.stderr.write(f"Error: Expected {ACTION_DIM} weights but got {len(weights)}\n")
            sys.exit(1)
            
        # Apply domain-specific constraints to weights AFTER denormalization
        # These ensure the weights are usable by the downstream routing tool
        weights[0] = max(1.0, weights[0])  # drc_weight >= 1.0
        weights[1] = max(0.0, weights[1])  # marker_weight >= 0.0
        weights[2] = max(1.0, weights[2])  # fixed_weight >= 1.0
        weights[3] = max(0.9, min(1.0, weights[3]))  # 0.9 <= decay_weight <= 1.0
        
        sys.stderr.write(f"Constrained weights: {weights}\n")
        return weights
    except Exception as e:
        sys.stderr.write(f"Error predicting weights: {e}\n")
        # Return default weights in case of error
        return np.array([1.0, 1.0, 1.0, 0.95])

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Extract state features in the correct order
    state_features = [
        args.drv,
        args.wireLength,
        # args.total_box_drv, # Commented out
        args.max_box_drv,
        args.average_box_drv, # Added average
        args.num_violating,
        args.pin_count,
        args.net_count,
        args.iteration,
        args.drv_lag_1,
        args.drv_lag_2,
        args.drv_lag_3
    ]
    
    try:
        # Load model parameters and scaler using hardcoded paths
        model_algo = load_model(DEFAULT_MODEL_PATH)
        scaler = load_scaler(DEFAULT_SCALER_PATH)
        
        # Predict weights
        weights = predict_weights(model_algo, scaler, state_features)
        
        # Output the predicted weights in the specified format
        weight_names = ["drc_weight", "marker_weight", "fixed_weight", "decay_weight"]
        if args.output_format == "json":
            weight_dict = {name: float(value) for name, value in zip(weight_names, weights)}
            print(json.dumps(weight_dict))
        else:  # CSV format
            print(",".join([str(float(w)) for w in weights]))
            
    except Exception as e:
        sys.stderr.write(f"Unexpected error in main: {e}\n")
        # Output default weights in case of error
        if args.output_format == "json":
            print(json.dumps({
                "drc_weight": 1.0,
                "marker_weight": 1.0,
                "fixed_weight": 1.0,
                "decay_weight": 0.95
            }))
        else:
            print("1.0,1.0,1.0,0.95")
        sys.exit(1)

if __name__ == "__main__":
    main() 