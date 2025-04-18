#!/usr/bin/env python3

import argparse
import sys
import os
import json
import numpy as np
import joblib
import torch
import random # Import random for exploration

# --- D3RLPY Imports ---
try:
    import d3rlpy
    from d3rlpy.algos import DecisionTransformerConfig # Needed for loading
    from d3rlpy.preprocessing import MinMaxActionScaler # Potentially needed if action_scaler info isn't saved in .d3
except ImportError:
    sys.stderr.write("Error: d3rlpy package not found. Please install it: pip install d3rlpy\\n")
    sys.exit(1)

# --- Constants ---
# Ensure this matches the state features used during training/dataset creation
OBSERVATION_DIM = 19 # Original 11 + 8 static features
ACTION_DIM = 4      # drc_weight, marker_weight, fixed_weight, decay_weight

# --- Hardcoded Paths (Update these to your DT model/scaler paths) ---
# Point to the DT logs directory and the relevant files
# **** IMPORTANT: Update this path to your specific DT training log directory ****
# DEFAULT_DT_LOG_DIR = "d3rlpy_logs/DT_Initial_Train_20250413021145" # Updated path from previous run
# # DEFAULT_POLICY_PATH = os.path.join(DEFAULT_DT_LOG_DIR, "policy.pt")
# DEFAULT_MODEL_PATH = os.path.join(DEFAULT_DT_LOG_DIR, "model_final.d3") # Point to .d3 model
# # Scaler is saved in the 'data' directory by create_dataset.py
# DEFAULT_SCALER_PATH = "data/state_scaler.pkl"

DEFAULT_MODEL_PATH = "/home/pseudo/expOpenRoad/OpenROAD-flow-scripts/tools/OpenROAD/src/drt/src/dr/scripts/rl_model/dt/model_50000.d3"
DEFAULT_SCALER_PATH = "/home/pseudo/expOpenRoad/OpenROAD-flow-scripts/tools/OpenROAD/src/drt/src/dr/scripts/rl_model/dt/state_scaler.pkl"


class MixedActionWrapper:
    """Wrapper logic to handle mixed discrete/continuous action space for routing weights."""

    def __init__(self):
        # Define discrete values for the first three weights
        self.discrete_values = np.array([0.0, 0.5, 1.0, 2.0, 4.0], dtype=np.float32)

    def process_actions(self, actions):
        """Applies discretization to the first 3 actions in a batch or single action."""
        if actions.ndim == 1: # Single action prediction
            actions = actions.reshape(1, -1) # Add batch dimension
            single_pred = True
        else:
            single_pred = False

        processed_actions = actions.copy() # Avoid modifying input directly

        # Discretize first three weights to nearest valid value
        for i in range(3):  # drc_weight, marker_weight, fixed_weight
            # Calculate distances to discrete values for the current weight column
            current_weight_col = processed_actions[:, i].reshape(-1, 1)
            distances = np.abs(current_weight_col - self.discrete_values)
            # Find indices of the minimum distances
            indices = distances.argmin(axis=1)
            # Assign the corresponding discrete values
            processed_actions[:, i] = self.discrete_values[indices]

        # decay_weight (index 3) remains continuous

        if single_pred:
            return processed_actions.flatten() # Return as single array
        else:
            return processed_actions # Return batch

def parse_args():
    parser = argparse.ArgumentParser(description="Predict routing weights using trained Decision Transformer model")

    # Required state features as positional arguments
    parser.add_argument("drv", type=float, help="Current DRV (from iteration T-1)")
    parser.add_argument("wireLength", type=float, help="Current wire length (from iteration T-1)")
    parser.add_argument("max_box_drv", type=float, help="Maximum box_drv observed (from iteration T-1)")
    parser.add_argument("average_box_drv", type=float, help="Average box_drv observed (from iteration T-1)")
    parser.add_argument("num_violating", type=float, help="Count of boxes with box_drv > 0 (from iteration T-1)")
    parser.add_argument("pin_count", type=float, help="Total pin count for the design")
    parser.add_argument("net_count", type=float, help="Total net count for the design")
    parser.add_argument("iteration", type=float, help="Current iteration number (T)")
    parser.add_argument("drv_lag_1", type=float, help="DRV from iteration T-2, use -1 if not available")
    parser.add_argument("drv_lag_2", type=float, help="DRV from iteration T-3, use -1 if not available")
    parser.add_argument("drv_lag_3", type=float, help="DRV from iteration T-4, use -1 if not available")

    # --- ADDED: Static Design Features ---
    parser.add_argument("pin_density", type=float, help="Static: Pin density")
    parser.add_argument("num_macros", type=float, help="Static: Number of macros")
    parser.add_argument("component_density", type=float, help="Static: Component density")
    parser.add_argument("guide_density", type=float, help="Static: Guide density")
    parser.add_argument("avg_pins_per_net", type=float, help="Static: Average pins per net")
    parser.add_argument("net_density", type=float, help="Static: Net density")
    parser.add_argument("num_layers", type=float, help="Static: Number of layers")
    parser.add_argument("terminal_density", type=float, help="Static: Terminal density")
    # --- End Static Design Features ---

    # parser.add_argument("--policy_path", type=str, default=DEFAULT_POLICY_PATH,
    #                     help=f"Path to the exported DT policy (policy.pt) (default: {DEFAULT_POLICY_PATH})")
    parser.add_argument("--scaler_path", type=str, default=DEFAULT_SCALER_PATH,
                        help=f"Path to the saved state scaler (.pkl file) (default: {DEFAULT_SCALER_PATH})")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Path to the saved d3rlpy DT model (.d3) (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--output_format", type=str, choices=["json", "csv"], default="json",
                      help="Output format for the predicted weights (default: json)")
    # Target return is now hardcoded
    # parser.add_argument("--target_return", type=float, default=4.315487,
    #                   help="Target return-to-go for DT inference")
    return parser.parse_args()

def load_model_and_wrapper(model_path, target_return):
    """Load the full d3rlpy model and create a stateful wrapper."""
    try:
        if not os.path.exists(model_path):
            sys.stderr.write(f"Error: Model file not found at {model_path}\n")
            sys.exit(1)

        sys.stderr.write(f"Loading d3rlpy model from {model_path}...\n")
        # Determine device
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        sys.stderr.write(f"Using device: {device}\n")
            
        # Load the learnable object (model + config)
        dt_algo = d3rlpy.load_learnable(model_path, device=device)
        sys.stderr.write(f"Model loaded successfully.\n")
        
        # Create the stateful wrapper with the target return
        sys.stderr.write(f"Creating stateful wrapper with target_return={target_return:.6f}...\n")
        # Note: Assumes the default action sampler is appropriate
        wrapper = dt_algo.as_stateful_wrapper(target_return=target_return)
        sys.stderr.write(f"Stateful wrapper created successfully.\n")
        
        return wrapper, device # Return wrapper and device

    except Exception as e:
        sys.stderr.write(f"Error loading model or creating wrapper from {model_path}: {e}\n")
        sys.exit(1)

def load_scaler(scaler_path):
    """Load the saved state scaler."""
    try:
        if not os.path.exists(scaler_path):
            sys.stderr.write(f"Error: Scaler file not found at {scaler_path}\n")
            sys.exit(1)
        sys.stderr.write(f"Loading scaler from {scaler_path}\n")
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        sys.stderr.write(f"Error loading scaler from {scaler_path}: {e}\n")
        sys.exit(1)

def predict_weights(wrapper, device, scaler, state_features, action_wrapper, epsilon=0.1):
    """Normalize state, predict continuous actions (with exploration if stuck), discretize/round, constrain."""
    if len(state_features) != OBSERVATION_DIM:
         sys.stderr.write(f"Error: Expected {OBSERVATION_DIM} state features, but got {len(state_features)}\\n")
         sys.stderr.write(f"Received features ({len(state_features)}): {state_features}\\n")
         sys.exit(1)
         
    state = np.array(state_features, dtype=np.float32).reshape(1, -1)
    sys.stderr.write(f"Input state: {state}\n")

    # --- Epsilon-Greedy Exploration --- 
    # Check if stuck: current DRV > 0 and same as previous DRV?
    current_drv = state_features[0]
    prev_drv = state_features[8] # drv_lag_1 is at index 8
    is_stuck = (current_drv > 0 and current_drv == prev_drv)
    
    if is_stuck and random.random() < epsilon:
        sys.stderr.write(f"Stuck detected (drv={current_drv}) and exploration triggered (epsilon={epsilon})!\\n")
        # Generate exploratory weights based on expert bias
        # Ranges are indicative, adjust as needed
        weights = np.zeros(ACTION_DIM, dtype=np.float32)
        weights[0] = round(random.uniform(1.0, 40.0))  # drc_weight: Higher range
        weights[1] = round(random.uniform(0.0, 10.0))  # marker_weight: Mid range, slightly higher bias possible
        weights[2] = round(random.uniform(1.0, 30.0))  # fixed_weight: Lower range 
        weights[3] = random.uniform(0.9, 1.0)          # decay_weight: High range (no rounding needed)
        sys.stderr.write(f"Exploratory raw weights: {weights}\\n")
        
        # Apply final constraints directly to exploratory weights
        weights[0] = max(1.0, weights[0])  
        weights[1] = max(0.0, weights[1])  
        weights[2] = max(1.0, weights[2])  
        weights[3] = max(0.9, min(1.0, weights[3]))
        
        sys.stderr.write(f"Constrained exploratory weights: {weights}\\n")
        return weights
        
    else:
        # --- Normal Prediction using Wrapper --- 
        if is_stuck:
             sys.stderr.write(f"Stuck detected (drv={current_drv}) but taking greedy action.\\n")
        
        # Apply scaler
        try:
            normalized_state = scaler.transform(state)
            sys.stderr.write(f"Normalized state: {normalized_state}\n")
            
            # Predict using the stateful wrapper
            # Wrapper prediction expects (observation, reward)
            # We provide the current normalized state and assume reward=0 for the step before this
            # Reset wrapper state before each prediction for stateless script usage
            wrapper.reset()
            
            # Predict using the wrapper (already handles device internally? Check d3rlpy)
            # No need for torch.no_grad() as predict should handle it.
            # wrapper expects numpy array
            continuous_actions = wrapper.predict(normalized_state, 0.0) 
            
            # Ensure output is numpy array (should be)
            if not isinstance(continuous_actions, np.ndarray):
                # This might happen if predict returns list or tensor
                continuous_actions = np.array(continuous_actions)
            
            continuous_actions = continuous_actions.flatten()
            sys.stderr.write(f"Raw continuous actions (from wrapper): {continuous_actions}\n")
            
            # Apply domain constraints
            weights = continuous_actions
            weights[0] = max(1, round(weights[0]))  # drc_weight >= 1 (implicit via max(1,..))
            weights[1] = max(0.0, round(weights[1]))  # marker_weight >= 0.0
            weights[2] = max(1, round(weights[2]))  # fixed_weight >= 1 (implicit via max(1,..))
            weights[3] = max(0.9, min(1.0, weights[3])) # Keep constraint, continuous

            sys.stderr.write(f"Constrained weights: {weights}\n")
            return weights

        except Exception as e:
            sys.stderr.write(f"Error predicting weights: {e}\n")
            # Return default weights in case of error
            return np.array([1.0, 1.0, 1.0, 0.95], dtype=np.float32)

def main():
    args = parse_args()

    state_features = [
        args.drv, args.wireLength, args.max_box_drv, args.average_box_drv,
        args.num_violating, args.pin_count, args.net_count, args.iteration,
        args.drv_lag_1, args.drv_lag_2, args.drv_lag_3,
        # --- ADDED: Static Design Features (must match order in create_dataset_DT.py) ---
        args.pin_density,
        args.num_macros,
        args.component_density,
        args.guide_density,
        args.avg_pins_per_net,
        args.net_density,
        args.num_layers,
        args.terminal_density
    ]

    # Define hardcoded target return
    target_rtg = 4.315487

    try:
        # Load model and wrapper
        wrapper, device = load_model_and_wrapper(args.model_path, target_rtg)
        scaler = load_scaler(args.scaler_path)
        action_wrapper = MixedActionWrapper() # Instantiate the wrapper
        
        # Pass epsilon=0.1 (or make it an argument) for exploration
        weights = predict_weights(wrapper, device, scaler, state_features, action_wrapper, epsilon=0.1)
        
        weight_names = ["drc_weight", "marker_weight", "fixed_weight", "decay_weight"]
        if args.output_format == "json":
            weight_dict = {name: float(value) for name, value in zip(weight_names, weights)}
            print(json.dumps(weight_dict))
        else:
            print(",".join([str(float(w)) for w in weights]))

    except Exception as e:
        sys.stderr.write(f"Unexpected error in main: {e}\n")
        if args.output_format == "json":
            print(json.dumps({ "drc_weight": 1.0, "marker_weight": 1.0, "fixed_weight": 1.0, "decay_weight": 0.95 }))
        else:
            print("1.0,1.0,1.0,0.95")
        sys.exit(1)

if __name__ == "__main__":
    main()