#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import os
import sys
import d3rlpy # Import d3rlpy
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer # Import ReplayBuffer components

# --- Constants ---
# Removed constants related to raw data processing
DEFAULT_REPLAY_BUFFER_PATH = "data/routing_dataset_dt_replaybuffer.h5" # Default path for the DT buffer

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Max Scaled RTG from a d3rlpy ReplayBuffer.")
    parser.add_argument("--buffer_path", type=str, default=DEFAULT_REPLAY_BUFFER_PATH,
                        help=f"Path to the ReplayBuffer HDF5 file (default: {DEFAULT_REPLAY_BUFFER_PATH})")
    return parser.parse_args()

# --- Removed Replicated Helper Functions --- 
# load_and_clean_data, aggregate_iterations, engineer_features, 
# calculate_rewards, normalize_rewards, combine_rewards, scale_final_rewards
# are no longer needed as we load the final buffer.

# --- Main Calculation ---
def main():
    args = parse_args()

    print(f"--- Loading ReplayBuffer from: {args.buffer_path} ---")
    if not os.path.exists(args.buffer_path):
        sys.stderr.write(f"Error: ReplayBuffer file not found at {args.buffer_path}\n")
        sys.stderr.write(f"Please ensure the dataset for DT was created using:")
        sys.stderr.write(f"python src/data_processing/create_dataset.py --algo_type dt --output_filename {os.path.basename(args.buffer_path)}\n")
        sys.exit(1)

    try:
        # Load the ReplayBuffer
        # We need to provide a buffer instance (e.g., InfiniteBuffer) during load
        with open(args.buffer_path, "rb") as f:
            replay_buffer = ReplayBuffer.load(f, buffer=InfiniteBuffer())
        print("ReplayBuffer loaded successfully.")
        print(f"  Contains {len(replay_buffer.episodes)} episodes and {replay_buffer.transition_count} transitions.")

        # Calculate total scaled reward per episode directly from the buffer
        print("\n--- Calculating Total Scaled Reward per Episode --- ")
        if not replay_buffer.episodes:
            print("Warning: ReplayBuffer contains no episodes.")
            episode_returns = []
        else:
            episode_returns = [ep.rewards.sum() for ep in replay_buffer.episodes]

        if not episode_returns:
             print("No episode returns calculated.")
             max_scaled_rtg = np.nan
             avg_scaled_rtg = np.nan
             min_scaled_rtg = np.nan
             returns_series = pd.Series([], dtype=float)
        else:
             episode_returns = np.array(episode_returns)
        max_scaled_rtg = episode_returns.max()
        avg_scaled_rtg = episode_returns.mean()
        min_scaled_rtg = episode_returns.min()
        returns_series = pd.Series(episode_returns)

        print("\n--- Results --- ")
        print(f"Maximum Scaled Return-to-Go (Target RTG for DT): {max_scaled_rtg:.6f}")
        print(f"Average Scaled Return-to-Go: {avg_scaled_rtg:.6f}")
        print(f"Minimum Scaled Return-to-Go: {min_scaled_rtg:.6f}")
        print(f"\nDistribution of Scaled Episode Returns:")
        # Use describe() on the pandas Series for nice formatting
        if not returns_series.empty:
            print(returns_series.describe().to_string())
        else:
            print("(No returns to describe)")
        print("\nNOTE: The 'Maximum Scaled Return-to-Go' is the recommended initial value for 'target_return' during DT inference.")

        # --- Stuck/Unstuck analysis removed --- 
        # print("\n--- Analyzing Transitions Where Agent Got UNSTUCK (Removed) ---")
        # print("(This analysis is complex to perform accurately from the normalized buffer data and has been removed from this script)")

    except Exception as e:
        sys.stderr.write(f"\nAn error occurred: {e}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()