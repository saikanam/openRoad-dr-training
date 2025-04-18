import argparse
import os
import d3rlpy
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer, Episode, load # Added Episode, load
import numpy as np

def analyze_buffer(buffer_path):
    """Loads and analyzes a d3rlpy ReplayBuffer."""
    print(f"--- Analyzing ReplayBuffer: {buffer_path} ---")

    if not os.path.exists(buffer_path):
        print(f"Error: Buffer file not found at {buffer_path}")
        return

    try:
        print("Loading episodes directly to bypass ReplayBuffer.load issues...")
        # Load episodes directly
        with open(buffer_path, "rb") as f:
            episodes = load(Episode, f) # Load list of Episode objects
        print(f"Successfully loaded {len(episodes)} episodes.")

        if not episodes:
            print("No episodes found in the file.")
            return

        total_transitions = sum(len(ep) for ep in episodes)
        print(f"Total transitions calculated from episodes: {total_transitions}")

        # --- Calculate Actual Maximum Timestep --- 
        print("\n--- Calculating Actual Maximum Timestep ---")
        actual_max_timestep = 0
        max_len_episode_idx = -1
        for i, ep in enumerate(episodes):
            # Timesteps for DT are usually 0-indexed or 1-indexed length
            # d3rlpy expects timesteps from 0 to max_timestep - 1
            # The length of the episode corresponds to the number of steps.
            # If episode has length L, timesteps are typically 0, 1, ..., L-1
            # The maximum timestep value encountered is episode_length - 1.
            ep_max_timestep = len(ep) - 1
            if ep_max_timestep > actual_max_timestep:
                 actual_max_timestep = ep_max_timestep
                 max_len_episode_idx = i
        
        print(f"Actual maximum timestep found across all episodes: {actual_max_timestep}")
        print(f"(Found in episode index: {max_len_episode_idx} with length {len(episodes[max_len_episode_idx])})")
        # d3rlpy's max_timestep parameter should be > actual_max_timestep
        required_d3rlpy_max_timestep = actual_max_timestep + 1
        print(f"Required d3rlpy max_timestep parameter should be >= {required_d3rlpy_max_timestep}")

        # --- Inspect Sample Data from First Episode (Keep for quick check) ---
        print(f"\n--- Inspecting First Episode Thoroughly (First 5 steps) ---")
        try:
            episode = episodes[0]
            print(f"Episode length: {len(episode)}")
            print(f"Observations shape: {episode.observations.shape}, dtype: {episode.observations.dtype}")
            print(f"Actions shape: {episode.actions.shape}, dtype: {episode.actions.dtype}")
            print(f"Rewards shape: {episode.rewards.shape}, dtype: {episode.rewards.dtype}")
            print(f"Terminated: {episode.terminated}")
            print("First 5 Actions in Episode (Values & Types):")
            for i in range(min(5, len(episode.actions))):
                action_sample = episode.actions[i]
                print(f"  Step {i}: {action_sample}")
                print(f"    dtype: {action_sample.dtype}")
                print(f"    shape: {action_sample.shape}")
                if action_sample.shape[0] == 4:
                    print(f"    4th weight value: {action_sample[3]:.6f}") # Increased precision
        except Exception as e:
            print(f"Error inspecting first episode: {e}")

        # --- Analyze 4th Weight Distribution Across All Episodes ---
        print("\n--- Analyzing 4th Action Value (decay_weight) Distribution ---")
        all_fourth_weights = []
        action_dtypes = set()
        action_shapes = set()
        num_actions_analyzed = 0

        for i, ep in enumerate(episodes):
            if hasattr(ep, 'actions') and ep.actions is not None and len(ep.actions) > 0:
                # Check shape and type consistency
                action_dtypes.add(str(ep.actions.dtype))
                action_shapes.add(str(ep.actions.shape[1:])) # Shape per step

                if ep.actions.shape[1] == 4:
                    all_fourth_weights.extend(ep.actions[:, 3].tolist()) # Extract 4th column
                    num_actions_analyzed += len(ep.actions)
                else:
                    print(f"Warning: Episode {i} has unexpected action shape {ep.actions.shape}")
            else:
                 print(f"Warning: Episode {i} has no actions or actions are None.")

        print(f"Analyzed {num_actions_analyzed} actions across {len(episodes)} episodes.")
        print(f"Action dtypes found: {action_dtypes}")
        print(f"Action shapes (per step) found: {action_shapes}")

        if all_fourth_weights:
            weights_array = np.array(all_fourth_weights)
            print("Statistics for 4th Action Value (decay_weight):")
            print(f"  Min: {np.min(weights_array):.6f}")
            print(f"  Max: {np.max(weights_array):.6f}")
            print(f"  Mean: {np.mean(weights_array):.6f}")
            print(f"  Std Dev: {np.std(weights_array):.6f}")
            print(f"  Unique values count: {len(np.unique(weights_array))}")
            # Print first 10 unique values to see variety
            unique_vals = np.unique(weights_array)
            print(f"  First 10 unique values: {unique_vals[:10]}")
        else:
            print("Could not extract any 4th weight values for analysis.")

    except FileNotFoundError:
        print(f"Error: File not found at {buffer_path}")
    except Exception as e:
        print(f"An unexpected error occurred during loading or analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze episodes from a d3rlpy HDF5 file.")
    parser.add_argument("buffer_path", type=str,
                        help="Path to the dataset HDF5 file (.h5).")

    args = parser.parse_args()
    analyze_buffer(args.buffer_path) 