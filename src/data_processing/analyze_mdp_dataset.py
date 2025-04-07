import os
import numpy as np
from d3rlpy.dataset import MDPDataset, ReplayBuffer, InfiniteBuffer

def analyze_mdp_dataset(dataset_path="data/routing_dataset.h5"):
    """
    Analyze the contents of a saved MDPDataset.
    """
    print(f"Loading dataset from {dataset_path}...")
    try:
        with open(dataset_path, "rb") as f:
            replay_buffer = ReplayBuffer.load(f, buffer=InfiniteBuffer())
        
        # Get dataset components using the buffer's episodes
        episodes = replay_buffer.episodes
        
        # Collect all transitions
        observations = []
        actions = []
        rewards = []
        terminals = []
        
        for episode in episodes:
            observations.extend(episode.observations)
            actions.extend(episode.actions)
            rewards.extend(episode.rewards)
            terminals.extend([0] * (len(episode.observations) - 1) + [1])
        
        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        terminals = np.array(terminals)
        
        print("\nDataset Statistics:")
        print(f"Number of episodes: {len(episodes)}")
        print(f"Number of transitions: {len(observations)}")
        print(f"Observation shape: {observations.shape}")
        print(f"Action shape: {actions.shape}")
        print(f"Number of terminal states: {np.sum(terminals)}")
        
        print("\nObservation Statistics:")
        print(f"Mean: {np.mean(observations, axis=0)}")
        print(f"Std: {np.std(observations, axis=0)}")
        print(f"Min: {np.min(observations, axis=0)}")
        print(f"Max: {np.max(observations, axis=0)}")
        
        print("\nAction Statistics (weights):")
        print("drc_weight, marker_weight, fixed_weight, decay_weight")
        print(f"Mean: {np.mean(actions, axis=0)}")
        print(f"Std: {np.std(actions, axis=0)}")
        print(f"Min: {np.min(actions, axis=0)}")
        print(f"Max: {np.max(actions, axis=0)}")
        
        print("\nReward Statistics:")
        print(f"Mean: {np.mean(rewards):.4f}")
        print(f"Std: {np.std(rewards):.4f}")
        print(f"Min: {np.min(rewards):.4f}")
        print(f"Max: {np.max(rewards):.4f}")
        
        # Calculate episode lengths
        episode_lengths = [len(episode.observations) for episode in episodes]
            
        print("\nEpisode Statistics:")
        print(f"Number of episodes: {len(episode_lengths)}")
        print(f"Average episode length: {np.mean(episode_lengths):.2f}")
        print(f"Min episode length: {np.min(episode_lengths)}")
        print(f"Max episode length: {np.max(episode_lengths)}")
        
        return {
            'n_transitions': len(observations),
            'n_episodes': len(episodes),
            'obs_shape': observations.shape,
            'action_shape': actions.shape,
            'avg_episode_length': np.mean(episode_lengths),
            'n_terminals': np.sum(terminals)
        }
        
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        return None
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        return None

if __name__ == "__main__":
    analyze_mdp_dataset() 