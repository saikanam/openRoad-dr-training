# Script to process raw data and save it in .npz format for clean-offline-rl 
import pandas as pd
import numpy as np
import glob
import os
import argparse
from sklearn.preprocessing import RobustScaler
import joblib
import warnings

# Suppress specific warnings if needed, e.g., PerformanceWarning from pandas
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# --- Configuration ---
# These could be moved to a config file or made arguments
# Adjust input/output paths as needed relative to this script's location
DEFAULT_INPUT_DIR = '../../../training_data'
DEFAULT_OUTPUT_DIR = '../../../data/clean_cql_data' # Separate output dir
DEFAULT_OUTPUT_FILENAME = 'routing_dataset_cleanrl.npz'
DEFAULT_SCALER_FILENAME = 'state_scaler_cleanrl.joblib'
KNOWN_BAD_ITERATIONS = { # Example structure (designID, iteration) - Update with actual bad iterations
    # ('designA', 3),
    # ('designB', 5),
    # ... fill this based on your data analysis
}
DUPLICATE_FILE_PATTERN = 'trainingdata_bp_be_base.csv' # File known to have duplicates
HISTORY_K = 3 # Number of past DRV values to include in state
STUCK_WINDOW = 3 # Number of steps to check for 'stuck' condition (drv_{t} >= drv_{t-1} and drv_{t-1} >= drv_{t-2})
STUCK_TOLERANCE = 1e-5 # Tolerance for checking DRV increase
STUCK_PENALTY = -1.0 # Penalty value if stuck
CONVERGENCE_BONUS = 10.0 # Bonus value if drv_t == 0
# Reward component weights (adjust based on tuning)
BETA_PRIMARY = 1.0
BETA_BOX = 0.0 # Set to 0 if not using box_reward
BETA_NUM_VIOLATING = 1.0
BETA_STUCK = 0.5
BETA_CONVERGENCE = 1.0


# --- Helper Functions ---
def load_data(input_dir):
    """Loads all trainingdata_*.csv files from the input directory."""
    all_files = glob.glob(os.path.join(input_dir, "trainingdata_*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No training data CSV files found in {input_dir}")
    print(f"Found {len(all_files)} data files.")
    df_list = []
    for f in all_files:
        try:
            df_temp = pd.read_csv(f, low_memory=False) # low_memory=False can help with mixed types
            df_list.append(df_temp)
            print(f"Loaded {os.path.basename(f)}: {len(df_temp)} rows")
        except Exception as e:
            print(f"Error loading {f}: {e}")
    if not df_list:
        raise ValueError("No data loaded, check CSV files and paths.")
    df = pd.concat(df_list, ignore_index=True)
    print(f"Combined data shape: {df.shape}")
    # Basic type conversion and cleaning
    numeric_cols = ['iteration', 'pin_count', 'net_count', 'drv', 'wireLength',
                    'drc_weight', 'marker_weight', 'fixed_weight', 'decay_weight',
                    'box_size', 'box_drv', 'L_N_box', 'L_N_drv', 'R_N_box', 'R_N_drv']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def filter_invalid_data(df, known_bad_iterations):
    """Filters rows based on known bad iterations or missing critical values."""
    initial_rows = len(df)
    # Filter known bad iterations (if any provided)
    # Assuming known_bad_iterations is a set of tuples (designID, iteration)
    if known_bad_iterations:
         indices_to_drop = df[df.apply(lambda row: (row['designID'], row['iteration']) in known_bad_iterations, axis=1)].index
         df = df.drop(indices_to_drop)
         print(f"Dropped {len(indices_to_drop)} rows based on known bad iterations.")

    # Filter rows with missing critical values needed for aggregation/state/reward
    critical_cols = ['uniqueID', 'designID', 'iteration', 'drv', 'wireLength',
                     'drc_weight', 'marker_weight', 'fixed_weight', 'decay_weight',
                     'box_drv']
    missing_before = df[critical_cols].isnull().any(axis=1).sum()
    df = df.dropna(subset=critical_cols)
    missing_after = initial_rows - len(df) - (initial_rows - df[critical_cols].isnull().any(axis=1).sum() if 'indices_to_drop' not in locals() else len(indices_to_drop))
    print(f"Dropped {missing_after} additional rows due to NaN in critical columns.")
    print(f"Data shape after filtering: {df.shape}")
    if df.empty:
        raise ValueError("DataFrame is empty after filtering invalid data.")
    return df

def handle_duplicates(df, file_pattern):
    """Handles duplicates specifically for files matching the pattern."""
    # Example: Drop duplicates based on uniqueID, iteration, boxID, keeping the first
    # More sophisticated logic might be needed depending on the nature of duplicates
    mask = df['designID'].str.contains(os.path.splitext(file_pattern)[0], na=False) # Assuming designID relates to filename
    duplicates_before = df[mask].duplicated(subset=['uniqueID', 'iteration', 'boxID'], keep=False).sum()
    if duplicates_before > 0:
        print(f"Found {duplicates_before} potential duplicate box entries in files matching {file_pattern}.")
        # Apply specific logic, e.g., keep='first' or average values
        df = df.drop_duplicates(subset=['uniqueID', 'iteration', 'boxID'], keep='first')
        print(f"Applied 'drop_duplicates(keep=\"first\")' for relevant files.")
        print(f"Data shape after handling duplicates: {df.shape}")
    else:
        print(f"No duplicate box entries found based on criteria in files matching {file_pattern}.")
    return df

def aggregate_per_iteration(df):
    """Aggregates box-level data to the iteration level."""
    print("Aggregating data per iteration...")
    agg_funcs = {
        # Iteration-level features (take first value, should be constant)
        'designID': 'first',
        'pin_count': 'first',
        'net_count': 'first',
        'drv': 'first', # Iteration DRV
        'wireLength': 'first',
        'drc_weight': 'first',
        'marker_weight': 'first',
        'fixed_weight': 'first',
        'decay_weight': 'first',
        # Box-level aggregated features
        'box_drv': ['sum', 'mean', 'max', lambda x: (x > 0).sum()] # sum, mean, max, count_violating
    }
    # Check if all columns exist before aggregation
    cols_to_agg = {k: v for k, v in agg_funcs.items() if k in df.columns}
    missing_agg_cols = set(agg_funcs.keys()) - set(cols_to_agg.keys())
    if missing_agg_cols:
        print(f"Warning: Columns missing for aggregation: {missing_agg_cols}")

    if not cols_to_agg:
         raise ValueError("No columns available for aggregation.")

    df_agg = df.groupby(['uniqueID', 'iteration'], observed=True, sort=False).agg(cols_to_agg) # sort=False preserves original order

    # Rename aggregated columns for clarity
    df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]
    df_agg = df_agg.rename(columns={
        'box_drv_sum': 'total_box_drv',
        'box_drv_mean': 'average_box_drv',
        'box_drv_max': 'max_box_drv',
        'box_drv_<lambda>': 'num_violating' # Needs specific rename
    })
    # Rename the lambda column correctly if it exists
    lambda_col_name = next((col for col in df_agg.columns if 'lambda' in col), None)
    if lambda_col_name:
        df_agg = df_agg.rename(columns={lambda_col_name: 'num_violating'})

    # <<< ADD Rename for columns aggregated with 'first' >>>
    first_agg_cols = {
        'designID_first': 'designID',
        'pin_count_first': 'pin_count',
        'net_count_first': 'net_count',
        'drv_first': 'drv',
        'wireLength_first': 'wireLength',
        'drc_weight_first': 'drc_weight',
        'marker_weight_first': 'marker_weight',
        'fixed_weight_first': 'fixed_weight',
        'decay_weight_first': 'decay_weight'
    }
    # Rename only the columns that actually exist
    cols_to_rename_first = {k: v for k, v in first_agg_cols.items() if k in df_agg.columns}
    df_agg = df_agg.rename(columns=cols_to_rename_first)
    # <<< END Rename >>>

    # Calculate additional iteration features
    if 'drv' in df_agg.columns and 'total_box_drv' in df_agg.columns:
        df_agg['drv_difference'] = df_agg['total_box_drv'] - df_agg['drv']
    # Calculate violating percentage if 'num_violating' exists and box count is available (requires another pass or join)
    # For simplicity, we might omit percentage if box count per iteration isn't readily available here.

    print(f"Aggregated data shape: {df_agg.shape}")
    print("Aggregation complete.")
    return df_agg.reset_index() # Reset index to make uniqueID and iteration columns again

def calculate_rewards_and_terminals(df_agg, k=3, stuck_window=3, stuck_tolerance=1e-5,
                                    stuck_penalty=-1.0, convergence_bonus=10.0,
                                    beta_primary=1.0, beta_box=0.0, beta_num_violating=1.0,
                                    beta_stuck=0.5, beta_convergence=1.0):
    """Calculates rewards, terminals, and lagged features."""
    print("Calculating lagged features, rewards, and terminals...")
    df_agg = df_agg.sort_values(['uniqueID', 'iteration']).reset_index(drop=True)

    # --- Lagged Features for State & Reward ---
    # Lag features for CURRENT state (based on t-1 results)
    lag_cols_state = ['drv', 'wireLength', 'total_box_drv', 'average_box_drv', 'max_box_drv', 'num_violating']
    for col in lag_cols_state:
        if col in df_agg.columns:
            df_agg[f'{col}_lag_1'] = df_agg.groupby('uniqueID')[col].shift(1)

    # Lag features for historical DRV (state s_t needs drv_{t-1} ... drv_{t-k})
    if 'drv' in df_agg.columns:
        for i in range(1, k + 1):
            df_agg[f'drv_lag_{i}'] = df_agg.groupby('uniqueID')['drv'].shift(i)

    # Lag features for reward calculation (need t and t-1, sometimes t-2 for stuck)
    # We already have drv_lag_1, drv_lag_2 needed for stuck check
    if 'total_box_drv' in df_agg.columns:
        df_agg['total_box_drv_lag_1'] = df_agg.groupby('uniqueID')['total_box_drv'].shift(1) # Redundant if in lag_cols_state
    if 'num_violating' in df_agg.columns:
        df_agg['num_violating_lag_1'] = df_agg.groupby('uniqueID')['num_violating'].shift(1) # Redundant if in lag_cols_state

    # --- Calculate Reward Components ---
    # Rewards are calculated based on the change from state s_t to s_{t+1}
    # i.e. reward r_t depends on (drv_t, drv_{t-1}), (num_violating_t, num_violating_{t-1}) etc.
    if 'drv' in df_agg.columns and 'drv_lag_1' in df_agg.columns:
        df_agg['primary_reward'] = -(df_agg['drv'] - df_agg['drv_lag_1'])
    else: df_agg['primary_reward'] = 0.0

    if 'total_box_drv' in df_agg.columns and 'total_box_drv_lag_1' in df_agg.columns:
        df_agg['box_reward'] = -(df_agg['total_box_drv'] - df_agg['total_box_drv_lag_1'])
    else: df_agg['box_reward'] = 0.0

    if 'num_violating' in df_agg.columns and 'num_violating_lag_1' in df_agg.columns:
        df_agg['num_violating_reward'] = -(df_agg['num_violating'] - df_agg['num_violating_lag_1'])
    else: df_agg['num_violating_reward'] = 0.0

    # Stuck Penalty: requires drv_t, drv_{t-1}, drv_{t-2}
    df_agg['stuck_penalty'] = 0.0
    if all(f'drv_lag_{i}' in df_agg.columns for i in range(stuck_window)): # Check if drv_lag_0 (drv), drv_lag_1, drv_lag_2 exist
        stuck_cond = ((df_agg['drv'] >= df_agg['drv_lag_1'] - stuck_tolerance) &
                      (df_agg['drv_lag_1'] >= df_agg['drv_lag_2'] - stuck_tolerance))
        df_agg.loc[stuck_cond, 'stuck_penalty'] = stuck_penalty

    # Convergence Bonus: based on drv_t == 0
    df_agg['convergence_bonus'] = 0.0
    if 'drv' in df_agg.columns:
        converged_cond = (df_agg['drv'] == 0)
        df_agg.loc[converged_cond, 'convergence_bonus'] = convergence_bonus

    # --- Normalize Reward Components (Optional but recommended) ---
    # Calculate stats only on valid (non-NaN) rewards
    reward_components = ['primary_reward', 'box_reward', 'num_violating_reward', 'stuck_penalty']
    reward_stats = {}
    print("Normalizing reward components (excluding first step NaNs)...")
    for col in reward_components:
        if col in df_agg.columns:
            valid_rewards = df_agg[col].dropna()
            if len(valid_rewards) > 1:
                mean = valid_rewards.mean()
                std = valid_rewards.std()
                # Avoid division by zero if std is very small
                if std > 1e-6:
                    df_agg[f'{col}_norm'] = (df_agg[col] - mean) / std
                    reward_stats[col] = {'mean': mean, 'std': std}
                    print(f"  Normalized {col} (mean={mean:.3f}, std={std:.3f})")
                else:
                    df_agg[f'{col}_norm'] = 0.0 # Set to zero if std is zero
                    reward_stats[col] = {'mean': mean, 'std': 0.0}
                    print(f"  {col} has zero std dev, setting normalized to 0.0")

            else:
                df_agg[f'{col}_norm'] = 0.0 # Not enough data to normalize
                print(f"  Not enough data to normalize {col}, setting normalized to 0.0")
        else:
            df_agg[f'{col}_norm'] = 0.0 # Component doesn't exist

    # --- Combine Rewards ---
    df_agg['final_reward'] = (beta_primary * df_agg['primary_reward_norm'] +
                              beta_box * df_agg['box_reward_norm'] +
                              beta_num_violating * df_agg['num_violating_reward_norm'] +
                              beta_stuck * df_agg['stuck_penalty_norm'] + # Use normalized stuck penalty
                              beta_convergence * df_agg['convergence_bonus']) # Bonus usually not normalized

    # --- Impute Reward for First Transition ---
    # The first reward in each trajectory (where lagged values were NaN) will be NaN. Impute with 0.0.
    num_nan_rewards = df_agg['final_reward'].isna().sum()
    df_agg['final_reward'] = df_agg['final_reward'].fillna(0.0)
    print(f"Imputed {num_nan_rewards} NaN rewards (first steps) with 0.0.")

    # --- Determine Terminals ---
    # Terminal state is reached if drv_t == 0
    df_agg['terminal'] = (df_agg['drv'] == 0).astype(np.float32) # Use float32 for consistency

    # --- Shift Terminals for Transitions ---
    # The 'terminal' flag for transition (s_t, a_t, r_t, s_{t+1}) indicates if s_{t+1} is terminal.
    # So, we need to shift the terminal flag calculated based on drv_t.
    df_agg['terminal_shift'] = df_agg.groupby('uniqueID')['terminal'].shift(-1, fill_value=1.0) # Assume terminal if next state unknown

    print("Finished calculating rewards and terminals.")
    # Return stats for potential saving/logging
    return df_agg, reward_stats

def construct_transitions(df, k=3):
    """Constructs state, action, next_state tuples, handling padding and filtering."""
    print("Constructing transitions (s, a, r, s', terminal)...")

    # Define state features for s_t (based on iteration t-1 results)
    state_features = [
        'iteration', 'pin_count', 'net_count', # General features (iteration t is context)
        'drv_lag_1', 'wireLength_lag_1', # Performance from t-1
        'total_box_drv_lag_1', 'average_box_drv_lag_1', 'max_box_drv_lag_1', 'num_violating_lag_1' # Aggregated box from t-1
    ]
    # Add historical DRVs (drv_{t-1} to drv_{t-k})
    hist_drv_features = [f'drv_lag_{i}' for i in range(1, k + 1)]
    state_features.extend(hist_drv_features)

    # Define action features a_t (weights used in iteration t)
    action_features = ['drc_weight', 'marker_weight', 'fixed_weight', 'decay_weight']

    # Features for next state s_{t+1} (based on iteration t results - need non-lagged versions)
    next_state_base_features = ['drv', 'wireLength', 'total_box_drv', 'average_box_drv', 'max_box_drv', 'num_violating']
    next_state_features = ['iteration'] # Use iteration t+1 as context for s_{t+1}
    next_state_features.extend([f'{col}' for col in next_state_base_features]) # drv, wireLength, etc. at time t
     # Add historical DRVs for s_{t+1} (drv_t to drv_{t-k+1}) -> these are drv_lag_0 to drv_lag_{k-1}
    next_hist_drv_features = [f'drv_lag_{i}' for i in range(k)] # drv_lag_0=drv, drv_lag_1, ... drv_lag_{k-1}
    next_state_features.extend(next_hist_drv_features)


    # Filter out rows where essential components for s_t, a_t, or s_{t+1} might be missing
    # Crucially, the *first* row of each group will lack lagged features needed for s_t.
    # The *last* row of each group might lack next_state info if not handled by reward logic.
    required_features = state_features + action_features + ['final_reward', 'terminal_shift'] + \
                        ['iteration', 'pin_count', 'net_count'] # Base features for s_{t+1} context
    
    # Check for missing columns defensively
    missing_cols = [col for col in required_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing for transition construction: {missing_cols}")

    # Drop rows where ANY required feature for s_t or a_t is NaN.
    # s_t requires lagged features, so the first row of each trajectory will be dropped here.
    initial_rows = len(df)
    df_valid = df.dropna(subset=state_features + action_features).copy()
    dropped_rows = initial_rows - len(df_valid)
    print(f"Dropped {dropped_rows} rows with missing state/action features (expected for first steps).")

    if df_valid.empty:
         raise ValueError("No valid transitions remaining after dropping rows with missing features.")

    # Extract data, handling padding for historical features
    print("Extracting and padding states...")
    # State s_t
    states = df_valid[state_features].copy()
    for col in hist_drv_features:
        states[col] = states[col].fillna(-1.0) # Pad missing history with -1

    # Action a_t
    actions = df_valid[action_features].copy()

    # Reward r_t
    rewards = df_valid['final_reward'].copy()

    # Terminal flag (shifted)
    terminals = df_valid['terminal_shift'].copy()

    # Construct Next State s_{t+1} - Requires careful indexing from the original df
    # We need features corresponding to the *next* iteration step for each row in df_valid
    # Use shift(-1) on the *original* dataframe (before dropping NaNs) grouped by uniqueID
    print("Constructing next states...")
    df_next_state_data = pd.DataFrame(index=df.index) # Align with original index
    # Context for s_{t+1} is iteration t+1
    df_next_state_data['iteration'] = df.groupby('uniqueID')['iteration'].shift(-1) + 1
    df_next_state_data['pin_count'] = df.groupby('uniqueID')['pin_count'].shift(-1)
    df_next_state_data['net_count'] = df.groupby('uniqueID')['net_count'].shift(-1)
    # Base features for s_{t+1} are non-lagged features from iteration t
    for col in next_state_base_features:
         df_next_state_data[col] = df.groupby('uniqueID')[col].shift(-1)
    # Historical DRVs for s_{t+1} (drv_t ... drv_{t-k+1}) -> lag_0 ... lag_{k-1}
    for i in range(k):
         df_next_state_data[f'drv_lag_{i}'] = df.groupby('uniqueID')[f'drv_lag_{i}'].shift(-1)

    # Select only the rows corresponding to our valid transitions
    next_states_df = df_next_state_data.loc[df_valid.index]

    # Select the final required features for s_{t+1}
    missing_next_state_cols = [col for col in next_state_features if col not in next_states_df.columns]
    if missing_next_state_cols:
         raise ValueError(f"Columns missing for next state construction after shift: {missing_next_state_cols}")
         
    next_states = next_states_df[next_state_features].copy()


    # Pad missing history in next_states (should only affect last states if fill_value wasn't sufficient)
    for i in range(k):
        col = f'drv_lag_{i}'
        if col in next_states.columns:
             next_states[col] = next_states[col].fillna(-1.0)
        else:
             print(f"Warning: Expected history column {col} not found in next_states during padding.")

    # Handle NaNs that might arise from shifting the very last state of a trajectory
    final_nan_mask = next_states.isnull().any(axis=1)
    if final_nan_mask.any():
        print(f"Found {final_nan_mask.sum()} transitions where next state has NaNs (likely end of trajectory).")
        # Option 1: Fill remaining NaNs (e.g., with state s_t or zeros)
        # next_states = next_states.fillna(method='ffill', axis=1) # Or specific value
        # Option 2: Drop these transitions (might lose final step info)
        print("Dropping transitions with NaN next states.")
        states = states[~final_nan_mask]
        actions = actions[~final_nan_mask]
        rewards = rewards[~final_nan_mask]
        terminals = terminals[~final_nan_mask]
        next_states = next_states[~final_nan_mask]


    print(f"Final number of transitions: {len(states)}")
    if len(states) == 0:
        raise ValueError("No transitions constructed. Check data processing steps.")

    return states.values, actions.values, rewards.values.reshape(-1, 1), next_states.values, terminals.values.reshape(-1, 1)


def main(args):
    """Main data processing pipeline."""
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)
    scaler_path = os.path.join(args.output_dir, args.scaler_filename)

    df = load_data(args.input_dir)
    df = filter_invalid_data(df, KNOWN_BAD_ITERATIONS)
    # df = handle_duplicates(df, DUPLICATE_FILE_PATTERN) # Uncomment if needed
    df_agg = aggregate_per_iteration(df)
    df_rewards, reward_stats = calculate_rewards_and_terminals(
        df_agg, k=args.history_k, stuck_window=STUCK_WINDOW, stuck_tolerance=STUCK_TOLERANCE,
        stuck_penalty=STUCK_PENALTY, convergence_bonus=CONVERGENCE_BONUS,
        beta_primary=BETA_PRIMARY, beta_box=BETA_BOX, beta_num_violating=BETA_NUM_VIOLATING,
        beta_stuck=BETA_STUCK, beta_convergence=BETA_CONVERGENCE
    )

    # Save reward normalization stats if needed (e.g., to JSON)
    # import json
    # with open(os.path.join(args.output_dir, 'reward_stats.json'), 'w') as f:
    #     json.dump(reward_stats, f, indent=4)

    states, actions, rewards, next_states, terminals = construct_transitions(df_rewards, k=args.history_k)

    # --- Normalize States ---
    print(f"Normalizing states using RobustScaler (shape: {states.shape})...")
    scaler = RobustScaler()
    states_scaled = scaler.fit_transform(states)
    next_states_scaled = scaler.transform(next_states) # Use the same scaler

    # Save the scaler
    joblib.dump(scaler, scaler_path)
    print(f"Saved state scaler to {scaler_path}")

    # --- Save Dataset ---
    print(f"Saving processed data to {output_path}...")
    np.savez(
        output_path,
        observations=states_scaled.astype(np.float32),
        actions=actions.astype(np.float32),
        rewards=rewards.astype(np.float32),
        next_observations=next_states_scaled.astype(np.float32),
        terminals=terminals.astype(np.float32) # D4RL often uses 'terminals' or 'dones'
    )
    print("Dataset creation complete.")
    print(f"  Observations shape: {states_scaled.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Next observations shape: {next_states_scaled.shape}")
    print(f"  Terminals shape: {terminals.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process routing data for offline RL (clean-rl format).")
    parser.add_argument('--input-dir', type=str, default=DEFAULT_INPUT_DIR,
                        help='Directory containing raw trainingdata_*.csv files.')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save the processed .npz dataset and scaler.')
    parser.add_argument('--output-filename', type=str, default=DEFAULT_OUTPUT_FILENAME,
                        help='Name for the output .npz dataset file.')
    parser.add_argument('--scaler-filename', type=str, default=DEFAULT_SCALER_FILENAME,
                        help='Name for the saved state scaler file.')
    parser.add_argument('--history-k', type=int, default=HISTORY_K,
                        help='Number of past DRV values to include in state.')

    args = parser.parse_args()
    main(args) 