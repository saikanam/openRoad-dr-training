import pandas as pd
import numpy as np
import glob
import os
import argparse
from sklearn.preprocessing import RobustScaler  # Using RobustScaler as planned
import joblib # Import joblib for saving the scaler
from d3rlpy.dataset import MDPDataset # Import d3rlpy
# from d3rlpy.dataset import MDPDataset # Import later when actually creating dataset

# --- Constants & Configuration ---

# Data Paths
# TODO: Make these configurable via args
DEFAULT_INPUT_DIR = "training_data" # Changed default relative path
DEFAULT_OUTPUT_DIR = "data" # Relative to WORKSPACE ROOT
DEFAULT_OUTPUT_FILENAME = "routing_dataset.h5"

# Known problematic iterations to filter (uniqueID, iteration)
# From dataset_test.md analysis
KNOWN_BAD_ITERATIONS = [
    ('aes_cipher_top_run_46', 2), ('bp_be_top_run_69', 8),
    ('ispd18_test1_run_12', 1), ('ispd18_test3_run_14', 11),
    ('ispd18_test3_run_30', 16), ('ispd18_test4_run_17', 6)
]

# Duplicate BoxID handling target
DUPLICATE_BOXID_FILE = "trainingdata_bp_be_base.csv"

# Reward Calculation Parameters (Placeholders - values TBD/tuned)
STUCK_WINDOW = 2
STUCK_PENALTY_VALUE = -50.0 # Original example value
CONVERGENCE_BONUS_VALUE = 100.0 # Original example value
BETA_PRIMARY_REWARD = 1.0  # Weight for primary_reward
BETA_BOX_REWARD = 1.0 # Original example weight
BETA_NUM_VIOLATING_REWARD = 1.0 # Original example weight
BETA_STUCK_PENALTY = 1.0 # Original example weight
REWARD_NORMALIZATION_EPSILON = 1e-8
# USE_REWARD_NORMALIZATION = False # Remove this flag, always use normalization now

# State History Window (k)
STATE_HISTORY_K = 3 # Number of previous iterations' DRV to include in state

# Columns to convert to numeric
NUMERIC_COLS = [
    'iteration', 'pin_count', 'net_count', 'drv',
    'wireLength', 'drc_weight', 'marker_weight', 'fixed_weight',
    'decay_weight', 'box_drv', 'L_N_drv', 'R_N_drv'
]

# --- Argument Parsing ---

def parse_args():
    parser = argparse.ArgumentParser(description="Process raw routing data into an MDPDataset for offline RL.")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing raw trainingdata_*.csv files (default: {DEFAULT_INPUT_DIR}, relative to project root)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save the processed dataset and scaler (default: {DEFAULT_OUTPUT_DIR}, relative to project root)")
    parser.add_argument("--output_filename", type=str, default=DEFAULT_OUTPUT_FILENAME,
                        help=f"Filename for the output MDPDataset (default: {DEFAULT_OUTPUT_FILENAME})")
    # Add more arguments as needed for configuration (e.g., reward weights, scaler type)
    return parser.parse_args()

# --- Helper Functions ---

def print_step_header(step_num, title):
    print(f"\n--- Step {step_num}: {title} ---")

# --- Data Loading & Cleaning ---

def load_and_clean_data(input_dir):
    print_step_header(1, "Load and Clean Raw Data")
    all_files = glob.glob(os.path.join(input_dir, "trainingdata_*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No training data CSVs found in {input_dir}")

    print(f"Found {len(all_files)} data files.")
    df_list = []
    total_rows_before_cleaning = 0
    for f in all_files:
        filename = os.path.basename(f)
        try:
            df_file = pd.read_csv(f)
            total_rows_before_cleaning += len(df_file)
            print(f"  Read {filename} ({len(df_file)} rows)")

            # --- Initial Cleaning Steps --- 
            initial_rows_file = len(df_file)
            rows_filtered_bad_iter = 0
            
            # Filter known bad iterations
            for uid, it in KNOWN_BAD_ITERATIONS:
                rows_before = len(df_file)
                df_file = df_file[~((df_file['uniqueID'] == uid) & (df_file['iteration'] == it))]
                rows_filtered_bad_iter += (rows_before - len(df_file))
            if rows_filtered_bad_iter > 0:
                print(f"    Filtered {rows_filtered_bad_iter} rows due to KNOWN_BAD_ITERATIONS.")

            # Handle duplicate boxIDs in the specific file
            if filename == DUPLICATE_BOXID_FILE:
                dupe_rows_before = len(df_file[df_file.duplicated(subset=['uniqueID', 'iteration', 'boxID'], keep=False)])
                if dupe_rows_before > 0:
                    df_file = df_file.drop_duplicates(subset=['uniqueID', 'iteration', 'boxID'], keep='first')
                    print(f"    Handled {dupe_rows_before} duplicate boxID rows in {filename} by keeping first.")
            
            # Convert columns to numeric, coercing errors
            for col in NUMERIC_COLS:
                if col in df_file.columns:
                    df_file[col] = pd.to_numeric(df_file[col], errors='coerce')
                else:
                    print(f"    Warning: Expected numeric column '{col}' not found in {filename}.")
            
            df_list.append(df_file)

        except Exception as e:
            print(f"  Error processing file {filename}: {e}. Skipping this file.")
            continue

    if not df_list:
        raise ValueError("No dataframes could be loaded or processed.")

    df_combined = pd.concat(df_list, ignore_index=True)
    print(f"Combined data shape before cleaning: ({total_rows_before_cleaning}, {df_combined.shape[1]})")
    print(f"Combined data shape after initial cleaning: {df_combined.shape}")

    # Check for rows entirely NA (might indicate issues)
    na_rows = df_combined.isnull().all(axis=1).sum()
    if na_rows > 0:
        print(f"Warning: Found {na_rows} rows that are entirely NA after loading/cleaning.")

    return df_combined

# --- Iteration Aggregation ---

def aggregate_iterations(df):
    print_step_header(2, "Aggregate Data by Iteration")
    if df is None or df.empty:
        raise ValueError("Input DataFrame for aggregation is empty.")

    # Columns expected to be constant within an iteration group
    iteration_constant_cols = [
        'designID', 'pin_count', 'net_count', 'drv', 'wireLength',
        'drc_weight', 'marker_weight', 'fixed_weight', 'decay_weight'
    ]

    # Ensure required columns exist
    required_cols = ['uniqueID', 'iteration', 'box_drv'] + iteration_constant_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for aggregation: {missing_cols}")

    print("Grouping data by uniqueID and iteration...")
    grouped = df.groupby(['uniqueID', 'iteration'])

    print("Performing aggregation...")
    aggregated_data = grouped.agg(
        total_box_drv=('box_drv', 'sum'),
        max_box_drv=('box_drv', 'max'),
        num_violating=('box_drv', lambda x: (x > 0).sum())
    ).reset_index() # Keep uniqueID and iteration as columns for now

    # Get the first value for columns assumed constant within the group
    print("Merging constant iteration columns...")
    first_values = grouped[iteration_constant_cols].first().reset_index()

    # Merge aggregated values with the first values
    final_aggregated = pd.merge(aggregated_data, first_values, on=['uniqueID', 'iteration'], how='left')

    print(f"Aggregation complete. Result shape: {final_aggregated.shape}")
    # Check for NaNs introduced potentially by first() if group was empty after cleaning (shouldn't happen ideally)
    na_check = final_aggregated[iteration_constant_cols].isnull().sum()
    if na_check.sum() > 0:
        print(f"Warning: NaNs found in supposedly constant columns after aggregation:\n{na_check[na_check > 0]}")
        print(f"Dropping {na_check.sum()} rows with NaN values in drv or wireLength after aggregation.")
        final_aggregated = final_aggregated.dropna(subset=['drv', 'wireLength'])
        print(f"Shape after dropping NaN drv/wireLength: {final_aggregated.shape}")

    # Sort for consistency (important for lag features later)
    final_aggregated = final_aggregated.sort_values(by=['uniqueID', 'iteration']).reset_index(drop=True)

    return final_aggregated

# --- Feature Engineering ---

def engineer_features(df):
    print_step_header(3, "Engineer Features (Lag/Lead)")
    if df is None or df.empty:
        raise ValueError("Input DataFrame for feature engineering is empty.")

    # Ensure data is sorted for correct lag/lead calculation
    df = df.sort_values(by=['uniqueID', 'iteration']).reset_index(drop=True)

    print(f"Initial shape for feature engineering: {df.shape}")

    # --- Calculate Lagged Features --- 
    # Need up to lag STATE_HISTORY_K for state, and lag STUCK_WINDOW+1 for stuck penalty
    max_lag_needed = max(STATE_HISTORY_K, STUCK_WINDOW + 1)
    print(f"Calculating lagged DRV features (up to lag {max_lag_needed})...")
    for k in range(1, max_lag_needed + 1):
        df[f'drv_lag_{k}'] = df.groupby('uniqueID')['drv'].shift(k)

    # --- Calculate *Previous* Step Features (for Reward Calculation) ---
    # Instead of next, we look at current vs previous.
    # These features represent the state at t-1, used to calculate reward for action a_t
    reward_lag_features = ['drv', 'total_box_drv', 'max_box_drv', 'num_violating']
    print(f"Calculating previous step features for reward: {reward_lag_features}...")
    for col in reward_lag_features:
        df[f'{col}_lag_1_reward'] = df.groupby('uniqueID')[col].shift(1)

    # --- Keep ALL rows for now --- 
    # Filtering based on NaN rewards happens after reward calculation
    print(f"Shape after feature engineering (no filtering here): {df.shape}")

    return df

# --- Reward Calculation ---

def calculate_rewards(df):
    print_step_header(4, "Calculate Reward Components")
    if df is None or df.empty:
        raise ValueError("Input DataFrame for reward calculation is empty.")

    reward_df = df.copy()
    print(f"Calculating rewards for {len(reward_df)} potential transitions...")

    # Required columns for reward: current values and lag_1_reward values
    required_current = ['drv', 'total_box_drv', 'max_box_drv', 'num_violating']
    required_lags_reward = [f'{col}_lag_1_reward' for col in required_current]
    required_lags_stuck = [f'drv_lag_{k}' for k in range(1, STUCK_WINDOW + 1)]
    all_req = required_current + required_lags_reward + required_lags_stuck
    
    missing_cols = [col for col in all_req if col not in reward_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for reward calculation: {missing_cols}")
    
    # --- Calculate Individual Reward Components --- 
    # Reward = S_{t-1} - S_t
    print("  Calculating reward components...")
    reward_df['primary_reward'] = reward_df['drv_lag_1_reward'] - reward_df['drv']
    reward_df['box_reward'] = reward_df['total_box_drv_lag_1_reward'] - reward_df['total_box_drv']
    reward_df['max_box_drv_reward'] = reward_df['max_box_drv_lag_1_reward'] - reward_df['max_box_drv']
    reward_df['num_violating_reward'] = reward_df['num_violating_lag_1_reward'] - reward_df['num_violating']

    # Stuck Penalty: Check if DRV hasn't decreased for STUCK_WINDOW steps
    is_stuck = pd.Series(True, index=reward_df.index)
    # Check drv[i] >= drv[i-1], drv[i-1] >= drv[i-2], ...
    for k in range(STUCK_WINDOW):
        current_drv_col = 'drv' if k == 0 else f'drv_lag_{k}'
        prev_drv_col = f'drv_lag_{k+1}'
        # If previous is NA (start of sequence), not stuck
        is_stuck = is_stuck & (reward_df[current_drv_col] >= reward_df[prev_drv_col]).fillna(False)
    reward_df['stuck_penalty'] = 0.0
    reward_df.loc[is_stuck, 'stuck_penalty'] = STUCK_PENALTY_VALUE

    # Convergence Bonus: If current state DRV is 0
    reward_df['convergence_bonus'] = 0.0
    reward_df.loc[reward_df['drv'] == 0, 'convergence_bonus'] = CONVERGENCE_BONUS_VALUE
    
    # --- Log Reward Distributions --- 
    reward_cols = ['primary_reward', 'box_reward', 'max_box_drv_reward', 'num_violating_reward', 'stuck_penalty', 'convergence_bonus']
    print("  Reward Component Distributions (Before NaN filtering):")
    for col in reward_cols:
        if col in reward_df.columns:
            # Calculate describe on non-NaN values only
            desc = reward_df[col].dropna().describe()
            print(f"    {col}: (Valid count: {int(desc['count'])})")
            print(desc.to_string())
        else:
            print(f"    {col}: Not calculated.")

    # Clean up intermediate lag columns used only for reward calculation
    cols_to_drop = required_lags_reward
    # Also drop DRV lags beyond what's needed for state history
    for k in range(STATE_HISTORY_K + 1, max(STATE_HISTORY_K, STUCK_WINDOW + 1) + 1):
        lag_col = f'drv_lag_{k}'
        if lag_col in reward_df.columns:
            cols_to_drop.append(lag_col)
            
    reward_df = reward_df.drop(columns=cols_to_drop, errors='ignore')
    print(f"Shape after reward calculation and cleanup: {reward_df.shape}")
            
    return reward_df

# --- Reward Normalization ---

def normalize_rewards(df):
    print_step_header(5, "Normalize Reward Components")
    if df is None or df.empty:
        raise ValueError("Input DataFrame for reward normalization is empty.")

    norm_df = df.copy()
    reward_cols_to_normalize = [
        'primary_reward', 'box_reward', 'max_box_drv_reward',
        'num_violating_reward', 'stuck_penalty', 'convergence_bonus'
    ]
    print(f"Normalizing reward components: {reward_cols_to_normalize}...")

    for col in reward_cols_to_normalize:
        if col not in norm_df.columns:
            print(f"  Warning: Column '{col}' not found for normalization. Skipping.")
            continue
        
        # Calculate std dev on non-NaN values
        valid_values = norm_df[col].dropna()
        if valid_values.empty:
            print(f"  Warning: No valid values for {col}. Skipping normalization.")
            norm_df[f'{col}_norm'] = norm_df[col] # Keep NaNs
            continue
            
        std_dev = valid_values.std()
        if pd.isna(std_dev) or std_dev < REWARD_NORMALIZATION_EPSILON:
            print(f"  Warning: Std dev for {col} is near zero ({std_dev}) based on {len(valid_values)} values. Skipping normalization.")
            norm_df[f'{col}_norm'] = norm_df[col] # Assign original values
        else:
            mean = valid_values.mean()
            norm_df[f'{col}_norm'] = (norm_df[col] - mean) / (std_dev + REWARD_NORMALIZATION_EPSILON)
            print(f"  Normalized {col} (mean={mean:.4f}, std={std_dev:.4f}).")

    return norm_df

# --- Final Reward Combination ---

def combine_rewards(df):
    # Reverted to always combining normalized rewards
    print_step_header(6, "Combine Normalized Rewards")
    if df is None or df.empty:
        raise ValueError("Input DataFrame for reward combination is empty.")

    final_df = df.copy()

    # Always use normalized components now
    print("Combining **normalized** reward components...")
    # Define which normalized components to include and their weights
    reward_components = {
        # Using BETA constants which are now back to defaults (1.0)
        'primary_reward_norm': 1.0, # Always weight primary component by 1.0
        'box_reward_norm': BETA_BOX_REWARD,
        'num_violating_reward_norm': BETA_NUM_VIOLATING_REWARD,
        'stuck_penalty_norm': BETA_STUCK_PENALTY,
        'convergence_bonus_norm': 1.0 # Weight for convergence bonus (usually 1.0)
    }

    print("Calculating final_reward using weighted sum of normalized components:")
    final_df['final_reward'] = 0.0
    for col, weight in reward_components.items():
        # Check if the normalized component column exists
        if col in final_df.columns:
            print(f"  Adding: {col} * {weight}")
            # Ensure the column is numeric and fill potential NaNs with 0 before adding
            final_df['final_reward'] += final_df[col].fillna(0.0) * weight
        else:
            print(f"  Warning: Component '{col}' not found for final reward calculation. Skipping.")

    print("Final Reward Distribution:")
    try:
         print(final_df['final_reward'].describe().to_string())
    except Exception as e:
         print(f"  Could not describe final_reward: {e}")

    return final_df

# --- State Construction ---

def construct_states(df):
    print_step_header(7, "Construct State Vectors")
    if df is None or df.empty:
        raise ValueError("Input DataFrame for state construction is empty.")

    # Define state features
    state_features = [
        'drv', 'wireLength', 'total_box_drv', 'max_box_drv', 'num_violating',
        'pin_count', 'net_count',
        'iteration' # Include iteration number as a feature
    ]
    # Add lagged DRV features
    lagged_drv_features = []
    for k in range(1, STATE_HISTORY_K + 1):
        lagged_drv_features.append(f'drv_lag_{k}')
    state_features.extend(lagged_drv_features)

    print(f"Constructing states using features: {state_features}")

    # Check if all features exist
    missing_features = [f for f in state_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required state features: {missing_features}")

    # --- Padding Logic --- 
    print("Applying padding (-1) to missing lagged DRV features...")
    states_df = df[state_features].copy()
    for col in lagged_drv_features:
        states_df[col] = states_df[col].fillna(-1)
        
    # Check if padding worked (no NaNs should remain in lagged columns)
    nans_after_padding = states_df[lagged_drv_features].isnull().sum().sum()
    if nans_after_padding > 0:
        print(f"Warning: {nans_after_padding} NaNs still present in lagged features after padding.")

    # Convert to NumPy array
    states = states_df.astype(np.float32).to_numpy()

    # Check for NaNs/Infs in the final state array (excluding padding)
    if np.isnan(states).any() or np.isinf(states).any():
        print("Warning: NaNs or Infs found in the constructed state vectors OUTSIDE padding. Check input data and feature calculations.")
        # Handle them if necessary

    print(f"Constructed state vectors. Shape: {states.shape}")
    return states

# --- State Normalization ---

def normalize_states(states, output_dir):
    print_step_header(8, "Normalize State Vectors")
    if states is None or states.shape[0] == 0:
        raise ValueError("Input states array for normalization is empty.")

    scaler = RobustScaler()
    print(f"Fitting RobustScaler to states of shape {states.shape}...")
    scaler.fit(states) # Fit scaler

    print("Transforming states...")
    states_normalized = scaler.transform(states) # Normalize

    # Save the scaler
    scaler_path = os.path.join(output_dir, "state_scaler.pkl")
    print(f"Saving fitted scaler to {scaler_path}...")
    try:
        joblib.dump(scaler, scaler_path)
    except Exception as e:
        print(f"Error saving scaler: {e}")
        # Decide if this should be a fatal error

    print(f"State normalization complete. Normalized shape: {states_normalized.shape}")
    return states_normalized, scaler # Return both normalized states and the scaler

# --- Action & Terminal Extraction ---

def extract_actions_terminals(df):
    print_step_header(9, "Extract Actions and Terminals")
    if df is None or df.empty:
        raise ValueError("Input DataFrame for action/terminal extraction is empty.")

    action_cols = ['drc_weight', 'marker_weight', 'fixed_weight', 'decay_weight']
    # Terminal condition is based on the DRV of the *current* state (s_t)
    terminal_col = 'drv' 

    print(f"Extracting actions using columns: {action_cols}")
    # Check for missing action columns
    missing_action_cols = [f for f in action_cols if f not in df.columns]
    if missing_action_cols:
        raise ValueError(f"Missing required action features: {missing_action_cols}")
        
    actions = df[action_cols].astype(np.float32).to_numpy()

    print(f"Extracting terminals based on '{terminal_col}' == 0")
    if terminal_col not in df.columns:
         raise ValueError(f"Missing required column for terminal calculation: '{terminal_col}'")
    
    # Terminal is True if the *current* state has DRV = 0
    terminals = (df[terminal_col] == 0).astype(np.float32).to_numpy()
    
    print(f"Extracted actions shape: {actions.shape}")
    print(f"Extracted terminals shape: {terminals.shape} (Number of terminals: {int(terminals.sum())})")

    # Check for NaNs/Infs in actions
    if np.isnan(actions).any() or np.isinf(actions).any():
        print("Warning: NaNs or Infs found in the extracted action vectors. Check input data.")

    return actions, terminals

# --- Assemble & Save MDPDataset ---

def save_dataset(df, output_dir, output_filename):
    print_step_header(10, "Assemble and Save MDPDataset")
    if df is None or df.empty:
        raise ValueError("Input DataFrame for dataset assembly is empty.")

    # --- Calculate next_drv for terminal condition ---
    # Shift requires sorted data
    df = df.sort_values(by=['uniqueID', 'iteration']).reset_index(drop=True)
    df['next_drv'] = df.groupby('uniqueID')['drv'].shift(-1)
    # For the last state in each trajectory, next_drv will be NaN.
    # The terminal status for the *last* transition should be based on the final state's DRV.
    # So, where next_drv is NaN, use the current drv to determine terminal.
    # A state is terminal if its DRV is 0.
    # terminals[t] should be True if state s_{t+1} is terminal (drv_{t+1} == 0).
    # We use next_drv (which is drv_{t+1}) when available.
    # When next_drv is NaN (at the end of a trajectory), the episode has ended *after* this state,
    # but d3rlpy typically uses the 'terminal' flag to indicate if s_{t+1} requires bootstrapping.
    # If drv_t is 0, the episode ended, and the transition (s_t, a_t, r_t) leads to this terminal state s_t.
    # The 'terminals' flag in d3rlpy often corresponds to 'done' flags in Gym environments.
    # Let's define terminal[t] = (df['next_drv'] == 0).fillna(False) for now.
    # This means the last transition in a trajectory that doesn't end in drv=0 won't be marked terminal.
    # If a trajectory ends with drv=0 at step T, then next_drv for step T-1 is 0, so terminals[T-1] = True.
    # The terminal flag for the very last state T itself isn't explicitly used in standard RL updates.
    df['is_terminal_next'] = (df['next_drv'] == 0)
    # For the very last state of each trajectory, next_drv is NaN.
    # We can fillna based on the current drv for the last state, but typically d3rlpy's
    # terminal flag is associated with the *transition*, not the absolute last state.
    # Let's fill NaN terminals (last step of each trajectory) with False.
    # This means the bootstrap term in Bellman update is only zeroed if the *next* state is terminal.
    terminals_calc = df['is_terminal_next'].fillna(False).astype(np.float32).to_numpy()


    # --- Impute NaN Rewards ---
    # Rewards are NaN for the first step of each trajectory because lag_1 features are NaN
    nan_reward_mask = df['final_reward'].isna()
    num_nan_rewards = nan_reward_mask.sum()
    if num_nan_rewards > 0:
        print(f"Imputing {num_nan_rewards} NaN final_rewards (first steps) with 0.0.")
        df['final_reward'] = df['final_reward'].fillna(0.0)
    else:
        print("No NaN rewards found to impute.")

    # Now use the full dataframe as all transitions are kept
    df_filtered = df

    # --- Extract Final Components ---
    print("Extracting final observations, actions, rewards, terminals...")
    # Construct final states (s_t) using the full df
    observations = construct_states(df_filtered)
    # Normalize final states (s_t)
    observations_normalized, _ = normalize_states(observations, output_dir)
    # Extract final actions (a_t) - No change needed here
    action_cols = ['drc_weight', 'marker_weight', 'fixed_weight', 'decay_weight']
    if not all(c in df_filtered.columns for c in action_cols):
        raise ValueError(f"Missing action columns for dataset creation: {action_cols}")
    actions = df_filtered[action_cols].astype(np.float32).to_numpy()

    # Use the pre-calculated terminals based on next_drv
    terminals = terminals_calc

    # Extract final rewards (r_t) - should have no NaNs now
    rewards = df_filtered['final_reward'].astype(np.float32).to_numpy()

    # Input validation
    if not (observations_normalized.shape[0] == actions.shape[0] == rewards.shape[0] == terminals.shape[0]):
        raise ValueError(f"Final components have inconsistent lengths: "
                         f"Obs={observations_normalized.shape[0]}, Act={actions.shape[0]}, Rew={rewards.shape[0]}, Term={terminals.shape[0]}")
    if rewards.ndim != 1:
        rewards = rewards.flatten()
    if terminals.ndim != 1:
        terminals = terminals.flatten()

    # Final check for NaN rewards after imputation
    if np.isnan(rewards).any():
        raise ValueError("NaN values still present in final rewards after imputation! Check logic.")

    n_transitions = observations_normalized.shape[0]
    print(f"Assembling final MDPDataset with {n_transitions} transitions.")
    print(f"  Observations shape: {observations_normalized.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Terminals shape: {terminals.shape} (Number of terminals: {int(terminals.sum())})") # Now reflects next state terminal

    # Create MDPDataset
    dataset = MDPDataset(
        observations=observations_normalized,
        actions=actions,
        rewards=rewards,
        terminals=terminals # Use the correctly calculated terminals
    )

    # Save dataset
    output_path = os.path.join(output_dir, output_filename)
    print(f"Saving dataset to {output_path}...")
    try:
        dataset.dump(output_path)
        print("Dataset saved successfully.")
    except Exception as e:
        print(f"Error saving dataset: {e}")

# --- Main Function ---

def main():
    args = parse_args()

    print("--- Starting Dataset Preparation ---")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output filename: {args.output_filename}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Ensured output directory exists: {args.output_dir}")

    # --- Phase 1 Steps ---
    try:
        # Step 1: Load and Clean Data
        raw_df = load_and_clean_data(args.input_dir)

        # Step 2: Aggregate Iterations
        aggregated_df = aggregate_iterations(raw_df)

        # Step 3: Feature Engineering (Lag)
        # Keeps all rows, adds lag columns
        feature_df = engineer_features(aggregated_df)

        # Step 4: Calculate Reward Components
        # Rewards might be NaN for the first step
        reward_df = calculate_rewards(feature_df)

        # Step 5: Normalize Reward Components
        # Normalizes based on non-NaN values
        normalized_reward_df = normalize_rewards(reward_df)

        # Step 6: Combine Rewards
        # final_reward might be NaN for the first step
        final_reward_df = combine_rewards(normalized_reward_df)

        # Step 7-10 are now combined in save_dataset after filtering
        # Step 7: Construct States (happens inside save_dataset)
        # Step 8: Normalize States (happens inside save_dataset)
        # Step 9: Extract Actions & Terminals (happens inside save_dataset)
        # Step 10: Assemble & Save MDPDataset (includes filtering NaN rewards)
        save_dataset(final_reward_df, args.output_dir, args.output_filename)

        print("\n--- Dataset Preparation Script Finished Successfully ---")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Consider adding more specific exception handling as needed


if __name__ == "__main__":
    main() 