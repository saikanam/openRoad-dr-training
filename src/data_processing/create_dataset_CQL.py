import pandas as pd
import numpy as np
import glob
import os
import argparse
import json # Added json import
from sklearn.preprocessing import RobustScaler, StandardScaler  # Using RobustScaler as planned
import joblib # Import joblib for saving the scaler
from d3rlpy.dataset import MDPDataset # Import d3rlpy
from d3rlpy.dataset import Episode, ReplayBuffer, InfiniteBuffer # Add ReplayBuffer imports
from d3rlpy.dataset import BasicTransitionPicker, BasicTrajectorySlicer, BasicWriterPreprocess # Add ReplayBuffer component imports
from d3rlpy.dataset import Signature # Import Signature
from d3rlpy.constants import ActionSpace # Import ActionSpace
# from d3rlpy.dataset import MDPDataset # Import later when actually creating dataset
import gc # Added garbage collection import

# --- Constants & Configuration ---

# Data Paths
# TODO: Make these configurable via args
DEFAULT_INPUT_DIR = "../training_data" # Changed default relative path
DEFAULT_OUTPUT_DIR = "data" # Relative to WORKSPACE ROOT
DEFAULT_OUTPUT_FILENAME = "routing_dataset.h5"

# Known problematic iterations to filter (uniqueID, iteration)
# From dataset_test.md analysis
# KNOWN_BAD_ITERATIONS = [
#     ('aes_cipher_top_run_46', 2), ('bp_be_top_run_69', 8),
#     ('ispd18_test1_run_12', 1), ('ispd18_test3_run_14', 11),
#     ('ispd18_test3_run_30', 16), ('ispd18_test4_run_17', 6)
# ]
KNOWN_BAD_ITERATIONS = []

# Duplicate BoxID handling target
# DUPLICATE_BOXID_FILE = "trainingdata_bp_be_base.csv"
DUPLICATE_BOXID_FILE = ""

# Reward Calculation Parameters
STUCK_WINDOW = 2 # Reverted to 2
STUCK_PENALTY_VALUE = -100.0 # Increased penalty
CONVERGENCE_BONUS_VALUE = 1000.0 # Increased bonus
STEP_PENALTY_VALUE = -1.0 # Added constant step penalty
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
    parser = argparse.ArgumentParser(description="Process raw routing data into an MDPDataset or ReplayBuffer for offline RL.")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing raw trainingdata_*.csv files (default: {DEFAULT_INPUT_DIR}, relative to project root)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save the processed dataset and scaler (default: {DEFAULT_OUTPUT_DIR}, relative to project root)")
    parser.add_argument("--output_filename", type=str, default=DEFAULT_OUTPUT_FILENAME,
                        help=f"Filename for the output dataset (default: {DEFAULT_OUTPUT_FILENAME})")
    parser.add_argument("--algo_type", type=str, default="cql", choices=["cql", "dt"],
                        help="Algorithm type to create dataset for ('cql' for MDPDataset, 'dt' for ReplayBuffer) (default: cql)")
    # Add more arguments as needed for configuration (e.g., reward weights, scaler type)
    return parser.parse_args()

# --- Helper Functions ---

def print_step_header(step_num, title):
    print(f"\n--- Step {step_num}: {title} ---")

# --- Data Loading & Cleaning (Generator) ---

def load_and_clean_data_generator(input_dir):
    """
    Generator function that loads, cleans, and yields DataFrames one file at a time.
    """
    print_step_header("1", "Load and Clean Raw Data (Iteratively)")
    all_files = glob.glob(os.path.join(input_dir, "trainingdata_*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No training data CSVs found in {input_dir}")

    print(f"Found {len(all_files)} data files. Processing one by one...")
    files_processed_count = 0

    for f in all_files:
        filename = os.path.basename(f)
        print(f"\n  Processing file: {filename}...")
        try:
            # Consider adding dtype optimizations here if needed later
            # For example: dtype={'iteration': 'int32', ...}
            df_file = pd.read_csv(f)
            initial_rows_file = len(df_file)
            print(f"    Read {initial_rows_file} rows.")

            # --- Initial Cleaning Steps ---
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
            numeric_cols_found = []
            for col in NUMERIC_COLS:
                if col in df_file.columns:
                    df_file[col] = pd.to_numeric(df_file[col], errors='coerce')
                    numeric_cols_found.append(col)
                else:
                    print(f"    Warning: Expected numeric column '{col}' not found in {filename}.")
            print(f"    Converted columns to numeric: {numeric_cols_found}")

            # Check for rows entirely NA (might indicate issues)
            na_rows = df_file.isnull().all(axis=1).sum()
            if na_rows > 0:
                print(f"    Warning: Found {na_rows} rows that are entirely NA after cleaning in this file.")

            print(f"    Finished cleaning. Shape: {df_file.shape}")
            files_processed_count += 1
            yield df_file # Yield the cleaned DataFrame for this file

        except Exception as e:
            print(f"  Error processing file {filename}: {e}. Skipping this file.")
            continue

    print(f"\nFinished processing {files_processed_count} files.")

# --- ADDED: Load Static Design Features ---
def load_static_features(static_features_dir):
    """Loads static design features from JSON files."""
    print_step_header("0", "Load Static Design Features")
    static_features_path = os.path.join(static_features_dir, "*_features.json")
    all_feature_files = glob.glob(static_features_path)
    if not all_feature_files:
        raise FileNotFoundError(f"No static feature JSON files found in {static_features_dir}")

    print(f"Found {len(all_feature_files)} static feature files.")
    static_features_dict = {}
    required_static_keys = [
        'pin_density', 'num_macros', 'component_density', 'guide_density',
        'avg_pins_per_net', 'net_density', 'num_layers', 'terminal_density'
    ]

    for f_path in all_feature_files:
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
            design_name = data.get('design_name')
            if not design_name:
                print(f"  Warning: Skipping {os.path.basename(f_path)}, missing 'design_name'.")
                continue

            # Extract only the required features
            features = {key: data.get(key) for key in required_static_keys}

            # Validate that all required keys were found
            if None in features.values():
                 missing = [key for key, value in features.items() if value is None]
                 print(f"  Warning: Skipping {design_name}, missing required keys: {missing}")
                 continue

            static_features_dict[design_name] = features
            print(f"  Loaded features for: {design_name}")
        except json.JSONDecodeError:
            print(f"  Warning: Error decoding JSON from {os.path.basename(f_path)}. Skipping.")
        except Exception as e:
            print(f"  Warning: Error processing file {os.path.basename(f_path)}: {e}. Skipping.")

    if not static_features_dict:
         raise ValueError("Could not load any valid static features.")

    print(f"Successfully loaded static features for {len(static_features_dict)} designs.")
    return static_features_dict

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
        # total_box_drv=('box_drv', 'sum'), # Commented out
        max_box_drv=('box_drv', 'max'),
        average_box_drv=('box_drv', 'mean'),
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
    reward_lag_features = ['drv', 
                           # 'total_box_drv', # Commented out
                           'max_box_drv', 'average_box_drv', 'num_violating']
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
    required_current = ['drv', 'max_box_drv', 'average_box_drv', 'num_violating']
    required_lags_reward = [f'{col}_lag_1_reward' for col in required_current]
    # Ensure lags needed for the potentially increased STUCK_WINDOW are present
    required_lags_stuck = [f'drv_lag_{k}' for k in range(1, STUCK_WINDOW + 1)]
    all_req = required_current + required_lags_reward + required_lags_stuck

    missing_cols = [col for col in all_req if col not in reward_df.columns]
    if missing_cols:
        # Check if missing lags are just beyond the max history K, which is expected if STUCK_WINDOW > STATE_HISTORY_K
        missing_due_to_stuck_window = all(col.startswith('drv_lag_') and int(col.split('_')[-1]) > STATE_HISTORY_K for col in missing_cols)
        if not missing_due_to_stuck_window:
            # Raise error only if missing columns are not explainable by STUCK_WINDOW > STATE_HISTORY_K
             raise ValueError(f"Missing required columns for reward calculation: {missing_cols}")
        else:
             print(f"    Note: Some required lag columns for STUCK_WINDOW={STUCK_WINDOW} ({missing_cols}) might be missing if STUCK_WINDOW > STATE_HISTORY_K={STATE_HISTORY_K}. The engineer_features step only calculates up to max(STATE_HISTORY_K, STUCK_WINDOW+1). Calculation will proceed but stuck penalty might be affected at sequence starts.")


    # --- Calculate Individual Reward Components --- 
    # Reward = S_{t-1} - S_t
    print("  Calculating reward components...")
    reward_df['primary_reward'] = reward_df['drv_lag_1_reward'] - reward_df['drv']
    # reward_df['box_reward'] = reward_df['total_box_drv_lag_1_reward'] - reward_df['total_box_drv'] # Commented out
    reward_df['max_box_drv_reward'] = reward_df['max_box_drv_lag_1_reward'] - reward_df['max_box_drv']
    reward_df['average_box_drv_reward'] = reward_df['average_box_drv_lag_1_reward'] - reward_df['average_box_drv']
    reward_df['num_violating_reward'] = reward_df['num_violating_lag_1_reward'] - reward_df['num_violating']

    # Stuck Penalty: Check if DRV hasn't decreased for STUCK_WINDOW steps
    is_stuck = pd.Series(True, index=reward_df.index)
    # Check drv[t] >= drv[t-1], drv[t-1] >= drv[t-2], ...
    for k in range(STUCK_WINDOW):
        current_drv_col = 'drv' if k == 0 else f'drv_lag_{k}'
        prev_drv_col = f'drv_lag_{k+1}'
        # Handle cases where needed lag columns might be missing (start of sequences or if STUCK_WINDOW > STATE_HISTORY_K)
        if current_drv_col not in reward_df.columns or prev_drv_col not in reward_df.columns:
            print(f"    Skipping stuck check for index {k} due to missing columns ('{current_drv_col}' or '{prev_drv_col}'). Assuming not stuck for this comparison.")
            # If required columns are missing, cannot determine stuck status based on this comparison
            is_stuck = pd.Series(False, index=reward_df.index) # Assume not stuck if data is missing for check
            break # Can't continue the check for this step
        # If previous is NA (start of sequence), not stuck based on this comparison
        # Combine conditions: is_stuck remains True only if it was True before AND current >= prev
        is_stuck = is_stuck & (reward_df[current_drv_col] >= reward_df[prev_drv_col]).fillna(False)

    reward_df['stuck_penalty'] = 0.0
    reward_df.loc[is_stuck, 'stuck_penalty'] = STUCK_PENALTY_VALUE

    # Convergence Bonus: If current state DRV is 0
    reward_df['convergence_bonus'] = 0.0
    reward_df.loc[reward_df['drv'] == 0, 'convergence_bonus'] = CONVERGENCE_BONUS_VALUE

    # Step Penalty: Apply constant penalty to every step
    reward_df['step_penalty'] = STEP_PENALTY_VALUE

    # --- Log Reward Distributions ---
    reward_cols = ['primary_reward',
                   # 'box_reward', # Commented out
                   'max_box_drv_reward', 'average_box_drv_reward', 'num_violating_reward',
                   'stuck_penalty', 'convergence_bonus', 'step_penalty'] # Added step_penalty
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

# --- Reward Normalization (REMOVED) ---
# def normalize_rewards(df):
#     # ... (This step is now removed) ...
#     pass 

# --- Final Reward Combination ---

def combine_rewards(df):
    print_step_header(6, "Combine Raw Rewards (Simplified)") # Updated header
    if df is None or df.empty:
        raise ValueError("Input DataFrame for reward combination is empty.")

    final_df = df.copy()

    print("Combining **raw** reward components (Simplified version)...") # Updated print
    # Define which raw components to include and their weights
    reward_components = {
        'primary_reward': BETA_PRIMARY_REWARD, 
        # 'num_violating_reward': BETA_NUM_VIOLATING_REWARD, # Temporarily removed
        'stuck_penalty': BETA_STUCK_PENALTY, 
        'convergence_bonus': 1.0, # Keep convergence bonus weight at 1.0 unless defined otherwise
        'step_penalty': 1.0 # Keep step penalty weight at 1.0 unless defined otherwise
    }

    print("Calculating final_reward using weighted sum of raw components:")
    final_df['final_reward'] = 0.0
    for col, weight in reward_components.items():
        if col in final_df.columns:
            print(f"  Adding: {col} * {weight}")
            # Use fillna(0.0) for potential NaNs (should only be first step)
            final_df['final_reward'] += final_df[col].fillna(0.0) * weight 
        else:
            print(f"  Warning: Component '{col}' not found for final reward calculation. Skipping.")

    print("Final Reward Distribution (Raw, Simplified):") # Updated print
    try:
         print(final_df['final_reward'].describe().to_string())
    except Exception as e:
         print(f"  Could not describe final_reward: {e}")

    # Clean up intermediate reward component columns (optional, but good practice)
    cols_to_drop = list(reward_components.keys()) # Drop the raw components we just used
    # Also drop normalized columns if they somehow exist from previous runs (unlikely)
    norm_cols_to_drop = [f"{col}_norm" for col in reward_components.keys()]
    final_df = final_df.drop(columns=cols_to_drop + norm_cols_to_drop, errors='ignore')
    print(f"Shape after reward combination and cleanup: {final_df.shape}")

    return final_df

# --- ADDED: Global Final Reward Scaling --- 
def scale_final_reward(df, output_dir):
    print_step_header("6.5", "Scale Final Reward Globally")
    if df is None or df.empty or 'final_reward' not in df.columns:
        raise ValueError("Input DataFrame for final reward scaling is invalid.")

    scaler = StandardScaler()
    rewards = df[['final_reward']].astype(np.float32) # Select as DataFrame, ensure float32
    
    # Check for NaNs before scaling
    if rewards.isnull().any().any():
        print("Warning: NaNs found in final_reward before scaling. Imputing with 0.0")
        rewards = rewards.fillna(0.0)
        # Update the original dataframe as well if imputation happened
        df['final_reward'] = rewards['final_reward']
        
    print(f"Fitting StandardScaler to final_reward column (shape: {rewards.shape})...")
    scaler.fit(rewards)

    print("Transforming final_reward...")
    scaled_rewards = scaler.transform(rewards)

    # Update the DataFrame column with scaled values
    df['final_reward_scaled'] = scaled_rewards.flatten() # Add as new column

    print("Scaled Final Reward Distribution:")
    try:
        print(df['final_reward_scaled'].describe().to_string())
    except Exception as e:
        print(f"  Could not describe scaled_final_reward: {e}")
        
    # Save the scaler
    scaler_path = os.path.join(output_dir, "final_reward_scaler.pkl")
    print(f"Saving fitted reward scaler to {scaler_path}...")
    try:
        joblib.dump(scaler, scaler_path)
    except Exception as e:
        print(f"Error saving reward scaler: {e}")

    return df, scaler # Return df with new column and the scaler

# --- State Construction ---

def construct_states(df):
    print_step_header(7, "Construct State Vectors")
    if df is None or df.empty:
        raise ValueError("Input DataFrame for state construction is empty.")

    # Define state features
    dynamic_features = [
        'drv', 'wireLength',
        # 'total_box_drv', # Commented out
        'max_box_drv',
        'average_box_drv',
        'num_violating',
        'pin_count', 'net_count',
        'iteration' # Include iteration number as a feature
    ]
    # Add lagged DRV features
    lagged_drv_features = []
    for k in range(1, STATE_HISTORY_K + 1):
        lagged_drv_features.append(f'drv_lag_{k}')

    # Define static features to include (ensure these columns exist after merging)
    static_features = [
        'pin_density', 'num_macros', 'component_density', 'guide_density',
        'avg_pins_per_net', 'net_density', 'num_layers', 'terminal_density'
    ]

    state_features = dynamic_features + lagged_drv_features + static_features

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

# --- Assemble & Save MDPDataset / ReplayBuffer ---

def save_dataset(df, output_dir, output_filename, algo_type):
    print_step_header(10, f"Assemble and Save Dataset for {algo_type.upper()}")
    if df is None or df.empty:
        raise ValueError("Input DataFrame for dataset assembly is empty.")

    # --- Calculate Max Timestep (Max Iteration per Trajectory) ---
    print("Calculating maximum timestep (max iterations per uniqueID)...")
    # Iteration is 1-based, timestep is usually 0-based in sequences,
    # but d3rlpy's max_timestep seems to relate to the maximum step count.
    # Let's use the maximum iteration number found.
    max_timestep = df.groupby('uniqueID')['iteration'].max().max()
    print(f"  Maximum timestep found across all trajectories: {max_timestep}")

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

    # --- Enforce Action Space Constraints on Data --- 
    action_cols = ['drc_weight', 'marker_weight', 'fixed_weight', 'decay_weight']
    print("Enforcing action space constraints (rounding/clipping) on dataset actions...")
    if not all(c in df_filtered.columns for c in action_cols):
        raise ValueError(f"Missing action columns for constraint enforcement: {action_cols}")
        
    #Round first three weights to nearest integer, clip to [0, 100]
    for col in action_cols[:3]:
        df_filtered[col] = df_filtered[col].round().astype(float).clip(0, 100)
        print(f"  Rounded and clipped '{col}' to integer [0, 100].")
        
    # Clip decay_weight to [0, 1]
    # df_filtered[action_cols[3]] = df_filtered[action_cols[3]].clip(0.0, 1.0)
    # print(f"  Clipped '{action_cols[3]}' to [0.0, 1.0].")

    # --- Calculate and Save Action Min/Max & Reward Min/Max (AFTER constraints) ---
    # action_cols defined above
    print("Recalculating action and reward statistics AFTER enforcing constraints...")
    action_min_vals = df_filtered[action_cols].min().astype(np.float32).to_numpy()
    action_max_vals = df_filtered[action_cols].max().astype(np.float32).to_numpy()
    reward_min_val = df_filtered['final_reward'].min()
    reward_max_val = df_filtered['final_reward'].max()
    
    print(f"  Action Minimums: {action_min_vals}")
    print(f"  Action Maximums: {action_max_vals}")
    print(f"  Reward Minimum: {reward_min_val:.4f}")
    print(f"  Reward Maximum: {reward_max_val:.4f}")
    
    action_stats_path = os.path.join(output_dir, "action_reward_stats.npz") # Renamed file
    print(f"Saving action min/max, reward min/max, and max_timestep to {action_stats_path}...")
    try:
        # Save max_timestep along with action and reward stats
        np.savez(action_stats_path, 
                 action_minimum=action_min_vals, 
                 action_maximum=action_max_vals, 
                 reward_minimum=np.float32(reward_min_val),
                 reward_maximum=np.float32(reward_max_val),
                 max_timestep=max_timestep)
        print("Action/Reward stats and max_timestep saved successfully.")
    except Exception as e:
        print(f"Error saving stats: {e}")

    # --- Manually Scale Rewards (Min-Max to [0, 1]) --- 
    # print("Manually applying Min-Max scaling to rewards...") # REMOVED Min-Max scaling
    # if reward_max_val > reward_min_val:
    #     df_filtered['final_reward_scaled'] = (df_filtered['final_reward'] - reward_min_val) / (reward_max_val - reward_min_val)
    #     print(f"  Scaled rewards range: [{df_filtered['final_reward_scaled'].min():.4f}, {df_filtered['final_reward_scaled'].max():.4f}]")
    # else:
    #     # Handle case where all rewards are the same
    #     print("  Warning: Reward max equals reward min. Setting scaled rewards to 0.0.")
    #     df_filtered['final_reward_scaled'] = 0.0
        
    # --- Extract Final Components (Common for both types) ---
    print("Extracting final components (observations, actions, FINAL rewards)...")
    # Construct final states (s_t) using the full df
    observations = construct_states(df_filtered)
    # Normalize final states (s_t)
    observations_normalized, _ = normalize_states(observations, output_dir) # Scaler saved here
    # Extract final actions (a_t) - AFTER constraints applied to df_filtered
    action_cols = ['drc_weight', 'marker_weight', 'fixed_weight', 'decay_weight']
    if not all(c in df_filtered.columns for c in action_cols):
        raise ValueError(f"Missing action columns for dataset creation: {action_cols}")
    actions = df_filtered[action_cols].astype(np.float32).to_numpy()
    
    epsilon = 1e-7
    actions += epsilon
    print(f"Added epsilon ({epsilon}) to actions to ensure continuous detection.")

    # Extract final rewards (r_t) - Use final_reward directly, should have no NaNs now
    # rewards_scaled = df_filtered['final_reward_scaled'].astype(np.float32).to_numpy() # Old
    rewards = df_filtered['final_reward'].astype(np.float32).to_numpy() # Use final_reward directly

    # Input validation (lengths)
    # if not (observations_normalized.shape[0] == actions.shape[0] == rewards_scaled.shape[0]): # Old
    if not (observations_normalized.shape[0] == actions.shape[0] == rewards.shape[0]):
         raise ValueError(f"Intermediate components have inconsistent lengths: "
                          # f"Obs={observations_normalized.shape[0]}, Act={actions.shape[0]}, Rew={rewards_scaled.shape[0]}") # Old
                          f"Obs={observations_normalized.shape[0]}, Act={actions.shape[0]}, Rew={rewards.shape[0]}")
    # if rewards_scaled.ndim != 1: # Old
    #     rewards_scaled = rewards_scaled.flatten()
    if rewards.ndim != 1:
        rewards = rewards.flatten()

    # Final check for NaN rewards after imputation
    # if np.isnan(rewards_scaled).any(): # Old
    if np.isnan(rewards).any():
        raise ValueError("NaN values still present in final rewards! Check logic.")

    n_transitions = observations_normalized.shape[0]
    print(f"Total transitions processed: {n_transitions}")
    print(f"  Observations shape: {observations_normalized.shape}")
    print(f"  Actions shape: {actions.shape}")
    # print(f"  Rewards shape (scaled): {rewards_scaled.shape}") # Old
    print(f"  Rewards shape: {rewards.shape}")


    # --- Create and Save Appropriate Dataset Type ---
    output_path = os.path.join(output_dir, output_filename)

    if algo_type == 'cql':
        print("Creating MDPDataset for CQL...")
        # Need terminals for MDPDataset
        # Use the pre-calculated terminals based on next_drv
        terminals = terminals_calc # Already calculated earlier
        if terminals.ndim != 1:
            terminals = terminals.flatten()
        if terminals.shape[0] != n_transitions:
             raise ValueError(f"Terminals length ({terminals.shape[0]}) doesn't match other components ({n_transitions})")
        print(f"  Terminals shape: {terminals.shape} (Number of terminals: {int(terminals.sum())})")

        dataset = MDPDataset(
            observations=observations_normalized,
            actions=actions,
            # rewards=rewards_scaled, # Use scaled rewards for CQL too now # Old
            rewards=rewards, # Use original final_reward for CQL as well
            terminals=terminals
        )
        print(f"Saving MDPDataset to {output_path}...")
        try:
            dataset.dump(output_path)
            print("MDPDataset saved successfully.")
        except Exception as e:
            print(f"Error saving MDPDataset: {e}")

    elif algo_type == 'dt':
        print("Creating ReplayBuffer for DT...")
        
        # 1. Group transitions into Episode objects first
        print("Grouping transitions into episodes...")
        # Add original df indices to map back for grouping
        df_filtered['orig_index'] = df_filtered.index
        grouped_indices = df_filtered.groupby('uniqueID')['orig_index'].apply(list)

        episodes = []
        num_episodes = 0
        for unique_id, indices in grouped_indices.items():
            if not indices: continue
            # Slice the final numpy arrays based on original indices
            ep_observations = observations_normalized[indices]
            # Use the actions with added epsilon
            ep_actions = actions[indices] 
            # Rewards need to be reshaped to (N, 1) for Episode
            # ep_rewards = rewards_scaled[indices].reshape(-1, 1) # Use SCALED rewards # Old
            ep_rewards = rewards[indices].reshape(-1, 1) # Use FINAL rewards
            
            # Determine if the episode terminated (last state DRV == 0)
            # Access the original DRV from the dataframe for the last step
            last_original_drv = df_filtered.loc[indices[-1], 'drv']
            terminated = (last_original_drv == 0)

            episode = Episode(
                observations=ep_observations,
                actions=ep_actions,
                rewards=ep_rewards,
                terminated=terminated # Pass the terminated flag
            )
            episodes.append(episode)
            num_episodes += 1
        print(f"Created {num_episodes} episodes.")

        # 2. Initialize ReplayBuffer with the created episodes
        # Define buffer components (optional, defaults are usually fine)
        buffer = InfiniteBuffer() # Store all data
        # TransitionPicker and TrajectorySlicer have defaults, often suitable
        # transition_picker = BasicTransitionPicker()
        # trajectory_slicer = BasicTrajectorySlicer()

        print("Initializing ReplayBuffer with episodes...")
        replay_buffer = ReplayBuffer(
            buffer=buffer,
            # transition_picker=transition_picker, # Use default
            # trajectory_slicer=trajectory_slicer, # Use default
            episodes=episodes # Provide the episodes directly
            # No need for signatures or action_space when providing episodes
        )

        print(f"ReplayBuffer initialized. Size: {replay_buffer.transition_count} transitions.")

        # 3. Save ReplayBuffer
        print(f"Saving ReplayBuffer to {output_path}...")
        try:
            # Use ReplayBuffer's dump method
            with open(output_path, "w+b") as f:
                replay_buffer.dump(f)
            print("ReplayBuffer saved successfully.")
        except Exception as e:
            print(f"Error saving ReplayBuffer: {e}")

    else:
        # Should not happen due to argparse choices
        raise ValueError(f"Unknown algo_type: {algo_type}")


# --- Main Function ---

def main():
    args = parse_args()

    print("--- Starting Dataset Preparation ---")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output filename: {args.output_filename}")
    print(f"Target Algorithm Type: {args.algo_type.upper()}") # Print chosen type

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Ensured output directory exists: {args.output_dir}")

    # --- Phase 1 Steps (Iterative Loading/Aggregation) ---
    try:
        # --- ADDED Step 0: Load Static Features ---
        static_features_dict = load_static_features("../training_data/static_design_features") # Hardcoded path
        static_features_df = pd.DataFrame.from_dict(static_features_dict, orient='index')
        static_features_df.index.name = 'design_name' # Set index name for merging
        static_features_df = static_features_df.reset_index() # Make design_name a column

        # Step 1 & 2: Load, Clean, and Aggregate Iteratively
        aggregated_data_list = []
        file_generator = load_and_clean_data_generator(args.input_dir)

        file_count = 0
        for raw_df_chunk in file_generator:
            file_count += 1
            print(f"\n  Processing file {file_count}...")
            try:
                # Aggregate this chunk
                print(f"  Aggregating data from file {file_count}...")
                aggregated_chunk = aggregate_iterations(raw_df_chunk)
                if not aggregated_chunk.empty:
                    aggregated_data_list.append(aggregated_chunk)
                    print(f"    Finished aggregation for file {file_count}. Shape: {aggregated_chunk.shape}")
                else:
                    print(f"    Warning: Aggregation resulted in empty DataFrame for file {file_count}.")
            except ValueError as e:
                 print(f"    Error during aggregation for file {file_count}: {e}. Skipping chunk.")
            except Exception as e:
                 print(f"    Unexpected error during aggregation for file {file_count}: {e}. Skipping chunk.")

            # Explicitly delete the raw chunk and collect garbage
            del raw_df_chunk
            gc.collect()


        if not aggregated_data_list:
            raise ValueError("No data could be aggregated from any input file.")

        print("\n--- Combining Aggregated Data ---")
        aggregated_df = pd.concat(aggregated_data_list, ignore_index=True)
        del aggregated_data_list # Free memory
        gc.collect()
        print(f"Combined aggregated data shape: {aggregated_df.shape}")

        # --- ADDED: Extract design_name and Merge Static Features (AFTER Aggregation) ---
        print("\n--- Extracting design_name and Merging Static Features ---")
        # Extract design_name from uniqueID
        aggregated_df['design_name'] = aggregated_df['uniqueID'].str.rsplit('_run_', n=1).str[0]
        
        # Perform the merge on the combined aggregated data
        initial_rows = len(aggregated_df)
        aggregated_df = pd.merge(aggregated_df, static_features_df, on='design_name', how='left')
        rows_after_merge = len(aggregated_df)
        if initial_rows != rows_after_merge:
            print(f"Warning: Row count changed during merge! Before: {initial_rows}, After: {rows_after_merge}")
            
        # Check for rows where merge failed (static features are NaN)
        failed_merge_count = aggregated_df[static_features_df.columns.drop('design_name')].isnull().all(axis=1).sum()
        if failed_merge_count > 0:
            print(f"Warning: {failed_merge_count} rows could not be matched with static features after aggregation.")
            failed_designs = aggregated_df.loc[aggregated_df[static_features_df.columns.drop('design_name')].isnull().all(axis=1), 'design_name'].unique()
            print(f"  Unmatched design_names (first 5): {failed_designs[:5]}")
            # Consider dropping failed merges if static features are essential
            # aggregated_df = aggregated_df.dropna(subset=static_features_df.columns.drop('design_name'))
            # print(f"Dropped {failed_merge_count} rows with missing static features. New shape: {aggregated_df.shape}")
        
        print(f"Static features merged into aggregated data. Shape after merge: {aggregated_df.shape}")
        # --- END Static Feature Merge ---
        
        print("Sorting combined data by uniqueID and iteration...")
        # Sort for consistency (important for lag features)
        aggregated_df = aggregated_df.sort_values(by=['uniqueID', 'iteration']).reset_index(drop=True)
        print("Sorting complete.")


        # --- Phase 2 Steps (Applied to Combined Data) ---

        # Step 3: Feature Engineering (Lag)
        feature_df = engineer_features(aggregated_df)
        del aggregated_df # Free memory
        gc.collect()

        # Step 4: Calculate Raw Reward Components
        reward_df = calculate_rewards(feature_df)
        del feature_df # Free memory
        gc.collect()

        # Step 5: Normalize Reward Components (REMOVED)
        # normalized_reward_df = normalize_rewards(reward_df)
        # del reward_df # Free memory
        # gc.collect()

        # Step 6: Combine Raw Rewards
        final_reward_df = combine_rewards(reward_df) 
        del reward_df # Free memory
        gc.collect()

        # --- ADDED Step 6.5: Scale Final Reward Globally --- 
        scaled_reward_df, _ = scale_final_reward(final_reward_df, args.output_dir)
        del final_reward_df # Free memory
        gc.collect()

        # Step 10: Assemble & Save MDPDataset / ReplayBuffer
        save_dataset(scaled_reward_df, args.output_dir, args.output_filename, args.algo_type)

        print("\n--- Dataset Preparation Script Finished Successfully ---")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback # Import traceback here
        traceback.print_exc() # Print stack trace for unexpected errors


if __name__ == "__main__":
    main() 