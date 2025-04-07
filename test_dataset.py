import pandas as pd
import numpy as np
import glob
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt # Restore plotting
import seaborn as sns # Restore plotting
import argparse # For command-line arguments

# --- Configuration ---
DATA_DIR = "../training_data"
EXPECTED_COLUMNS = [
    'uniqueID', 'designID', 'iteration', 'pin_count', 'net_count', 'drv',
    'wireLength', 'drc_weight', 'marker_weight', 'fixed_weight',
    'decay_weight', 'boxID', 'box_size', 'box_drv', 'L_N_box', 'L_N_drv',
    'R_N_box', 'R_N_drv'
]
# Columns expected to be constant within an iteration group
ITERATION_CONSTANT_COLS = [
    'designID', 'pin_count', 'net_count', 'drv', 'wireLength',
    'drc_weight', 'marker_weight', 'fixed_weight', 'decay_weight'
]
WEIGHT_COLS = ['drc_weight', 'marker_weight', 'fixed_weight', 'decay_weight']
COUNT_COLS = ['pin_count', 'net_count', 'box_drv', 'drv']
NEIGHBOR_DRV_COLS = ['L_N_drv', 'R_N_drv']
DRV_COLS_CHECK = ['drv', 'box_drv'] + NEIGHBOR_DRV_COLS
VIS_SAMPLE_RATE = 0.01 # Sample 1% of records for visualization to manage memory
PLOT_DIR = "analysis_plots" # Restore plot directory
STUCK_WINDOW = 2 # Number of previous iterations to check for being stuck
STUCK_PENALTY_VALUE = -50 # Example penalty value
CONVERGENCE_BONUS_VALUE = 100 # Example bonus value
TEXT_HIST_BINS = 20 # Number of bins for text histograms
TEXT_HIST_WIDTH = 60 # Max width for text histogram bars

# --- Helper Functions ---
def parse_box_size(size_str):
    if pd.isna(size_str) or not isinstance(size_str, str):
        return None, None
    match = re.match(r'(\d+(\.\d+)?)\s*x\s*(\d+(\.\d+)?)', size_str)
    if match:
        try:
            return float(match.group(1)), float(match.group(3))
        except ValueError:
            return None, None
    return None, None

def print_section_header(title):
    print("\n" + "="*15 + f" {title} " + "="*15)

def print_check_result(check_name, result, details=""):
    status = "PASS" if result else "FAIL"
    print(f"- {check_name}: {status}")
    if not result and details:
        print(f"  Details: {details}")

def safe_describe(series, title):
    print(f"\n- {title} Distribution Summary:")
    if series is not None and not series.empty:
        try:
            print(series.describe())
            # Additional stats for rewards
            if 'Reward' in title or 'Penalty' in title or 'Bonus' in title:
                 positive_rewards = (series > 1e-6).sum() # Use tolerance for float comparison
                 zero_rewards = (abs(series) < 1e-6).sum()
                 negative_rewards = (series < -1e-6).sum()
                 print(f"  Positive (>0): {positive_rewards}, Zero (~0): {zero_rewards}, Negative (<0): {negative_rewards}")
        except Exception as e:
            print(f"  Could not generate describe statistics: {e}")
    else:
        print("  No data collected.")

# --- Text Visualization Functions ---
def print_text_histogram(series, title, bins=TEXT_HIST_BINS, width=TEXT_HIST_WIDTH):
    print(f"\n- Text Histogram: {title}")
    if series is None or series.empty or series.isnull().all():
        print("  No valid data to generate histogram.")
        return
    
    try:
        # Drop NaNs before calculating histogram
        valid_series = series.dropna()
        if valid_series.empty:
            print("  No valid data after dropping NaNs.")
            return
            
        counts, bin_edges = np.histogram(valid_series, bins=bins)
        max_count = counts.max()
        if max_count == 0:
            print("  All bins are empty.")
            return

        print(f"  (Bin Width approx {(bin_edges[1]-bin_edges[0]):.2g}, Max Count {max_count})")
        for i in range(len(counts)):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i+1]
            count = counts[i]
            bar_width = int((count / max_count) * width) if max_count > 0 else 0
            bar = '#' * bar_width
            # Format bin range carefully
            bin_label = f"[{bin_start: >8.2g}, {bin_end: >8.2g})"
            if i == len(counts) - 1: # Include the right edge for the last bin
                 bin_label = f"[{bin_start: >8.2g}, {bin_end: >8.2g}]"
            
            print(f"  {bin_label} | {bar} ({count})")
            
    except Exception as e:
        print(f"  Error generating text histogram: {e}")

def print_correlation_matrix(df, cols, title):
    print(f"\n- {title}")
    if df is None or df.empty or not all(c in df.columns for c in cols):
        print("  Skipping correlation matrix: Missing columns or no data.")
        return
    try:
        correlation_matrix = df[cols].corr()
        # Increase display width for pandas output
        with pd.option_context('display.width', 120, 'display.max_columns', None):
             print(correlation_matrix)
    except Exception as e:
        print(f"  Error calculating/printing correlation matrix: {e}")

# --- Plotting Visualization Functions ---
def plot_distribution(series, title, filename, plot_dir):
    if series is None or series.empty:
        print(f"  Skipping plot {filename}: No data.")
        return
    plt.figure(figsize=(10, 6))
    sns.histplot(series, kde=True, bins=50)
    plt.title(f'Distribution of {title}')
    plt.xlabel(title)
    plt.ylabel('Frequency')
    filepath = os.path.join(plot_dir, filename)
    try:
        plt.savefig(filepath)
        print(f"  Saved plot: {filepath}")
    except Exception as e:
        print(f"  Error saving plot {filepath}: {e}")
    plt.close()

def plot_correlation_matrix(df, cols, title, filename, plot_dir):
    if df is None or df.empty or not all(c in df.columns for c in cols):
        print(f"  Skipping plot {filename}: Missing columns or no data.")
        return
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    filepath = os.path.join(plot_dir, filename)
    try:
        plt.savefig(filepath)
        print(f"  Saved plot: {filepath}")
    except Exception as e:
        print(f"  Error saving plot {filepath}: {e}")
    plt.close()

# --- Main Test Function ---
def run_dataset_tests(data_dir, sections_to_run):
    # Create plot directory if it doesn't exist
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
        print(f"Created plot directory: {PLOT_DIR}")

    print_section_header("Dataset Test Report (Processing File-by-File)")
    all_files = glob.glob(os.path.join(data_dir, "trainingdata_*.csv"))
    
    # --- Initialize Accumulators --- 
    # (Moved initialization outside section checks)
    total_records = 0
    all_headers_consistent = True
    files_with_header_issues = []
    files_with_read_issues = []
    overall_missing_values = defaultdict(int)
    overall_inconsistent_iterations = 0
    overall_duplicate_box_ids = 0
    overall_drv_sum_mismatches = 0
    all_value_ranges = {col: {'min': np.inf, 'max': -np.inf} for col in EXPECTED_COLUMNS}
    all_counts_negative = defaultdict(int)
    all_weights_invalid = 0
    all_drvs_negative = defaultdict(int)
    all_wl_invalid = 0
    all_box_size_invalid = 0
    unique_to_design_map = {} # For cross-file check
    design_to_metrics_map = defaultdict(lambda: {'pin_counts': set(), 'net_counts': set()}) # For cross-file check

    all_primary_rewards = []
    all_box_rewards = []
    all_stuck_penalties = []
    all_convergence_bonuses = []
    all_max_box_drv_rewards = []
    all_num_violating_rewards = []
    all_trajectory_lengths = []
    sampled_data_for_vis = [] # Still collect sampled data for potential correlation analysis

    all_terminated_zero_drv = set()
    all_unique_ids = set()
    
    run_all = 'all' in sections_to_run
    run_sec_1_4 = run_all or '1-4' in sections_to_run
    run_sec_5 = run_all or '5' in sections_to_run

    if run_sec_1_4:
        print_section_header("1. File Level Checks")
        print(f"Found {len(all_files)} CSV files in {data_dir}.")
        print_check_result("File Existence", len(all_files) > 0, f"No files found matching trainingdata_*.csv in {data_dir}")
        if not all_files:
            return # Stop if no files found

    # --- Process files one by one ---
    # (Loop needs to run if either 1-4 or 5 needs data)
    if run_sec_1_4 or run_sec_5:
        for i, f in enumerate(all_files):
            filename = os.path.basename(f)
            print(f"\n--- Processing file {i+1}/{len(all_files)}: {filename} ---")
            try:
                # Load full file for aggregation and reward calculation
                df = pd.read_csv(f)
                total_records += len(df)
                all_unique_ids.update(df['uniqueID'].unique())

                # Sample data for visualization BEFORE modifying df
                if VIS_SAMPLE_RATE > 0 and len(df) > 0: # Added check for empty df
                     sample_size = max(1, int(len(df) * VIS_SAMPLE_RATE)) # Ensure at least 1 row if possible
                     sampled_data_for_vis.append(df.sample(n=sample_size, random_state=42))

                if run_sec_1_4:
                    # 1. Header Check
                    if list(df.columns) != EXPECTED_COLUMNS:
                        all_headers_consistent = False
                        files_with_header_issues.append(filename)
                        print(f"  FAIL: Header mismatch.")
                        # Don't continue if header fails, as other checks depend on it
                        # However, if only running section 5, we might want to proceed?
                        # For simplicity, let's skip file if header fails
                        continue
                    else:
                        print(f"  PASS: Header consistent.")

                    # --- Preprocessing for checks ---
                    # (Needed for both section groups)
                    numeric_cols = ['L_N_drv', 'R_N_drv', 'box_drv', 'drv', 'wireLength'] + WEIGHT_COLS + ['pin_count', 'net_count']
                    for col in numeric_cols:
                         if col in df.columns:
                             df[col] = pd.to_numeric(df[col], errors='coerce')

                    # --- Apply Data Cleaning Based on Plan ---
                    # (Needed for both section groups)
                    bad_iterations = [
                        ('aes_cipher_top_run_46', 2), ('bp_be_top_run_69', 8),
                        ('ispd18_test1_run_12', 1), ('ispd18_test3_run_14', 11),
                        ('ispd18_test3_run_30', 16), ('ispd18_test4_run_17', 6)
                    ]
                    initial_rows = len(df)
                    for uid, it in bad_iterations:
                        df = df[~((df['uniqueID'] == uid) & (df['iteration'] == it))]
                    rows_filtered = initial_rows - len(df)
                    if rows_filtered > 0:
                        print(f"  INFO: Filtered {rows_filtered} rows due to known missing values.")

                    if filename == 'trainingdata_bp_be_base.csv':
                        dupe_rows_before = len(df[df.duplicated(subset=['uniqueID', 'iteration', 'boxID'], keep=False)])
                        if dupe_rows_before > 0:
                            df = df.drop_duplicates(subset=['uniqueID', 'iteration', 'boxID'], keep='first')
                            print(f"  INFO: Handled duplicate boxIDs by keeping first occurrence (affected {dupe_rows_before} rows).")
                    # --- End Data Cleaning ---

                    # --- Aggregation Logic (Needed for Sec 5, useful for Sec 2) ---
                    aggregated_iterations = {}
                    if not df.empty:
                        try:
                            grouped = df.groupby(['uniqueID', 'iteration'])
                            for name, group in grouped:
                                # Check consistency within group if running Sec 1-4
                                # NOTE: This consistency check needs to happen *before* aggregation if
                                # we want to be precise about which rows were inconsistent.
                                # Doing it here on the `group` is less precise but simpler.
                                if run_sec_1_4:
                                    if not group[ITERATION_CONSTANT_COLS].apply(lambda x: x.nunique() <= 1).all():
                                        overall_inconsistent_iterations += 1
                                    if group['boxID'].duplicated().any():
                                        overall_duplicate_box_ids += 1

                                # Aggregate data
                                iter_data = {}
                                first_row = group.iloc[0]
                                iter_data['drv'] = first_row['drv']
                                iter_data['total_box_drv'] = group['box_drv'].sum()
                                iter_data['max_box_drv'] = group['box_drv'].max()
                                iter_data['num_violating'] = (group['box_drv'] > 0).sum()
                                iter_data['pin_count'] = first_row['pin_count']
                                iter_data['net_count'] = first_row['net_count']
                                aggregated_iterations[name] = iter_data

                                # DRV Sum Mismatch Check (if running Sec 1-4)
                                if run_sec_1_4:
                                    iteration_drv = iter_data['drv']
                                    sum_box_drv = iter_data['total_box_drv']
                                    if not pd.isna(iteration_drv) and not pd.isna(sum_box_drv) and not np.isclose(iteration_drv, sum_box_drv, atol=1e-5):
                                        overall_drv_sum_mismatches += 1
                        except KeyError as e:
                            print(f"    ERROR: Grouping/Aggregation failed: {e}")
                            # If aggregation fails, we cannot calculate rewards, might need to skip RL checks for this file
                            aggregated_iterations = {} # Reset

                # --- Run Sec 1-4 Checks (if requested) ---
                if run_sec_1_4:
                    print("  Section 2: Data Integrity & Consistency Checks...")
                    # Missing Values Check (on cleaned data)
                    cols_to_check_na = [c for c in df.columns if c not in ['L_N_box', 'L_N_drv', 'R_N_box', 'R_N_drv']]
                    missing_values = df[cols_to_check_na].isnull().sum()
                    missing_values = missing_values[missing_values > 0]
                    if len(missing_values) > 0:
                        print(f"    WARN: Unexpected Missing Values found AFTER cleaning: \n{missing_values}")
                        for col, count in missing_values.items(): overall_missing_values[col] += count

                    # Report counts calculated during aggregation
                    print(f"    Iteration Consistency Issues (reported above during aggregation)")
                    print(f"    Duplicate Box IDs AFTER handling (reported above during aggregation)")
                    print(f"    DRV Sum Mismatches (reported above during aggregation)")

                    # Data Range & Validity Checks
                    print("  Section 3: Data Range & Validity Checks...")
                    # ... (Checks need to be careful about df potentially being empty after filtering)
                    if not df.empty:
                        # Iteration Positive Ints
                        # Check for non-positive OR non-integer. Ensure 'iteration' column exists.
                        if 'iteration' in df.columns and (not (df['iteration'] > 0).all() or not pd.api.types.is_integer_dtype(df['iteration'].dropna())):
                             print(f"    WARN: Iteration numbers issue (non-positive or non-integer).")
                        # Counts Non-Negative
                        for col in COUNT_COLS:
                            if col in df.columns and (df[col].dropna() < 0).any():
                                count = (df[col].dropna() < 0).sum() # Calculate on dropped NA
                                all_counts_negative[col] += count
                                print(f"    WARN: Negative values found in {col} ({count})")
                        # Weights Validity
                        weights_invalid = df[WEIGHT_COLS].isnull().sum().sum() + np.isinf(df[WEIGHT_COLS].to_numpy()).sum()
                        if weights_invalid > 0:
                            all_weights_invalid += weights_invalid
                            print(f"    WARN: NaN/Inf values found in weights ({weights_invalid})")
                        # DRV Values Non-Negative
                        for col in DRV_COLS_CHECK:
                             if col in df.columns and (df[col].dropna() < 0).any():
                                count = (df[col].dropna() < 0).sum() # Calculate on dropped NA
                                all_drvs_negative[col] += count
                                print(f"    WARN: Negative values found in {col} ({count})")
                        # Wire Length Check (after cleaning)
                        wl_invalid_count = 0
                        if 'wireLength' in df.columns:
                             wl_invalid_count = (df['wireLength'] < 0).sum() + df['wireLength'].isnull().sum()
                        if wl_invalid_count > 0:
                            all_wl_invalid += wl_invalid_count
                            print(f"    WARN: Negative/NaN wireLength AFTER cleaning ({wl_invalid_count}).")
                        # Box Size Check
                        try:
                            if 'box_size' in df.columns:
                                box_dims = df['box_size'].apply(parse_box_size)
                                box_width = box_dims.apply(lambda x: x[0] if x else None) # Handle None from parse
                                box_height = box_dims.apply(lambda x: x[1] if x else None) # Handle None from parse
                                # Count invalid where parse failed (None) or dim <= 0
                                invalid_count = box_dims.isnull().sum() + (box_width.dropna() <= 0).sum() + (box_height.dropna() <= 0).sum()
                                if invalid_count > 0:
                                    all_box_size_invalid += invalid_count
                                    print(f"    WARN: Invalid box sizes found ({invalid_count})")
                            else:
                                print("    WARN: box_size column not found.")
                        except Exception as e:
                            print(f"    ERROR: Parsing box_size failed: {e}")
                            all_box_size_invalid += len(df) # Assume all failed if error
                        # Update overall ranges
                        for col in df.select_dtypes(include=np.number).columns:
                            if col in all_value_ranges and not df[col].isnull().all():
                                try:
                                    all_value_ranges[col]['min'] = min(all_value_ranges[col]['min'], df[col].min(skipna=True))
                                    all_value_ranges[col]['max'] = max(all_value_ranges[col]['max'], df[col].max(skipna=True))
                                except TypeError:
                                    print(f"    WARN: Could not compute min/max for column {col}.")
                    else:
                        print("  Skipping Range/Validity checks as DataFrame is empty after cleaning.")

                    # Cross-File Checks Accumulation
                    print("  Section 4: Cross-File Checks (Accumulating)...")
                    if not df.empty:
                        for unique_id, design_id in df[['uniqueID', 'designID']].drop_duplicates().values:
                            if unique_id in unique_to_design_map and unique_to_design_map[unique_id] != design_id:
                                print(f"    ERROR: Inconsistent designID for {unique_id}!")
                            unique_to_design_map[unique_id] = design_id
                            # Get pin/net counts safely
                            pin_net_df = df.loc[df['uniqueID'] == unique_id, ['pin_count', 'net_count']]
                            if not pin_net_df.empty:
                                current_pin = pin_net_df['pin_count'].iloc[0]
                                current_net = pin_net_df['net_count'].iloc[0]
                                if pd.notna(current_pin): design_to_metrics_map[design_id]['pin_counts'].add(current_pin)
                                if pd.notna(current_net): design_to_metrics_map[design_id]['net_counts'].add(current_net)

                # --- Calculate Rewards & Accumulate RL Data (if needed) ---
                if run_sec_5:
                    print("  Section 5: Calculating Rewards & Accumulating RL Data...")
                    agg_df = pd.DataFrame.from_dict(aggregated_iterations, orient='index')
                    if not agg_df.empty:
                        try: # Wrap reward calculation in try-except
                            agg_df.index = pd.MultiIndex.from_tuples(agg_df.index, names=['uniqueID', 'iteration'])
                            agg_df = agg_df.sort_index()

                            # Calculate next state values
                            agg_df['next_drv'] = agg_df.groupby('uniqueID')['drv'].shift(-1)
                            agg_df['next_total_box_drv'] = agg_df.groupby('uniqueID')['total_box_drv'].shift(-1)
                            agg_df['next_max_box_drv'] = agg_df.groupby('uniqueID')['max_box_drv'].shift(-1)
                            agg_df['next_num_violating'] = agg_df.groupby('uniqueID')['num_violating'].shift(-1)

                            # Reward components
                            agg_df['primary_reward'] = agg_df['drv'] - agg_df['next_drv']
                            agg_df['box_reward'] = agg_df['total_box_drv'] - agg_df['next_total_box_drv']
                            agg_df['max_box_drv_reward'] = agg_df['max_box_drv'] - agg_df['next_max_box_drv']
                            agg_df['num_violating_reward'] = agg_df['num_violating'] - agg_df['next_num_violating']
                            agg_df['stuck_penalty'] = 0.0
                            for k in range(1, STUCK_WINDOW + 1):
                                agg_df[f'drv_lag_{k}'] = agg_df.groupby('uniqueID')['drv'].shift(k)
                            is_stuck = pd.Series(True, index=agg_df.index)
                            for k in range(STUCK_WINDOW):
                                current_drv_col = 'drv' if k == 0 else f'drv_lag_{k}'
                                prev_drv_col = f'drv_lag_{k+1}'
                                # Check if both columns exist before comparison
                                if current_drv_col in agg_df and prev_drv_col in agg_df:
                                    is_stuck = is_stuck & (agg_df[current_drv_col] >= agg_df[prev_drv_col]).fillna(False)
                                else:
                                    is_stuck = pd.Series(False, index=agg_df.index) # Cannot be stuck if lag doesn't exist
                            agg_df.loc[is_stuck, 'stuck_penalty'] = STUCK_PENALTY_VALUE
                            # Clean up lag columns after use
                            for k in range(1, STUCK_WINDOW + 1):
                                if f'drv_lag_{k}' in agg_df.columns:
                                    del agg_df[f'drv_lag_{k}']

                            agg_df['convergence_bonus'] = (agg_df['next_drv'] == 0).astype(float) * CONVERGENCE_BONUS_VALUE

                            # Accumulate (ensure columns exist before extending)
                            if 'primary_reward' in agg_df: all_primary_rewards.extend(agg_df['primary_reward'].dropna().tolist())
                            if 'box_reward' in agg_df: all_box_rewards.extend(agg_df['box_reward'].dropna().tolist())
                            if 'max_box_drv_reward' in agg_df: all_max_box_drv_rewards.extend(agg_df['max_box_drv_reward'].dropna().tolist())
                            if 'num_violating_reward' in agg_df: all_num_violating_rewards.extend(agg_df['num_violating_reward'].dropna().tolist())
                            if 'stuck_penalty' in agg_df: all_stuck_penalties.extend(agg_df['stuck_penalty'].dropna().tolist())
                            if 'convergence_bonus' in agg_df: all_convergence_bonuses.extend(agg_df['convergence_bonus'].dropna().tolist())

                            # Calculate trajectory lengths and terminations
                            file_traj_lengths = agg_df.reset_index().groupby('uniqueID')['iteration'].max()
                            all_trajectory_lengths.extend(file_traj_lengths.tolist())
                            if 'next_drv' in agg_df:
                                terminated_ids = agg_df[agg_df['next_drv'] == 0].reset_index()['uniqueID'].unique()
                                all_terminated_zero_drv.update(terminated_ids)
                        except Exception as e_reward:
                            print(f"    ERROR: Calculating rewards failed for file {filename}: {e_reward}")

                    else:
                        print("  No aggregated data for this file. Skipping reward calculation.")

            # Added except block for the outer try
            except Exception as e_file:
                files_with_read_issues.append(f"{filename}: {e_file}")
                print(f"  ERROR: Failed processing file {filename}: {e_file}")
                # Continue to the next file if one fails

    # --- Combine sampled data ---
    # (Do this regardless of sections run, for potential correlation matrix)
    vis_df = None
    if sampled_data_for_vis:
        try:
            vis_df = pd.concat(sampled_data_for_vis, ignore_index=True)
            print(f"\nCombined {len(vis_df)} sampled records.")
            # Ensure numeric types for vis_df
            for col in numeric_cols: # Use numeric_cols defined earlier
                if col in vis_df.columns:
                     vis_df[col] = pd.to_numeric(vis_df[col], errors='coerce')
        except Exception as e_concat:
            print(f"  ERROR: Failed to combine sampled data: {e_concat}")
            vis_df = None # Ensure vis_df is None if concat fails

    # --- Print Final Summary Report ---
    print_section_header("Overall Summary Report")

    if run_sec_1_4:
        print_section_header("1. File Level Checks Summary")
        print(f"Total files scanned: {len(all_files)}")
        print(f"Total records loaded: {total_records}")
        print_check_result("File Readability", not files_with_read_issues, f"Issues reading files: {files_with_read_issues}")
        print_check_result("Header Consistency", all_headers_consistent, f"Inconsistent headers in files: {files_with_header_issues}")

        print_section_header("2. Data Integrity & Consistency Summary")
        if any(overall_missing_values.values()):
            print_check_result("Unexpected Missing Values", False, f"Total AFTER cleaning: {dict(overall_missing_values)}")
        else:
            print_check_result("Unexpected Missing Values", True)
        print_check_result("Iteration Column Consistency", overall_inconsistent_iterations == 0, f"Total inconsistent iterations: {overall_inconsistent_iterations}")
        print_check_result("Box ID Uniqueness per Iteration", overall_duplicate_box_ids == 0, f"Total duplicate Box IDs AFTER handling: {overall_duplicate_box_ids}")
        print_check_result("Iteration DRV vs Sum(Box DRV)", overall_drv_sum_mismatches == 0, f"Total DRV sum mismatches found: {overall_drv_sum_mismatches}")

        print_section_header("3. Data Range & Validity Checks Summary")
        print("- Overall Value Ranges (Min/Max):")
        for col, ranges in all_value_ranges.items():
            if ranges['min'] != np.inf and ranges['max'] != -np.inf:
                print(f"    {col}: [{ranges['min']}, {ranges['max']}]")
        print_check_result("Counts Non-Negative", sum(all_counts_negative.values())==0, f"Negative counts found: {dict(all_counts_negative)}")
        print_check_result("Weights Validity (No NaN/Inf)", all_weights_invalid == 0, f"Total NaN/Inf values found in weights: {all_weights_invalid}")
        print_check_result("DRV Values Non-Negative", sum(all_drvs_negative.values())==0, f"Negative DRVs found: {dict(all_drvs_negative)}")
        print_check_result("Wire Length Non-Negative & Valid", all_wl_invalid == 0, f"Total negative or NaN wireLength AFTER cleaning: {all_wl_invalid}")
        print_check_result("Box Size Validity", all_box_size_invalid == 0, f"Total invalid box sizes: {all_box_size_invalid}")

        print_section_header("4. Cross-File / Run Checks Summary")
        print("- uniqueID to designID consistency checked during file processing.")
        inconsistent_designs = {}
        for design, metrics in design_to_metrics_map.items():
            pin_counts_valid = {c for c in metrics['pin_counts'] if pd.notna(c)}
            net_counts_valid = {c for c in metrics['net_counts'] if pd.notna(c)}
            if len(pin_counts_valid) > 1 or len(net_counts_valid) > 1:
                inconsistent_designs[design] = f"Pin#: {len(pin_counts_valid)}, Net#: {len(net_counts_valid)}"
        print_check_result("Design Metrics Consistency per designID", not inconsistent_designs, f"Inconsistent pin/net counts found: {inconsistent_designs}")

    if run_sec_5:
        print_section_header("5. RL Suitability Checks (Overall Summary)")
        # Reward Analysis
        safe_describe(pd.Series(all_primary_rewards) if all_primary_rewards else None, "Primary Reward (drv_t-1 - drv_t)")
        safe_describe(pd.Series(all_box_rewards) if all_box_rewards else None, "Box Reward (total_box_drv_t-1 - total_box_drv_t)")
        safe_describe(pd.Series(all_max_box_drv_rewards) if all_max_box_drv_rewards else None, "Max Box DRV Reward (max_t-1 - max_t)")
        safe_describe(pd.Series(all_num_violating_rewards) if all_num_violating_rewards else None, "Num Violating Reward (#violating_t-1 - #violating_t)")
        safe_describe(pd.Series(all_stuck_penalties) if all_stuck_penalties else None, "Stuck Penalty")
        safe_describe(pd.Series(all_convergence_bonuses) if all_convergence_bonuses else None, "Convergence Bonus")
        # Trajectory Analysis
        safe_describe(pd.Series(all_trajectory_lengths) if all_trajectory_lengths else None, "Trajectory Length")
        total_trajectories = len(all_unique_ids)
        print(f"\n- Total unique trajectories (uniqueID): {total_trajectories}")
        if total_trajectories > 0:
            percentage = (len(all_terminated_zero_drv)/total_trajectories*100)
            print(f"- Trajectories reaching DRV=0: {len(all_terminated_zero_drv)} / {total_trajectories} ({percentage:.1f}%)")
        else:
            print(f"- Trajectories reaching DRV=0: 0 / 0 (0.0%)")
        # Data Density
        approx_transitions = len(all_primary_rewards) # Use primary rewards as proxy for transitions
        print(f"\n- Data Density:")
        print(f"  - Total records (box-level, initial): {total_records}") # Report initial count
        print(f"  - Approx. valid transitions calculated: {approx_transitions}")

        # --- Generate Text AND Plot Visualizations ---
        # (Plot generation added back)
        print_section_header("6. Text & Plot Visualizations (from sampled data)")
        if vis_df is not None and not vis_df.empty:
            # --- Text Output ---
            print("   --- Text Histograms --- ")
            # Weight Distributions
            for col in WEIGHT_COLS:
                print_text_histogram(vis_df[col].dropna(), f'Sampled {col}')
            # Weight Correlations
            print_correlation_matrix(vis_df, WEIGHT_COLS, 'Sampled Weight Correlation Matrix')
            # State Feature Distributions
            print_text_histogram(vis_df['drv'].dropna(), 'Sampled Iteration DRV')
            print_text_histogram(vis_df['wireLength'].dropna(), 'Sampled Wire Length')
            print_text_histogram(vis_df['iteration'].dropna(), 'Sampled Iteration Number')
            # Reward Distributions (using overall accumulated rewards)
            print_text_histogram(pd.Series(all_primary_rewards) if all_primary_rewards else None, 'Primary Reward')
            print_text_histogram(pd.Series(all_box_rewards) if all_box_rewards else None, 'Box Reward')
            print_text_histogram(pd.Series(all_max_box_drv_rewards) if all_max_box_drv_rewards else None, 'Max Box DRV Reward')
            print_text_histogram(pd.Series(all_num_violating_rewards) if all_num_violating_rewards else None, 'Num Violating Reward')
            print_text_histogram(pd.Series(all_stuck_penalties) if all_stuck_penalties else None, 'Stuck Penalty')
            print_text_histogram(pd.Series(all_convergence_bonuses) if all_convergence_bonuses else None, 'Convergence Bonus')
            # Trajectory Length Distribution
            print_text_histogram(pd.Series(all_trajectory_lengths) if all_trajectory_lengths else None, 'Trajectory Length')

            # --- Plot Output ---
            print("\n   --- Plot Generation --- ")
            # Weight Distributions
            for col in WEIGHT_COLS:
                plot_distribution(vis_df[col].dropna(), f'Sampled {col}', f'dist_{col}.png', PLOT_DIR)
            # Weight Correlations
            plot_correlation_matrix(vis_df, WEIGHT_COLS, 'Sampled Weight Correlation Matrix', 'corr_weights.png', PLOT_DIR)
            # State Feature Distributions
            plot_distribution(vis_df['drv'].dropna(), 'Sampled Iteration DRV', 'dist_drv.png', PLOT_DIR)
            plot_distribution(vis_df['wireLength'].dropna(), 'Sampled Wire Length', 'dist_wirelength.png', PLOT_DIR)
            plot_distribution(vis_df['iteration'].dropna(), 'Sampled Iteration Number', 'dist_iteration.png', PLOT_DIR)
            # Reward Distributions (using overall accumulated rewards)
            plot_distribution(pd.Series(all_primary_rewards) if all_primary_rewards else None, 'Primary Reward', 'dist_reward_primary.png', PLOT_DIR)
            plot_distribution(pd.Series(all_box_rewards) if all_box_rewards else None, 'Box Reward', 'dist_reward_box.png', PLOT_DIR)
            plot_distribution(pd.Series(all_max_box_drv_rewards) if all_max_box_drv_rewards else None, 'Max Box DRV Reward', 'dist_reward_max_box_drv.png', PLOT_DIR)
            plot_distribution(pd.Series(all_num_violating_rewards) if all_num_violating_rewards else None, 'Num Violating Reward', 'dist_reward_num_violating.png', PLOT_DIR)
            plot_distribution(pd.Series(all_stuck_penalties) if all_stuck_penalties else None, 'Stuck Penalty', 'dist_reward_stuck.png', PLOT_DIR)
            plot_distribution(pd.Series(all_convergence_bonuses) if all_convergence_bonuses else None, 'Convergence Bonus', 'dist_reward_convergence.png', PLOT_DIR)
            # Trajectory Length Distribution
            plot_distribution(pd.Series(all_trajectory_lengths) if all_trajectory_lengths else None, 'Trajectory Length', 'dist_traj_length.png', PLOT_DIR)
        else: # Correctly indented else for the outer `if vis_df is not None`
            print("  Skipping text and plot visualizations: No sampled data available.")

# --- Run Tests ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run dataset tests.')
    parser.add_argument('--run_section', type=str, default='all', choices=['all', '1-4', '5'],
                        help='Specify which section(s) to run: "all", "1-4" (Checks), or "5" (RL Suitability)')
    args = parser.parse_args()
    
    run_dataset_tests(DATA_DIR, sections_to_run=args.run_section) 