import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

# --- Configuration ---
DEFAULT_DATA_DIR = "../training_data"
DEFAULT_PLOT_DIR = "analysis_plots/top_runs"
WEIGHT_COLS = ['drc_weight', 'marker_weight', 'fixed_weight', 'decay_weight']
ITERATION_COL = 'iteration'
RUN_ID_COL = 'uniqueID'
DESIGN_ID_COL = 'designID'
TOP_PERCENTILE = 0.20 # Top 20% shortest runs

# Known bad iterations to filter out
BAD_ITERATIONS = [
    ('aes_cipher_top_run_46', 2), ('bp_be_top_run_69', 8),
    ('ispd18_test1_run_12', 1), ('ispd18_test3_run_14', 11),
    ('ispd18_test3_run_30', 16), ('ispd18_test4_run_17', 6)
]

def load_and_clean_data(data_dir):
    """Loads all CSVs, cleans known bad iterations."""
    all_files = glob.glob(os.path.join(data_dir, "trainingdata_*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    df_list = []
    print(f"Loading {len(all_files)} files...")
    for f in all_files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"Warning: Could not read file {f}: {e}")
    
    if not df_list:
        raise ValueError("No data could be loaded.")
        
    df_raw = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df_raw)} total records.")

    # Convert relevant columns to numeric, coercing errors
    for col in WEIGHT_COLS + [ITERATION_COL]:
        if col in df_raw.columns:
             df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    # Filter known bad iterations
    initial_rows = len(df_raw)
    for uid, it in BAD_ITERATIONS:
        df_raw = df_raw[~((df_raw[RUN_ID_COL] == uid) & (df_raw[ITERATION_COL] == it))]
    rows_filtered = initial_rows - len(df_raw)
    if rows_filtered > 0:
        print(f"Filtered {rows_filtered} rows due to known bad iterations.")

    # Drop rows where essential columns might be NaN after coercion or initial load
    df_raw = df_raw.dropna(subset=[RUN_ID_COL, DESIGN_ID_COL, ITERATION_COL] + WEIGHT_COLS)
    print(f"Data ready with {len(df_raw)} records after cleaning NAs.")
    return df_raw

def find_top_runs(df):
    """Identifies the top X% shortest runs for each design and returns their IDs and lengths."""
    # Get trajectory length (max iteration) for each run
    run_lengths = df.groupby(RUN_ID_COL)[ITERATION_COL].max().reset_index()
    run_lengths = run_lengths.rename(columns={ITERATION_COL: 'trajectory_length'})

    # Get design ID for each run
    design_map = df[[RUN_ID_COL, DESIGN_ID_COL]].drop_duplicates()

    # Merge lengths and designs
    run_info = pd.merge(run_lengths, design_map, on=RUN_ID_COL)

    top_run_ids = []
    print(f"Identifying top {TOP_PERCENTILE*100:.0f}% runs per design...")
    for design_id, group in run_info.groupby(DESIGN_ID_COL):
        if len(group) == 0: continue
        # Calculate percentile threshold for length
        length_threshold = group['trajectory_length'].quantile(TOP_PERCENTILE)
        # Handle cases where all runs have the same length (threshold might exclude all if not inclusive)
        # Select runs with length <= threshold
        top_runs_for_design = group[group['trajectory_length'] <= length_threshold]
        
        # If percentile calculation resulted in 0 runs (e.g., only 1 run for design)
        # default to selecting the shortest run(s).
        if len(top_runs_for_design) == 0 and len(group) > 0:
             min_length = group['trajectory_length'].min()
             top_runs_for_design = group[group['trajectory_length'] == min_length]
             
        top_run_ids.extend(top_runs_for_design[RUN_ID_COL].tolist())
        print(f"  Design '{design_id}': Found {len(group)} runs. Threshold length <= {length_threshold:.2f}. Selected {len(top_runs_for_design)} top runs.")

    top_run_ids_set = set(top_run_ids)
    # Filter run_info to contain only the lengths of the selected top runs
    top_run_lengths = run_info[run_info[RUN_ID_COL].isin(top_run_ids_set)][[RUN_ID_COL, 'trajectory_length']]

    print(f"Total top runs selected across all designs: {len(top_run_ids_set)}")
    return top_run_ids_set, top_run_lengths

def analyze_and_plot_weights(df_top_runs, plot_dir):
    """Analyzes weights per iteration for top runs and plots results."""
    if df_top_runs.empty:
        print("No data from top runs to analyze.")
        return None

    # Calculate mean weights per iteration
    # Ensure iteration is treated numerically for grouping/aggregation
    df_top_runs[ITERATION_COL] = pd.to_numeric(df_top_runs[ITERATION_COL], errors='coerce')
    df_top_runs = df_top_runs.dropna(subset=[ITERATION_COL])
    df_top_runs[ITERATION_COL] = df_top_runs[ITERATION_COL].astype(int)

    # Group by iteration and calculate mean for each weight
    recommended_weights = df_top_runs.groupby(ITERATION_COL)[WEIGHT_COLS].mean().reset_index()

    print("Recommended Weights (Mean from Top Runs) per Iteration:")
    print(recommended_weights.round(4).to_string(index=False))

    # --- Plotting ---
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {plot_path.absolute()}")

    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
    
    # Plot each weight's mean value vs iteration
    fig, axes = plt.subplots(len(WEIGHT_COLS), 1, figsize=(12, 6 * len(WEIGHT_COLS)), sharex=True)
    if len(WEIGHT_COLS) == 1: # Handle case with only one weight column
        axes = [axes]

    for i, weight_col in enumerate(WEIGHT_COLS):
        sns.lineplot(data=recommended_weights, x=ITERATION_COL, y=weight_col, ax=axes[i], marker='o', label=f'Mean {weight_col}')
        # Optional: Add shaded area for std dev if desired (requires calculating std dev)
        # std_devs = df_top_runs.groupby(ITERATION_COL)[weight_col].std().reset_index()
        # merged = pd.merge(recommended_weights[[ITERATION_COL, weight_col]], std_devs, on=ITERATION_COL)
        # axes[i].fill_between(merged[ITERATION_COL], merged[weight_col] - merged[f'{weight_col}_y'], merged[weight_col] + merged[f'{weight_col}_y'], alpha=0.2, label=f'Std Dev {weight_col}')
        
        axes[i].set_title(f'Mean {weight_col.replace("_", " ").title()} vs. Iteration (Top {TOP_PERCENTILE*100:.0f}% Runs)')
        axes[i].set_ylabel('Mean Weight Value')
        axes[i].grid(True)
        axes[i].legend()

    axes[-1].set_xlabel('Iteration Number')
    plt.tight_layout()
    plot_filename = plot_path / 'mean_weights_vs_iteration_top_runs.png'
    try:
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved plot: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot {plot_filename}: {e}")
    plt.close(fig)

    # Optional: Box plot of weights per iteration (can be very wide)
    # Consider plotting only first N iterations if too many
    max_iter_for_boxplot = recommended_weights[ITERATION_COL].max()
    plot_iter_limit = min(max_iter_for_boxplot, 20) # Limit boxplot iterations if too many
    
    df_plot_subset = df_top_runs[df_top_runs[ITERATION_COL] <= plot_iter_limit]

    if not df_plot_subset.empty:
        fig, axes = plt.subplots(len(WEIGHT_COLS), 1, figsize=(15, 6 * len(WEIGHT_COLS)), sharex=True)
        if len(WEIGHT_COLS) == 1:
            axes = [axes]
            
        for i, weight_col in enumerate(WEIGHT_COLS):
            sns.boxplot(data=df_plot_subset, x=ITERATION_COL, y=weight_col, ax=axes[i], palette='viridis')
            axes[i].set_title(f'{weight_col.replace("_", " ").title()} Distribution per Iteration (Top Runs, Iter 1-{plot_iter_limit})')
            axes[i].set_ylabel('Weight Value')
            axes[i].grid(True, axis='y')

        axes[-1].set_xlabel(f'Iteration Number (up to {plot_iter_limit})')
        plt.tight_layout()
        plot_filename = plot_path / f'boxplot_weights_per_iteration_top_runs_lim{plot_iter_limit}.png'
        try:
            plt.savefig(plot_filename, dpi=300)
            print(f"Saved plot: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        plt.close(fig)
    else:
        print(f"Skipping boxplots as no data available within iteration limit {plot_iter_limit}.")

    return recommended_weights

def plot_tail_weight_trend(df_top_runs_with_len, weight_col, plot_dir, tail_length=5):
    """Plots the trend of a specific weight in the last few iterations."""
    if df_top_runs_with_len.empty:
        print(f"No data from top runs to analyze for tail behavior of {weight_col}.")
        return
    if 'trajectory_length' not in df_top_runs_with_len.columns:
        print(f"Error: trajectory_length column missing. Cannot plot tail behavior for {weight_col}.")
        return
    if weight_col not in df_top_runs_with_len.columns:
        print(f"Error: weight column '{weight_col}' not found. Cannot plot tail behavior.")
        return

    print(f"\nAnalyzing {weight_col} trend for the last {tail_length} iterations...")

    # Calculate iterations from the end (0 = last iteration, 1 = second last, etc.)
    # Ensure columns are numeric before subtraction
    df_top_runs_with_len['trajectory_length'] = pd.to_numeric(df_top_runs_with_len['trajectory_length'], errors='coerce')
    df_top_runs_with_len[ITERATION_COL] = pd.to_numeric(df_top_runs_with_len[ITERATION_COL], errors='coerce')
    df_top_runs_with_len = df_top_runs_with_len.dropna(subset=['trajectory_length', ITERATION_COL]) # Drop rows where conversion failed

    df_top_runs_with_len['iterations_from_end'] = df_top_runs_with_len['trajectory_length'] - df_top_runs_with_len[ITERATION_COL]
    # Ensure iterations_from_end is integer
    df_top_runs_with_len['iterations_from_end'] = df_top_runs_with_len['iterations_from_end'].astype(int)

    # Filter for the tail
    df_tail = df_top_runs_with_len[df_top_runs_with_len['iterations_from_end'] < tail_length].copy()

    if df_tail.empty:
        print(f"No data found within the last {tail_length} iterations.")
        return

    # Calculate mean weight per iteration from the end
    tail_mean_weights = df_tail.groupby('iterations_from_end')[weight_col].mean().reset_index()
    # Sort by iterations_from_end for plotting correctly
    tail_mean_weights = tail_mean_weights.sort_values('iterations_from_end')

    print(f"Mean {weight_col} during tail iterations:")
    print(tail_mean_weights.round(4).to_string(index=False))

    # --- Plotting ---
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True) # Ensure plot dir exists

    plt.style.use('seaborn-v0_8-darkgrid')

    # Line plot of mean weight vs. Iterations From End
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.lineplot(data=tail_mean_weights, x='iterations_from_end', y=weight_col, marker='o', ax=ax)
    ax.set_title(f'Mean {weight_col.replace("_"," ").title()} in Last {tail_length} Iterations (Top {TOP_PERCENTILE*100:.0f}% Runs)')
    ax.set_xlabel('Iterations From End (0 = Final Iteration)')
    ax.set_ylabel(f'Mean {weight_col.replace("_"," ").title()}')
    ax.grid(True)
    # Optional: Invert x-axis if preferred
    # ax.invert_xaxis()

    plot_filename = plot_path / f'mean_{weight_col}_tail_{tail_length}_iterations.png'
    try:
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved plot: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot {plot_filename}: {e}")
    plt.close(fig)

    # Box plot of weight distribution vs. Iterations From End
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    # Define order for x-axis
    order = sorted(df_tail['iterations_from_end'].unique())
    sns.boxplot(data=df_tail, x='iterations_from_end', y=weight_col, order=order, ax=ax, palette='viridis')
    ax.set_title(f'{weight_col.replace("_"," ").title()} Distribution in Last {tail_length} Iterations (Top {TOP_PERCENTILE*100:.0f}% Runs)')
    ax.set_xlabel('Iterations From End (0 = Final Iteration)')
    ax.set_ylabel(f'{weight_col.replace("_"," ").title()} Value')
    ax.grid(True, axis='y')

    plot_filename = plot_path / f'boxplot_{weight_col}_tail_{tail_length}_iterations.png'
    try:
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved plot: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot {plot_filename}: {e}")
    plt.close(fig)

def main(data_dir, plot_dir):
    """Main function to run the analysis."""
    try:
        df_cleaned = load_and_clean_data(data_dir)
        top_run_ids, top_run_lengths = find_top_runs(df_cleaned)
        
        if not top_run_ids:
            print("No top runs identified. Exiting.")
            return

        # Filter original cleaned data to include only rows from top runs
        df_top_runs = df_cleaned[df_cleaned[RUN_ID_COL].isin(top_run_ids)].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        # Analyze and plot overall weight trends (as before)
        analyze_and_plot_weights(df_top_runs, plot_dir)

        # Merge trajectory lengths into the top runs data for tail analysis
        df_top_runs_with_len = pd.merge(df_top_runs, top_run_lengths, on=RUN_ID_COL, how='left')

        # Plot the tail behavior for all weights
        for w_col in WEIGHT_COLS:
            plot_tail_weight_trend(df_top_runs_with_len, w_col, plot_dir, tail_length=5)

        print("\nAnalysis complete.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze routing weights from top performing runs.')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help=f'Directory containing trainingdata_*.csv files (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--plot_dir', type=str, default=DEFAULT_PLOT_DIR,
                        help=f'Directory to save output plots (default: {DEFAULT_PLOT_DIR})')
    
    args = parser.parse_args()
    
    main(args.data_dir, args.plot_dir) 