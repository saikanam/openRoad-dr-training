import pandas as pd
import numpy as np
import glob
import os
from collections import defaultdict

def load_original_data(data_dir="../training_data"):
    """
    Load and analyze the original training data from CSV files.
    Returns:
        dict: Statistics about the dataset
        pd.DataFrame: Combined dataframe of all runs
    """
    print("Loading original training data...")
    all_files = glob.glob(os.path.join(data_dir, "trainingdata_*.csv"))
    
    if not all_files:
        raise FileNotFoundError(f"No training data CSVs found in {data_dir}")

    print(f"Found {len(all_files)} data files.")
    
    # Initialize statistics
    stats = {
        'total_records': 0,
        'unique_runs': set(),
        'unique_designs': set(),
        'runs_per_design': defaultdict(int),
        'iterations_per_run': defaultdict(int),
        'total_iterations': 0
    }
    
    df_list = []
    for f in all_files:
        filename = os.path.basename(f)
        try:
            df_file = pd.read_csv(f)
            stats['total_records'] += len(df_file)
            
            # Collect unique runs and designs
            unique_runs = df_file['uniqueID'].unique()
            stats['unique_runs'].update(unique_runs)
            
            unique_designs = df_file['designID'].unique()
            stats['unique_designs'].update(unique_designs)
            
            # Count runs per design
            design_counts = df_file.groupby('designID')['uniqueID'].nunique()
            for design, count in design_counts.items():
                stats['runs_per_design'][design] += count
            
            # Count iterations per run
            run_iterations = df_file.groupby('uniqueID')['iteration'].max()
            for run, max_iter in run_iterations.items():
                stats['iterations_per_run'][run] = max(stats['iterations_per_run'][run], max_iter)
                stats['total_iterations'] += max_iter
            
            df_list.append(df_file)
            print(f"Processed {filename}: {len(df_file)} rows")
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue
    
    if not df_list:
        raise ValueError("No dataframes could be loaded or processed.")
    
    df_combined = pd.concat(df_list, ignore_index=True)
    
    # Convert defaultdict to regular dict for cleaner printing
    stats['runs_per_design'] = dict(stats['runs_per_design'])
    stats['iterations_per_run'] = dict(stats['iterations_per_run'])
    
    # Print summary statistics
    print("\nDataset Statistics:")
    print(f"Total records: {stats['total_records']}")
    print(f"Unique runs: {len(stats['unique_runs'])}")
    print(f"Unique designs: {len(stats['unique_designs'])}")
    print(f"Average iterations per run: {stats['total_iterations'] / len(stats['unique_runs']):.2f}")
    print("\nRuns per design:")
    for design, count in stats['runs_per_design'].items():
        print(f"  {design}: {count} runs")
    
    return stats, df_combined

if __name__ == "__main__":
    stats, df = load_original_data() 