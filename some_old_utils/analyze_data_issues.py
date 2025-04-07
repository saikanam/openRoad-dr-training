import pandas as pd
import numpy as np
import glob
import os
from collections import defaultdict

# Configuration
DATA_DIR = "../training_data"
SMALLER_FILES = ["trainingdata_gcd_base.csv", "trainingdata_ispd18_test1.csv", "trainingdata_ibex_base.csv"]

def investigate_missing_values():
    """Investigate missing drv and wireLength values"""
    print("\n=== Investigating Missing drv and wireLength values ===")
    missing_data = {'file': [], 'uniqueID': [], 'iteration': [], 'missing_drv': [], 'missing_wl': []}
    
    for filename in SMALLER_FILES:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        print(f"Processing {filename}...")
        try:
            df = pd.read_csv(file_path)
            
            # Check for missing values
            drv_null = df['drv'].isnull()
            wl_null = df['wireLength'].isnull()
            
            if drv_null.any() or wl_null.any():
                for idx in df[drv_null | wl_null].index:
                    row = df.loc[idx]
                    missing_data['file'].append(filename)
                    missing_data['uniqueID'].append(row['uniqueID'])
                    missing_data['iteration'].append(row['iteration'])
                    missing_data['missing_drv'].append(pd.isna(row['drv']))
                    missing_data['missing_wl'].append(pd.isna(row['wireLength']))
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    if missing_data['file']:
        missing_df = pd.DataFrame(missing_data)
        print(f"Found {len(missing_df)} rows with missing values")
        print("Sample of missing data:")
        print(missing_df.head(10))
        # Analyze by uniqueID and iteration
        print("\nMissing values by uniqueID and iteration:")
        by_run = missing_df.groupby(['uniqueID', 'iteration']).size().reset_index(name='count')
        print(by_run.head(10))
    else:
        print("No missing values found in the sampled files")

def investigate_duplicate_boxids():
    """Investigate duplicate boxIDs within same uniqueID and iteration"""
    print("\n=== Investigating Duplicate boxIDs ===")
    dupes = {'file': [], 'uniqueID': [], 'iteration': [], 'boxID': [], 'count': []}
    
    for filename in SMALLER_FILES:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            continue
            
        print(f"Processing {filename}...")
        try:
            df = pd.read_csv(file_path)
            
            # Group by uniqueID, iteration, boxID and count occurrences
            grouped = df.groupby(['uniqueID', 'iteration', 'boxID']).size()
            # Find duplicates (count > 1)
            duplicates = grouped[grouped > 1]
            
            if not duplicates.empty:
                for (uid, it, box_id), count in duplicates.items():
                    dupes['file'].append(filename)
                    dupes['uniqueID'].append(uid)
                    dupes['iteration'].append(it)
                    dupes['boxID'].append(box_id)
                    dupes['count'].append(count)
                
                # For one example, show the actual duplicate rows
                if len(dupes['file']) > 0:
                    example_uid = dupes['uniqueID'][0]
                    example_it = dupes['iteration'][0]
                    example_box = dupes['boxID'][0]
                    
                    dupe_rows = df[(df['uniqueID'] == example_uid) & 
                                   (df['iteration'] == example_it) & 
                                   (df['boxID'] == example_box)]
                    
                    print(f"\nExample duplicate from {filename}:")
                    print(f"uniqueID={example_uid}, iteration={example_it}, boxID={example_box}")
                    print("Duplicate rows:")
                    print(dupe_rows)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    if dupes['file']:
        dupes_df = pd.DataFrame(dupes)
        print(f"Found {len(dupes_df)} instances of duplicate boxIDs")
        print("Summary of duplicates:")
        print(dupes_df.head(10))
    else:
        print("No duplicate boxIDs found in the sampled files")

def investigate_drv_mismatches():
    """Investigate mismatches between iteration drv and sum of box_drv"""
    print("\n=== Investigating DRV Sum Mismatches ===")
    mismatches = {'file': [], 'uniqueID': [], 'iteration': [], 'drv': [], 'sum_box_drv': [], 'difference': [], 'box_count': []}
    
    for filename in SMALLER_FILES:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            continue
            
        print(f"Processing {filename}...")
        try:
            df = pd.read_csv(file_path)
            
            # Convert values to numeric if they aren't already
            df['drv'] = pd.to_numeric(df['drv'], errors='coerce')
            df['box_drv'] = pd.to_numeric(df['box_drv'], errors='coerce')
            
            # Group by uniqueID and iteration
            grouped = df.groupby(['uniqueID', 'iteration'])
            
            for (uid, it), group in grouped:
                # Get iteration-level drv (should be constant within the group)
                iter_drv = group['drv'].iloc[0]
                
                # Calculate sum of box_drv
                sum_box_drv = group['box_drv'].sum()
                
                # Check if they match (allowing small floating point differences)
                if not pd.isna(iter_drv) and not pd.isna(sum_box_drv) and not np.isclose(iter_drv, sum_box_drv, atol=1e-5):
                    mismatches['file'].append(filename)
                    mismatches['uniqueID'].append(uid)
                    mismatches['iteration'].append(it)
                    mismatches['drv'].append(iter_drv)
                    mismatches['sum_box_drv'].append(sum_box_drv)
                    mismatches['difference'].append(iter_drv - sum_box_drv)
                    mismatches['box_count'].append(len(group))
                    
                    # For the first mismatch, print more details
                    if len(mismatches['file']) == 1:
                        print(f"\nDetailed example for {uid}, iteration {it}:")
                        print(f"Iteration drv: {iter_drv}")
                        print(f"Sum of box_drv: {sum_box_drv}")
                        print(f"Difference: {iter_drv - sum_box_drv}")
                        print(f"Number of boxes: {len(group)}")
                        print("Distribution of box_drv values:")
                        print(group['box_drv'].value_counts().head(10))
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    if mismatches['file']:
        mismatch_df = pd.DataFrame(mismatches)
        print(f"\nFound {len(mismatch_df)} instances of DRV sum mismatches")
        print("Summary of mismatches:")
        print(mismatch_df.head(10))
        
        # Analyze the differences
        print("\nDifference statistics:")
        print(f"Mean diff: {mismatch_df['difference'].mean()}")
        print(f"Median diff: {mismatch_df['difference'].median()}")
        print(f"Min diff: {mismatch_df['difference'].min()}")
        print(f"Max diff: {mismatch_df['difference'].max()}")
        
        # Look for patterns
        print("\nAnalyzing if mismatches occur more at certain iterations:")
        by_iteration = mismatch_df.groupby('iteration').size().reset_index(name='count')
        print(by_iteration.head(10))
    else:
        print("No DRV sum mismatches found in the sampled files")

if __name__ == "__main__":
    print("Starting data issue investigation...")
    investigate_missing_values()
    investigate_duplicate_boxids()
    investigate_drv_mismatches()
    print("\nInvestigation complete.") 