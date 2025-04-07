import pandas as pd
import numpy as np
import glob
import os
from collections import defaultdict

DATA_DIR = "../training_data"
ALL_FILES = glob.glob(os.path.join(DATA_DIR, "trainingdata_*.csv"))

def investigate_missing_values():
    """Investigate patterns in missing drv and wireLength values"""
    print("\n=== Investigating Missing drv and wireLength values ===")
    missing_data = {'file': [], 'uniqueID': [], 'iteration': [], 'missing_count': [], 'total_rows': []}
    
    total_missing = 0
    
    for file in ALL_FILES:
        filename = os.path.basename(file)
        print(f"Processing {filename}...")
        
        try:
            # Process in chunks to avoid memory issues
            chunk_num = 0
            file_missing = 0
            
            for chunk in pd.read_csv(file, chunksize=50000):
                chunk_num += 1
                
                # Check for missing values in drv or wireLength
                missing_mask = chunk['drv'].isnull() | chunk['wireLength'].isnull()
                chunk_missing = missing_mask.sum()
                file_missing += chunk_missing
                
                if chunk_missing > 0:
                    print(f"  Found {chunk_missing} missing values in chunk {chunk_num}")
                    
                    # Group by uniqueID and iteration to see which ones have missing values
                    missing_chunk = chunk[missing_mask]
                    grouped = missing_chunk.groupby(['uniqueID', 'iteration'])
                    
                    for (uid, it), group in grouped:
                        # Check if all rows for this iteration have missing values
                        all_rows_for_iter = chunk[(chunk['uniqueID'] == uid) & (chunk['iteration'] == it)]
                        missing_in_iter = all_rows_for_iter['drv'].isnull().sum()
                        
                        # Store the information
                        missing_data['file'].append(filename)
                        missing_data['uniqueID'].append(uid)
                        missing_data['iteration'].append(it)
                        missing_data['missing_count'].append(missing_in_iter)
                        missing_data['total_rows'].append(len(all_rows_for_iter))
                
                # Check if we already found significant missing data
                if file_missing > 1000:
                    break
            
            total_missing += file_missing
            print(f"  Total missing values in {filename}: {file_missing}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    if missing_data['file']:
        missing_df = pd.DataFrame(missing_data)
        print("\nSummary of missing values by uniqueID and iteration:")
        print(missing_df)
        
        # Check if missing values typically affect entire iterations
        missing_df['missing_percentage'] = 100 * missing_df['missing_count'] / missing_df['total_rows']
        
        # Count iterations with all values missing vs partial missing
        all_missing = (missing_df['missing_percentage'] > 99.9).sum()
        partial_missing = (missing_df['missing_percentage'] <= 99.9).sum()
        
        print(f"\nIterations with 100% missing values: {all_missing}")
        print(f"Iterations with partial missing values: {partial_missing}")
        
        # Group by file to see distribution
        by_file = missing_df.groupby('file')['missing_count'].sum().reset_index()
        print("\nMissing values by file:")
        print(by_file)
    
    print(f"\nTotal missing drv/wireLength values across all files: {total_missing}")

def investigate_duplicate_boxids():
    """Investigate duplicate boxIDs within the same uniqueID and iteration"""
    print("\n=== Investigating Duplicate boxIDs ===")
    dupes = {'file': [], 'uniqueID': [], 'iteration': [], 'boxID': [], 'count': [], 'identical': []}
    
    for file in ALL_FILES:
        filename = os.path.basename(file)
        print(f"Processing {filename}...")
        
        try:
            # Process in chunks to avoid memory issues
            for chunk in pd.read_csv(file, chunksize=50000):
                # Group by uniqueID, iteration, boxID and count occurrences
                duplicate_counts = chunk.groupby(['uniqueID', 'iteration', 'boxID']).size()
                duplicates = duplicate_counts[duplicate_counts > 1]
                
                if not duplicates.empty:
                    for (uid, it, box_id), count in duplicates.items():
                        # Check if the duplicate rows are identical
                        dupe_rows = chunk[(chunk['uniqueID'] == uid) & 
                                          (chunk['iteration'] == it) & 
                                          (chunk['boxID'] == box_id)]
                        
                        # Check if duplicate rows are identical (ignoring index)
                        identical = len(dupe_rows.drop_duplicates()) == 1
                        
                        dupes['file'].append(filename)
                        dupes['uniqueID'].append(uid)
                        dupes['iteration'].append(it)
                        dupes['boxID'].append(box_id)
                        dupes['count'].append(count)
                        dupes['identical'].append(identical)
                        
                        # For the first few duplicates, show detail
                        if len(dupes['file']) <= 3:
                            print(f"\nExample duplicate {len(dupes['file'])}:")
                            print(f"File: {filename}")
                            print(f"uniqueID: {uid}, iteration: {it}, boxID: {box_id}")
                            print(f"Number of occurrences: {count}")
                            print(f"Identical rows: {identical}")
                            
                            # If not identical, show the differences
                            if not identical:
                                print("\nDuplicate rows (showing differences):")
                                print(dupe_rows)
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    if dupes['file']:
        dupes_df = pd.DataFrame(dupes)
        print(f"\nFound {len(dupes_df)} instances of duplicate boxIDs")
        
        # Summary of identical vs different duplicates
        identical_count = dupes_df['identical'].sum()
        different_count = len(dupes_df) - identical_count
        
        print(f"Duplicates with identical data: {identical_count}")
        print(f"Duplicates with different data: {different_count}")
        
        # Distribution by file
        by_file = dupes_df.groupby('file').size().reset_index(name='count')
        print("\nDuplicates by file:")
        print(by_file)
    else:
        print("No duplicate boxIDs found")

def investigate_drv_mismatches():
    """Investigate patterns in DRV mismatches"""
    print("\n=== Investigating DRV Mismatches ===")
    mismatches = {'file': [], 'uniqueID': [], 'iteration': [], 'drv': [], 'sum_box_drv': [], 
                  'difference': [], 'box_count': [], 'design': []}
    
    for file in ALL_FILES:
        filename = os.path.basename(file)
        design = filename.replace('trainingdata_', '').replace('.csv', '')
        print(f"Processing {filename}...")
        
        try:
            # Process in chunks to avoid memory issues
            file_mismatches = 0
            
            for chunk in pd.read_csv(file, chunksize=50000):
                # Convert values to numeric
                chunk['drv'] = pd.to_numeric(chunk['drv'], errors='coerce')
                chunk['box_drv'] = pd.to_numeric(chunk['box_drv'], errors='coerce')
                
                # Group by uniqueID and iteration
                grouped = chunk.groupby(['uniqueID', 'iteration'])
                
                for (uid, it), group in grouped:
                    # Get iteration-level drv (should be constant within the group)
                    iter_drv = group['drv'].iloc[0]
                    
                    # Calculate sum of box_drv
                    sum_box_drv = group['box_drv'].sum()
                    
                    # Check if they match (allowing small floating point differences)
                    if not pd.isna(iter_drv) and not pd.isna(sum_box_drv) and not np.isclose(iter_drv, sum_box_drv, atol=1e-5):
                        file_mismatches += 1
                        mismatches['file'].append(filename)
                        mismatches['uniqueID'].append(uid)
                        mismatches['iteration'].append(it)
                        mismatches['drv'].append(iter_drv)
                        mismatches['sum_box_drv'].append(sum_box_drv)
                        mismatches['difference'].append(iter_drv - sum_box_drv)
                        mismatches['box_count'].append(len(group))
                        mismatches['design'].append(design)
            
            print(f"  Found {file_mismatches} DRV mismatches in {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    if mismatches['file']:
        mismatch_df = pd.DataFrame(mismatches)
        print(f"\nFound {len(mismatch_df)} instances of DRV mismatches")
        
        # Analyze patterns in the mismatches
        print("\nMismatch statistics:")
        print(f"Mean difference: {mismatch_df['difference'].mean():.2f}")
        print(f"Min difference: {mismatch_df['difference'].min():.2f}")
        print(f"Max difference: {mismatch_df['difference'].max():.2f}")
        
        # Check if the difference is always negative (iteration drv < sum of box_drv)
        neg_count = (mismatch_df['difference'] < 0).sum()
        pos_count = (mismatch_df['difference'] > 0).sum()
        print(f"Negative differences (iter_drv < sum_box_drv): {neg_count}")
        print(f"Positive differences (iter_drv > sum_box_drv): {pos_count}")
        
        # Check if the pattern varies by design
        print("\nMismatches by design:")
        by_design = mismatch_df.groupby('design').agg({
            'difference': ['mean', 'min', 'max', 'count'],
            'box_count': 'mean'
        })
        print(by_design)
        
        # Look for patterns at different iterations
        print("\nMismatches by iteration:")
        by_iteration = mismatch_df.groupby('iteration').agg({
            'difference': ['mean', 'min', 'max', 'count'],
            'box_count': 'mean'
        })
        print(by_iteration)
        
        # Check for correlation between difference and box count
        correlation = mismatch_df['box_count'].corr(mismatch_df['difference'])
        print(f"\nCorrelation between box count and difference: {correlation:.4f}")
        
        # Calculate average ratio between drv and sum_box_drv
        mismatch_df['ratio'] = mismatch_df['drv'] / mismatch_df['sum_box_drv']
        ratio_mean = mismatch_df['ratio'].mean()
        ratio_median = mismatch_df['ratio'].median()
        print(f"\nRatio statistics (drv / sum_box_drv):")
        print(f"Mean ratio: {ratio_mean:.4f}")
        print(f"Median ratio: {ratio_median:.4f}")
        
        # Check if the ratio is consistent across iterations
        ratio_by_iteration = mismatch_df.groupby('iteration')['ratio'].mean()
        print("\nMean ratio by iteration:")
        print(ratio_by_iteration)

if __name__ == "__main__":
    print("Starting detailed investigation of specific data issues...")
    investigate_missing_values()
    investigate_duplicate_boxids()
    investigate_drv_mismatches()
    print("\nInvestigation complete.") 