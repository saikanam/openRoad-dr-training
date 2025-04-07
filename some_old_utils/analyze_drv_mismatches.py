import pandas as pd
import numpy as np
import glob
import os
from collections import defaultdict

# Configuration
DATA_DIR = "../training_data"
# Start with smaller files for fast results
SMALLER_FILES = ["trainingdata_gcd_base.csv", "trainingdata_ispd18_test1.csv", "trainingdata_ibex_base.csv"]

def investigate_drv_mismatches():
    """Detailed investigation of mismatches between iteration drv and sum of box_drv"""
    print("\n=== Investigating DRV Sum Mismatches in Detail ===")
    mismatches = {'file': [], 'uniqueID': [], 'iteration': [], 'drv': [], 'sum_box_drv': [], 
                  'difference': [], 'box_count': [], 'zero_drv_boxes': [], 'non_zero_drv_boxes': []}
    examples_examined = 0
    
    for filename in SMALLER_FILES:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
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
                
                # Count boxes with zero and non-zero DRV
                zero_drv_boxes = (group['box_drv'] == 0).sum()
                non_zero_drv_boxes = (group['box_drv'] > 0).sum()
                
                # Check if they match (allowing small floating point differences)
                if not pd.isna(iter_drv) and not pd.isna(sum_box_drv) and not np.isclose(iter_drv, sum_box_drv, atol=1e-5):
                    mismatches['file'].append(filename)
                    mismatches['uniqueID'].append(uid)
                    mismatches['iteration'].append(it)
                    mismatches['drv'].append(iter_drv)
                    mismatches['sum_box_drv'].append(sum_box_drv)
                    mismatches['difference'].append(iter_drv - sum_box_drv)
                    mismatches['box_count'].append(len(group))
                    mismatches['zero_drv_boxes'].append(zero_drv_boxes)
                    mismatches['non_zero_drv_boxes'].append(non_zero_drv_boxes)
                    
                    # Detailed examination of first few examples
                    if examples_examined < 5:
                        print("\n" + "="*50)
                        print(f"Example mismatch #{examples_examined+1}:")
                        print(f"File: {filename}")
                        print(f"uniqueID: {uid}")
                        print(f"iteration: {it}")
                        print(f"Iteration drv: {iter_drv}")
                        print(f"Sum of box_drv: {sum_box_drv}")
                        print(f"Difference: {iter_drv - sum_box_drv}")
                        print(f"Total boxes: {len(group)}")
                        print(f"Boxes with zero DRV: {zero_drv_boxes}")
                        print(f"Boxes with non-zero DRV: {non_zero_drv_boxes}")
                        
                        # Show distribution of box_drv values
                        print("\nDistribution of box_drv values:")
                        print(group['box_drv'].value_counts().sort_index())
                        
                        # Check if all iterations in this run have the same pattern
                        run_iterations = df[df['uniqueID'] == uid].groupby('iteration')
                        print("\nAll iterations for this run:")
                        iter_data = []
                        for it_num, it_group in run_iterations:
                            it_drv = it_group['drv'].iloc[0]
                            it_sum_drv = it_group['box_drv'].sum()
                            iter_data.append({
                                'iteration': it_num,
                                'drv': it_drv,
                                'sum_box_drv': it_sum_drv,
                                'diff': it_drv - it_sum_drv if not pd.isna(it_drv) and not pd.isna(it_sum_drv) else None
                            })
                        
                        iter_df = pd.DataFrame(iter_data)
                        print(iter_df)
                        
                        print("="*50)
                        examples_examined += 1
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    if mismatches['file']:
        mismatch_df = pd.DataFrame(mismatches)
        print(f"\nFound {len(mismatch_df)} instances of DRV sum mismatches")
        
        # Analyze patterns in the mismatches
        print("\nMismatch statistics:")
        print(f"Mean difference: {mismatch_df['difference'].mean()}")
        print(f"Min difference: {mismatch_df['difference'].min()}")
        print(f"Max difference: {mismatch_df['difference'].max()}")
        
        # Check if the difference is always negative (iteration drv < sum of box_drv)
        all_negative = (mismatch_df['difference'] < 0).all()
        print(f"Are all differences negative? {all_negative}")
        
        # Analyze the relationship between box count and difference
        if len(mismatch_df) > 5:
            correlation = mismatch_df['box_count'].corr(mismatch_df['difference'])
            print(f"Correlation between box count and difference: {correlation:.4f}")
        
        # Look for patterns at different iterations
        print("\nMismatches by iteration:")
        by_iteration = mismatch_df.groupby('iteration').agg({
            'difference': ['mean', 'min', 'max', 'count'],
            'box_count': 'mean'
        })
        print(by_iteration)
    else:
        print("No DRV sum mismatches found")

if __name__ == "__main__":
    print("Starting detailed DRV mismatch investigation...")
    investigate_drv_mismatches()
    print("\nInvestigation complete.") 