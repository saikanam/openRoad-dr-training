import pandas as pd
import numpy as np
import glob
import os
from collections import defaultdict

# Configuration
DATA_DIR = "../training_data"
ALL_FILES = glob.glob(os.path.join(DATA_DIR, "trainingdata_*.csv"))

def investigate_duplicate_boxids():
    """Investigate duplicate boxIDs within same uniqueID and iteration"""
    print("\n=== Investigating Duplicate boxIDs ===")
    dupes = {'file': [], 'uniqueID': [], 'iteration': [], 'boxID': [], 'count': []}
    
    for file in ALL_FILES:
        filename = os.path.basename(file)
        print(f"Processing {filename}...")
        
        try:
            # Process in chunks to avoid memory issues
            for chunk in pd.read_csv(file, chunksize=50000):
                # Group by uniqueID, iteration, boxID and count occurrences
                grouped = chunk.groupby(['uniqueID', 'iteration', 'boxID']).size()
                # Find duplicates (count > 1)
                duplicates = grouped[grouped > 1]
                
                if not duplicates.empty:
                    for (uid, it, box_id), count in duplicates.items():
                        dupes['file'].append(filename)
                        dupes['uniqueID'].append(uid)
                        dupes['iteration'].append(it)
                        dupes['boxID'].append(box_id)
                        dupes['count'].append(count)
                        
                    # For the first few duplicates, show the actual duplicate rows
                    if len(dupes['file']) <= 5:
                        example_uid = dupes['uniqueID'][-1]
                        example_it = dupes['iteration'][-1]
                        example_box = dupes['boxID'][-1]
                        
                        dupe_rows = chunk[(chunk['uniqueID'] == example_uid) & 
                                          (chunk['iteration'] == example_it) & 
                                          (chunk['boxID'] == example_box)]
                        
                        print(f"\nExample duplicate from {filename}:")
                        print(f"uniqueID={example_uid}, iteration={example_it}, boxID={example_box}")
                        print("Duplicate rows:")
                        print(dupe_rows)
                
                # Stop once we've found a reasonable number of duplicates
                if len(dupes['file']) >= 20:
                    break
                    
            # Stop once we've found a reasonable number of duplicates
            if len(dupes['file']) >= 20:
                break
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    if dupes['file']:
        dupes_df = pd.DataFrame(dupes)
        print(f"\nFound {len(dupes_df)} instances of duplicate boxIDs")
        print("Summary of duplicates:")
        print(dupes_df)
        
        # Analyze patterns
        print("\nDuplicates by file:")
        by_file = dupes_df.groupby('file').size().reset_index(name='count')
        print(by_file)
        
        # Check how many uniqueIDs are affected
        print(f"\nNumber of unique runs affected: {dupes_df['uniqueID'].nunique()}")
        
    else:
        print("No duplicate boxIDs found")

if __name__ == "__main__":
    print("Starting duplicate boxIDs investigation...")
    investigate_duplicate_boxids()
    print("\nInvestigation complete.") 