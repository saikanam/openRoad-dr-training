import pandas as pd
import numpy as np
import glob
import os
from collections import defaultdict

# Configuration
DATA_DIR = "../training_data"
ALL_FILES = glob.glob(os.path.join(DATA_DIR, "trainingdata_*.csv"))

def investigate_duplicate_boxids():
    """Investigate duplicate boxIDs within same uniqueID and iteration in detail"""
    print("\n=== Investigating Duplicate boxIDs ===")
    dupes = {'file': [], 'uniqueID': [], 'iteration': [], 'boxID': [], 'count': []}
    examples_shown = 0
    
    for file in ALL_FILES:
        filename = os.path.basename(file)
        print(f"Processing {filename}...")
        
        try:
            # Process in chunks to avoid memory issues
            chunk_num = 0
            for chunk in pd.read_csv(file, chunksize=50000):
                chunk_num += 1
                print(f"  Processing chunk {chunk_num}...")
                
                # Group by uniqueID, iteration, boxID and count occurrences
                grouped = chunk.groupby(['uniqueID', 'iteration', 'boxID']).size()
                # Find duplicates (count > 1)
                duplicates = grouped[grouped > 1]
                
                if not duplicates.empty:
                    print(f"  Found {len(duplicates)} duplicate boxIDs in this chunk")
                    
                    for (uid, it, box_id), count in duplicates.items():
                        dupes['file'].append(filename)
                        dupes['uniqueID'].append(uid)
                        dupes['iteration'].append(it)
                        dupes['boxID'].append(box_id)
                        dupes['count'].append(count)
                        
                        # Show example of first few duplicates
                        if examples_shown < 3:
                            dupe_rows = chunk[(chunk['uniqueID'] == uid) & 
                                              (chunk['iteration'] == it) & 
                                              (chunk['boxID'] == box_id)]
                            
                            print("\n" + "="*50)
                            print(f"Example duplicate #{examples_shown+1}:")
                            print(f"File: {filename}")
                            print(f"uniqueID: {uid}")
                            print(f"iteration: {it}")
                            print(f"boxID: {box_id}")
                            print(f"Number of duplicates: {count}")
                            print("\nDuplicate rows:")
                            with pd.option_context('display.max_columns', None):
                                print(dupe_rows)
                            print("="*50)
                            
                            examples_shown += 1
                
                # Stop after processing a reasonable number of chunks
                if chunk_num >= 5:
                    break
                    
            # Process at most 3 files
            if dupes['file'] and len(set(dupes['file'])) >= 3:
                break
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    if dupes['file']:
        dupes_df = pd.DataFrame(dupes)
        print(f"\nFound {len(dupes_df)} instances of duplicate boxIDs")
        
        # Analyze patterns
        print("\nDuplicates by file:")
        by_file = dupes_df.groupby('file').size().reset_index(name='count')
        print(by_file)
        
        # Check how many uniqueIDs are affected
        print(f"\nNumber of unique runs affected: {dupes_df['uniqueID'].nunique()}")
        
        # Distribution of duplicates per run
        print("\nTop runs with duplicate boxIDs:")
        by_run = dupes_df.groupby(['file', 'uniqueID']).size().reset_index(name='count')
        print(by_run.sort_values('count', ascending=False).head(10))
        
    else:
        print("No duplicate boxIDs found")

if __name__ == "__main__":
    print("Starting detailed duplicate boxIDs investigation...")
    investigate_duplicate_boxids()
    print("\nInvestigation complete.") 