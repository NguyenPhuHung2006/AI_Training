import numpy as np

def check_for_empty_samples(file_path):
    # 1. Load the data (use mmap_mode='r' to save RAM if the file is huge)
    print(f"Loading {file_path}...")
    X = np.load(file_path)
    
    all_zero_mask = np.all(X == 0, axis=(1, 2, 3))
    failed_indices = np.where(all_zero_mask)[0]
    
    total = len(X)
    failed_count = len(failed_indices)
    success_count = total - failed_count
    
    print("-" * 30)
    print(f"RESULTS FOR: {file_path}")
    print(f"Total Samples:      {total}")
    print(f"Valid Samples:      {success_count}")
    print(f"Failed (All Zeros): {failed_count}")
    print(f"Health Rate:        {(success_count/total)*100:.2f}%")
    print("-" * 30)
    
    return failed_indices

# Usage
train_fails = check_for_empty_samples("data/X_test.npy")

if len(train_fails) > 0:
    print(f"First 10 indices to fix: {train_fails[:10]}")