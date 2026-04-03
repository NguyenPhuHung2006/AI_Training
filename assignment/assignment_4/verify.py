import numpy as np

def check_for_empty_samples(file_path, key=None):
    print(f"Loading {file_path}...")

    data = np.load(file_path, mmap_mode='r')

    # ---- Handle .npz ----
    if isinstance(data, np.lib.npyio.NpzFile):
        if key is None:
            print(f"Available keys: {list(data.keys())}")
            raise ValueError("You must specify a key for .npz file")

        X = data[key]
    else:
        X = data  # .npy case

    # ---- Safety check ----
    if X.ndim != 4:
        raise ValueError(f"Expected 4D array (N,H,W,C), got shape {X.shape}")

    # ---- Check all-zero samples ----
    all_zero_mask = np.all(X == 0, axis=(1, 2, 3))
    failed_indices = np.where(all_zero_mask)[0]

    total = len(X)
    failed_count = len(failed_indices)
    success_count = total - failed_count

    print("-" * 30)
    print(f"RESULTS FOR: {file_path}")
    print(f"Shape:              {X.shape}")
    print(f"Total Samples:      {total}")
    print(f"Valid Samples:      {success_count}")
    print(f"Failed (All Zeros): {failed_count}")
    print(f"Health Rate:        {(success_count/total)*100:.2f}%")
    print("-" * 30)

    return failed_indices

# Usage
# train_fails = check_for_empty_samples("data/X_test.npy")

# if len(train_fails) > 0:
#     print(f"First 10 indices to fix: {train_fails[:10]}")

fails = check_for_empty_samples("data/npz_fixed/data_augmented.npz", key="X_train")

if len(fails) > 0:
    print(f"First 10 indices to fix: {fails[:100]}")