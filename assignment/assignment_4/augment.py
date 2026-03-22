import numpy as np

# -----------------------
# 1. Load your dataset
# -----------------------
npz_data = np.load("data/npz/compressed/data_64.npz")
X_train = npz_data["X_train"].astype('float32')  # ensure float32
print("Original dataset shape:", X_train.shape, "| min/max:", X_train.min(), X_train.max())

# -----------------------
# 2. Define augmentation functions
# -----------------------
def horizontal_flip(images):
    return np.flip(images, axis=2)

def vertical_flip(images):
    return np.flip(images, axis=1)

def rotate_90(images, k=1):
    return np.array([np.rot90(img, k=k, axes=(0,1)) for img in images])

def add_noise(images, scale=0.05):
    noisy = images + np.random.randn(*images.shape) * scale
    return np.clip(noisy, 0.0, 1.0).astype(images.dtype)

# -----------------------
# 3. Randomly select samples to augment
# -----------------------
max_samples = 60000
n_original = X_train.shape[0]

# Shuffle indices
indices = np.random.permutation(n_original)

# Limit number of augmented samples to not exceed max_samples
n_aug = min(max_samples - n_original, n_original)  # extra samples we can add
selected_indices = indices[:n_aug]

X_selected = X_train[selected_indices]

# -----------------------
# 4. Apply augmentations to selected samples
# -----------------------
X_augmented = []

for x in X_selected:
    X_augmented.append(x)                       # original
    X_augmented.append(horizontal_flip(x[np.newaxis])[0])
    X_augmented.append(vertical_flip(x[np.newaxis])[0])
    X_augmented.append(rotate_90(x[np.newaxis])[0])
    X_augmented.append(add_noise(x[np.newaxis], scale=0.05)[0])

X_augmented = np.array(X_augmented, dtype=np.float32)

# -----------------------
# 5. Combine with original dataset
# -----------------------
X_final = np.concatenate([X_train, X_augmented], axis=0)

# Clip to max_samples if exceeded
if X_final.shape[0] > max_samples:
    X_final = X_final[:max_samples]

print("Final dataset shape:", X_final.shape, "| min/max:", X_final.min(), X_final.max())

# -----------------------
# 6. Save
# -----------------------
np.save("data/npz/compressed/data_64_augmented_limited.npy", X_final)
print("Saved augmented dataset (limited) to data_64_augmented_limited.npy")