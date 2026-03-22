import numpy as np

# -----------------------
# 1. Load your dataset
# -----------------------
npz_data = np.load("data/npz/compressed/data_64.npz")
X_train = npz_data["X_train"]  # shape: (N, H, W, C) or (N, H, W)
print("Original dataset shape:", X_train.shape)

# -----------------------
# 2. Define augmentation functions
# -----------------------
def horizontal_flip(images):
    return np.flip(images, axis=2)  # flip width axis

def vertical_flip(images):
    return np.flip(images, axis=1)  # flip height axis

def rotate_90(images, k=1):
    # Rotate each image 90*k degrees counterclockwise
    return np.array([np.rot90(img, k=k, axes=(0,1)) for img in images])

def add_noise(images, scale=0.05):
    # Add random Gaussian noise
    noisy = images + np.random.randn(*images.shape) * scale * 255
    return np.clip(noisy, 0, 255).astype(images.dtype)

# -----------------------
# 3. Apply augmentations
# -----------------------
X_hflip = horizontal_flip(X_train)
X_vflip = vertical_flip(X_train)
X_rot90 = rotate_90(X_train)
X_noise = add_noise(X_train, scale=0.05)

# Combine all augmented data with original
X_augmented = np.concatenate([X_train, X_hflip, X_vflip, X_rot90, X_noise], axis=0)
print("Combined dataset shape:", X_augmented.shape)

# -----------------------
# 4. Save to a new file
# -----------------------
np.save("data/npz/compressed/data_64_augmented.npy", X_augmented)
print("Saved augmented dataset to data_64_augmented.npy")