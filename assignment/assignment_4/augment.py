import numpy as np

# -----------------------
# 1. Load NORMALIZED data
# -----------------------
npz_data = np.load("data/npz/compressed/data_64.npz")
X_train_norm = npz_data["X_train"]  # already in [0,1]

print("Normalized min/max:", X_train_norm.min(), X_train_norm.max())

# -----------------------
# 2. Convert back to RAW for augmentation
# -----------------------
X_train_raw = (X_train_norm * 255).clip(0, 255).astype(np.uint8)

print("Raw min/max:", X_train_raw.min(), X_train_raw.max())

# -----------------------
# 3. Augmentation functions (WORK ON RAW)
# -----------------------
def horizontal_flip(images):
    return np.flip(images, axis=2)

def vertical_flip(images):
    return np.flip(images, axis=1)

def rotate_90(images, k=1):
    return np.array([np.rot90(img, k=k, axes=(0,1)) for img in images])

def add_noise(images, scale=0.05):
    noisy = images.astype(np.float32) + np.random.randn(*images.shape) * scale * 255
    return np.clip(noisy, 0, 255).astype(np.uint8)

# -----------------------
# 4. Apply augmentations (RAW ONLY)
# -----------------------
X_hflip = horizontal_flip(X_train_raw)
X_vflip = vertical_flip(X_train_raw)
X_rot90 = rotate_90(X_train_raw)
X_noise = add_noise(X_train_raw, scale=0.05)

# Combine ONLY augmented (NOT normalized)
X_augmented = np.concatenate(
    [X_train_raw, X_hflip, X_vflip, X_rot90, X_noise],
    axis=0
).astype(np.uint8)

print("Augmented shape:", X_augmented.shape)
print("Augmented min/max:", X_augmented.min(), X_augmented.max())

# -----------------------
# 5. Save
# -----------------------
np.savez_compressed(
    "data/npz/compressed/data_64_augmented.npz",
    X_train=X_augmented
)

print("Saved successfully")