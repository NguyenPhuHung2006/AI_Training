import numpy as np
import torch
import torch.nn.functional as F

def resize_batch(X, size=(64, 64)):
    X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)  # NHWC → NCHW
    X = F.interpolate(X, size=size, mode='bilinear', align_corners=False)
    X = X.permute(0, 2, 3, 1)  # back to NHWC
    return X.numpy()

# Load original
X_train = np.load("data/npz/128/X_train.npy")
X_test = np.load("data/npz/128/X_test.npy")

# Resize
X_train_64 = resize_batch(X_train, (64, 64))
X_test_64 = resize_batch(X_test, (64, 64))

# Save as npz
np.savez_compressed("data/npz/data_64.npz", X_train=X_train_64, X_test=X_test_64)
print("completed")