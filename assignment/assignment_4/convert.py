import numpy as np

X_train = np.load("data/npz/full/data_64_augmented.npy")
np.savez_compressed("data/npz/compressed/data_64_augmented.npz", X_train=X_train)
print("completed")