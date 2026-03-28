import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df = pd.read_csv("data/train.csv")
# y_train = df['label']

# y_augmented = np.concatenate([y_train]*5, axis=0)

# df = pd.DataFrame({
#     "label": y_augmented
# })

# df.to_csv(f"data/y_augmented.csv", index=False)

# print("y_augmented:", y_augmented.shape)

# X = np.load("data/npz/compressed/data_64_augmented.npz")
# X = X["X_train"]
# print(X.dtype)
# X = X.astype('float32')
# print("Min:", X.min())
# print("Max:", X.max())

# def plot_img(X, index):
#     first_image = X[index].astype(np.float32)  # ensure compatible dtype

#     # If grayscale
#     if first_image.ndim == 3 and first_image.shape[2] == 1:
#         first_image = first_image[:, :, 0]

#     plt.imshow(first_image, cmap="gray" if first_image.ndim == 2 else None)
#     plt.axis("off")
#     plt.show()
    
# npz_data = np.load("data/npz/compressed/data_64_augmented.npz")
# X = npz_data["X_train"] / 255.0
# plot_img(X, 3)

# X = np.load("data/npz/full/data_64_augmented.npy")
# np.savez("data/npz/compressed/data_64_augmented.npz", X_train=X)








print("completed")