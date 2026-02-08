import numpy as np

data = np.load("model.npz")

for key in data:
    print(key, "=", np.array(data[key].tolist()))
