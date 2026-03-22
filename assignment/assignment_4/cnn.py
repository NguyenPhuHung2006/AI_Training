import numpy as np
import pandas as pd
from ai_model.deep_learning.nn_torch import CNN
from ai_model.deep_learning.nn_torch import MLP
from ai_model.deep_learning.nn_torch.callback import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

def plot_img(X, index):
    first_image = X[index].astype(np.float32)  # ensure compatible dtype

    # If grayscale
    if first_image.ndim == 3 and first_image.shape[2] == 1:
        first_image = first_image[:, :, 0]

    plt.imshow(first_image, cmap="gray" if first_image.ndim == 2 else None)
    plt.axis("off")
    plt.show()
    
def conv_block(model, out_channels, dropout):
    model.add_filter(
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        activation="relu",
        norm="batch2d",
        dropout=dropout
    )
    model.add_pool(pool_type="max", kernel_size=2, stride=2)
    
def build_cnn(model, n_classes, size="128"):
    if size == "128":
        channels = [32, 64, 128, 256]
        dropouts = [0.1, 0.1, 0.15, 0.2]
        fc_layers = [512, 256]

    elif size == "64":
        channels = [32, 64, 128, 128]
        dropouts = [0.1, 0.1, 0.15, 0.2]
        fc_layers = [128]

    else:
        raise ValueError("size must be '128' or '64'")

    # Conv blocks
    for ch, dr in zip(channels, dropouts):
        conv_block(model, ch, dr)

    model.add_flatten()

    # FC layers
    for units in fc_layers:
        model.add_fc(n_units=units, activation="relu", dropout=0.5)

    model.add_fc(n_units=n_classes)

def main():
    
    X_train_empty_indices = [94, 5313, 7655, 7932, 10878, 10920, 11036, 14827, 16820, 17286, 17397, 17483, 18601, 19430]
    X_test_empty_indices = [1151, 1255, 3044, 4237, 4427, 4709]
    
    npz_data = np.load("data/npz/compressed/data_64.npz")
    X_train = npz_data["X_train"]
    X_test = npz_data["X_test"]
    y_train = pd.read_csv("data/train.csv", usecols=["label"])
    
    X_train = np.delete(X_train, X_train_empty_indices, axis=0)
    y_train = y_train.drop(index=X_train_empty_indices).reset_index(drop=True)
    
    # (B, H, W, C) -> (B, C, H, W)
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))
    
    unique_labels = sorted(y_train["label"].unique())

    # Number of classes
    n_classes = len(unique_labels)

    # Create dictionary: integer ID -> label text
    label_dict = {i: label for i, label in enumerate(unique_labels)}

    # Reverse mapping: label text -> integer ID
    text_to_int = {label: i for i, label in enumerate(unique_labels)}

    # Map y_train labels to integer IDs
    y_labels = y_train["label"].map(text_to_int).values  # NumPy array of ints

    model = CNN(
        in_channels=3,
        input_size=(X_train.shape[2], X_train.shape[3]),
        cost="cce",
        lr=3e-4,
        weight_decay=1e-4
    )
    
    build_cnn(model, n_classes, size="64")
    
    model.build()
    
    nn_data_path = "nn_data"
    os.makedirs(nn_data_path, exist_ok=True)
    
    i = 0
    while os.path.exists(f"{nn_data_path}/cnn_{i}.pth"):
        i += 1
       
    # reload the model 
    # model.load(f"{nn_data_path}/cnn_{i - 1}.pth")
            
    model.fit(
        X_train, 
        y_labels, 
        epochs=200,
        batch_size=64,
        val_split=0.1,
        callbacks=[
            EarlyStopping(patience=15),
            ModelCheckpoint(f"{nn_data_path}/cnn_{i}.pth")
        ]
    )
    
    y_test = model.predict(X_test)
    y_pred = np.argmax(y_test, axis=1)
    
    y_result = [label_dict[i] for i in y_pred]
        
    most_popular_label = np.bincount(y_labels).argmax()
    
    for i in X_test_empty_indices:
        y_result[i] = label_dict[most_popular_label]
        
    df = pd.DataFrame({
        "ID": np.arange(0, len(y_result)),
        "label": y_result
    })
    
    output_path = "outputs"
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(f"{output_path}/cnn.csv", index=False)
    

if __name__ == "__main__":
    main()
    print("completed")