import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import Subset

class Callback:

    def on_train_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer, logs):
        pass

class EarlyStopping(Callback):

    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def on_epoch_end(self, trainer, logs):

        val_loss = logs.get("val_loss")

        if val_loss is None:
            return

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                print("Early stopping triggered")
                trainer.stop_training = True

class ModelCheckpoint(Callback):

    def __init__(self, path="best_model.pth"):
        self.path = path
        self.best_loss = float("inf")

    def on_epoch_end(self, trainer, logs):

        val_loss = logs.get("val_loss")

        if val_loss is None:
            return

        if val_loss < self.best_loss:

            self.best_loss = val_loss
            trainer.save(self.path)

class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # numpy array
        self.y = y  # numpy array

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        # Convert ONLY when needed (per sample)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y)

        return x, y

class BaseModel(nn.Module):
    def __init__(self, cost="cce", lr=0.001, weight_decay=0.0, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.configs = {
            "cost": {
                "cce": nn.CrossEntropyLoss,
                "mse": nn.MSELoss,
                "bce": nn.BCEWithLogitsLoss
            },
            "activation": {
                "relu": nn.ReLU,
                "leaky_relu": nn.LeakyReLU,
                "gelu": nn.GELU,
                "tanh": nn.Tanh,
                "sigmoid": nn.Sigmoid
            },
            "init": {
                "he": nn.init.kaiming_uniform_,
                "xavier": nn.init.xavier_uniform_,
                "normal": nn.init.normal_
            },
            "norm": {
                "batch": nn.BatchNorm1d,
                "layer": nn.LayerNorm,
                "batch2d": nn.BatchNorm2d
            },
            "dropout": {
                "standard": nn.Dropout,
                "feature": nn.Dropout1d,
                "alpha": nn.AlphaDropout
            },
            "scheduler": {
                "step": optim.lr_scheduler.StepLR,
                "exp": optim.lr_scheduler.ExponentialLR,
                "cosine": optim.lr_scheduler.CosineAnnealingLR,
                "reduce_on_plateau": optim.lr_scheduler.ReduceLROnPlateau
            }
        }

        self.criterion = self.configs["cost"][cost]()
        self.lr = lr
        self.weight_decay = weight_decay
        self.layers = nn.ModuleList()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.stop_training = False

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

    def _apply_utils(
        self,
        module,
        activation=None,
        dropout=0.0,
        dropout_type="standard",
        norm=None,
        norm_dim=None,
        init=None
    ):
        layer_group = [module]
        if norm:
            layer_group.append(self.configs["norm"][norm](norm_dim))
        if activation:
            layer_group.append(self.configs["activation"][activation]())
        if dropout > 0:
            layer_group.append(self.configs["dropout"][dropout_type](dropout))
        if init:
            self.configs["init"][init](module.weight)
        return nn.Sequential(*layer_group)

    def set_scheduler(self, name, **kwargs):
        if self.optimizer is None:
            raise RuntimeError("Call build() before setting the scheduler.")
        self.scheduler = self.configs["scheduler"][name](self.optimizer, **kwargs)
        return self

    def build(self):
        self.model = nn.Sequential(*self.layers).to(self.device)

        main_params = list(self.model.parameters())
        embed_params = list(self.embeddings.parameters()) if hasattr(self, 'embeddings') else []

        if embed_params:
            params_to_train = [
                {'params': main_params, 'lr': self.lr},
                {'params': embed_params, 'lr': self.lr * 10}
            ]
        else:
            params_to_train = [{'params': main_params, 'lr': self.lr}]

        self.optimizer = torch.optim.AdamW(params_to_train, weight_decay=self.weight_decay, lr=self.lr)
        return self

    def _to_tensor(self, X, y=None):
        X = torch.as_tensor(X).to(self.device).float()
        if y is None:
            return X
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            y_dtype = torch.long
        else:
            y_dtype = torch.float32
        return X, torch.as_tensor(y, dtype=y_dtype).to(self.device)

    def _accuracy(self, outputs, y):
        total = None
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            preds = torch.argmax(outputs, dim=1)
            total = y.size(0)
        elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
            preds = (torch.sigmoid(outputs) > 0.5).view_as(y).float()
            total = y.numel()
        else:
            return None

        correct = (preds == y.to(preds.dtype)).sum().item()

        return correct, total

    def _evaluate(self, X, y):
        self.model.eval()

        with torch.no_grad():
            outputs = self(X)
            loss = self.criterion(outputs, y).item()
            acc = self._accuracy(outputs, y)
            if acc is None:
                return loss, None
            correct, total = acc
            return loss, correct / total

    def fit(self, X, y, epochs=10, batch_size=64, val_split=0.1, callbacks=None):
          callbacks = callbacks or []

          # ---- Build dataset ONCE (no duplication) ----
          full_dataset = NumpyDataset(X, y)

          if val_split > 0:
              indices = list(range(len(X)))

              train_idx, val_idx = train_test_split(
                  indices,
                  test_size=val_split,
                  random_state=42,
                  stratify=y if isinstance(self.criterion, nn.CrossEntropyLoss) else None
              )

              train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
              val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
          else:
              train_dataset = full_dataset
              val_dataset = None

          # ---- DataLoaders ----
          train_loader = DataLoader(
              train_dataset,
              batch_size=batch_size,
              shuffle=True,
              num_workers=0
          )

          val_loader = None
          if val_dataset is not None:
              val_loader = DataLoader(
                  val_dataset,
                  batch_size=batch_size,
                  shuffle=False,
                  num_workers=0
              )

          # ---- Callbacks ----
          for cb in callbacks:
              cb.on_train_begin(self)

          # ---- Training loop ----
          for epoch in tqdm(range(epochs), desc="Training"):
              if self.stop_training:
                  break

              self.model.train()

              total_loss = 0
              train_correct = 0
              train_total = 0

              for xb, yb in train_loader:
                  xb = xb.to(self.device)
                  yb = yb.to(self.device)

                  self.optimizer.zero_grad()

                  out = self(xb)
                  loss = self.criterion(out, yb)

                  loss.backward()
                  torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                  self.optimizer.step()

                  total_loss += loss.item()

                  acc = self._accuracy(out, yb)
                  if acc is not None:
                      correct, total = acc
                      train_correct += correct
                      train_total += total

              train_loss = total_loss / len(train_loader)
              train_acc = train_correct / train_total if train_total > 0 else None

              # ---- Validation ----
              val_loss = None
              val_acc = None

              if val_loader is not None:
                  self.model.eval()
                  v_loss = 0
                  v_correct = 0
                  v_total = 0

                  with torch.no_grad():
                      for xb, yb in val_loader:
                          xb = xb.to(self.device)
                          yb = yb.to(self.device)

                          out = self(xb)
                          loss = self.criterion(out, yb)

                          v_loss += loss.item()

                          acc = self._accuracy(out, yb)
                          if acc is not None:
                              correct, total = acc
                              v_correct += correct
                              v_total += total

                  val_loss = v_loss / len(val_loader)
                  val_acc = v_correct / v_total if v_total > 0 else None

              # ---- Scheduler ----
              if self.scheduler is not None:
                  if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                      self.scheduler.step(val_loss if val_loss is not None else train_loss)
                  else:
                      self.scheduler.step()

              # ---- Logging ----
              self.history["train_loss"].append(train_loss)
              self.history["train_acc"].append(train_acc)
              self.history["val_loss"].append(val_loss)
              self.history["val_acc"].append(val_acc)

              logs = {
                  "train_loss": train_loss,
                  "train_acc": train_acc,
                  "val_loss": val_loss,
                  "val_acc": val_acc
              }

              for cb in callbacks:
                  cb.on_epoch_end(self, logs)

              msg = f"{epoch} | train_loss:{train_loss:.4f}"

              if train_acc is not None:
                  msg += f" | train_acc:{train_acc*100:.2f}%"

              if val_loss is not None:
                  msg += f" | val_loss:{val_loss:.4f}"

              if val_acc is not None:
                  msg += f" | val_acc:{val_acc*100:.2f}%"

              tqdm.write(msg)

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not built. Call build() first.")

        self.model.eval()
        X = self._to_tensor(X)

        with torch.no_grad():
            outputs = self(X)

        return outputs.cpu().numpy()

    def save(self, path):
        embeddings_state = None
        if hasattr(self, 'embeddings'):
            embeddings_state = self.embeddings.state_dict()

        torch.save({
            "model_state": self.model.state_dict(),
            "embeddings_state": embeddings_state,
            "optimizer_state": self.optimizer.state_dict(),
            "history": self.history
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])

        if "embeddings_state" in checkpoint and checkpoint["embeddings_state"] is not None:
            if hasattr(self, 'embeddings'):
                self.embeddings.load_state_dict(checkpoint["embeddings_state"])
            else:
                print("Warning: Checkpoint has embeddings, but current model does not.")

        if "optimizer_state" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]

        self.model.eval()

    def summary(self):

        total_params = 0

        print("\nModel Summary")
        print("-" * 40)

        for layer in self.model:

            params = sum(p.numel() for p in layer.parameters())
            total_params += params

            print(f"{layer.__class__.__name__:15} | params: {params}")

        print("-" * 40)
        print(f"Total parameters: {total_params}\n")

class CNN(BaseModel):
    def __init__(self, in_channels, input_size=None, **kwargs):
        super().__init__(**kwargs)
        self.current_channels = in_channels
        self.current_size = input_size  # (H, W)
        self.flatten_dim = None
        self.layers = nn.ModuleList()
        self.pool_types = {
            "max": nn.MaxPool2d,
            "avg": nn.AvgPool2d
        }

    def add_filter(self, out_channels, kernel_size=3, stride=1, padding=1, **kwargs):
        conv = nn.Conv2d(self.current_channels, out_channels, kernel_size, stride, padding)

        block = self._apply_utils(conv, **kwargs, norm_dim=out_channels)
        self.layers.append(block)

        self.current_channels = out_channels

        if self.current_size is not None:
            H, W = self.current_size
            H = (H + 2 * padding - kernel_size) // stride + 1
            W = (W + 2 * padding - kernel_size) // stride + 1
            self.current_size = (H, W)

        return self

    def add_pool(self, pool_type="max", kernel_size=2, stride=2):
        pool_layer = self.pool_types[pool_type](kernel_size=kernel_size, stride=stride)
        self.layers.append(pool_layer)

        # Update spatial size
        if self.current_size is not None:
            H, W = self.current_size
            H = (H - kernel_size) // stride + 1
            W = (W - kernel_size) // stride + 1
            self.current_size = (H, W)

        return self

    def add_flatten(self):
        self.layers.append(nn.Flatten())

        if self.current_size is not None:
            H, W = self.current_size
            self.flatten_dim = self.current_channels * H * W

        return self

    def add_fc(self, n_units, **kwargs):
        if self.flatten_dim is None:
            raise ValueError("Call add_flatten() before add_fc()")
        fc = nn.Linear(self.flatten_dim, n_units)
        block = self._apply_utils(fc, **kwargs, norm_dim=n_units)
        self.layers.append(block)
        self.flatten_dim = n_units
        return self

    def forward(self, x):
        return self.model(x)

def plot_img(X, index):
    first_image = X[index].astype(np.float32)  # ensure compatible dtype

    # If grayscale
    if first_image.ndim == 3 and first_image.shape[2] == 1:
        first_image = first_image[:, :, 0]

    plt.imshow(first_image, cmap="gray" if first_image.ndim == 2 else None)
    plt.axis("off")
    plt.show()

def conv_block(model, out_channels, dropout):
    # First conv
    model.add_filter(
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        activation="relu",
        norm="batch2d"
    )

    # Second conv (IMPORTANT)
    model.add_filter(
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        activation="relu",
        norm="batch2d"
    )

    # Pool AFTER feature extraction
    model.add_pool(pool_type="max", kernel_size=2, stride=2)

    # Optional dropout AFTER pooling (better placement)
    if dropout > 0:
        model.layers.append(nn.Dropout2d(dropout))

def build_cnn(model, n_classes, size="128"):
    if size == "128":
        channels = [32, 64, 128, 256]
        dropouts = [0.1, 0.1, 0.15, 0.2]
        fc_layers = [512, 256]

    elif size == "64":
        channels = [32, 64, 128, 256]
        dropouts = [0.2, 0.25, 0.3, 0.35]
        fc_layers = [256, 128]

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

    # X_train_empty_indices = [94, 5313, 7655, 7932, 10878, 10920, 11036, 14827, 16820, 17286, 17397, 17483, 18601, 19430]
    X_train_empty_indices = [
        94, 5313, 7655, 7932, 10878, 10920, 11036, 14827, 16820, 17286,
        17397, 17483, 18601, 19430, 20023, 25242, 27584, 27861, 30807,
        30849, 30965, 34756, 36749, 37215, 37326, 37412, 38530, 39359,
        39952, 45171, 47513, 47790, 50736, 50778, 50894, 54685, 56678,
        57144, 57255, 57341, 58459, 59288, 59881, 65100, 67442, 67719,
        70665, 70707, 70823, 74614, 76607, 77073, 77184, 77270, 78388,
        79217
    ]
    X_test_empty_indices = [1151, 1255, 3044, 4237, 4427, 4709]

    y_train = pd.read_csv(f"/content/drive/MyDrive/csv/y_augmented.csv", usecols=["label"])

    data = np.load('/content/data_64_augmented.npz')
    data_test = np.load('/content/data_64.npz')

    X_train = data['X_train']
    X_test  = data_test['X_test']
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
    X_train = X_train.astype('float32') / 255.0

    print("min/max:", X_train.min(), X_train.max())
    print("mean:", X_train.mean())

    model = CNN(
        in_channels=3,
        input_size=(X_train.shape[2], X_train.shape[3]),
        cost="cce",
        lr=3e-4,
        weight_decay=1e-3
    )

    build_cnn(model, n_classes, size="64")

    model.build()

    model.set_scheduler("reduce_on_plateau", mode="min", patience=5, factor=0.5)

    nn_data_path = "/content/drive/MyDrive/models"
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
            EarlyStopping(patience=25),
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

    output_path = "/content/drive/MyDrive/outputs"
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(f"{output_path}/cnn.csv", index=False)


if __name__ == "__main__":
    main()
    print("completed")