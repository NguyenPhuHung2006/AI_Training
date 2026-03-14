import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

# =========================
# Config
# =========================

COST_FUNCTIONS = {
    "cce": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "bce": nn.BCEWithLogitsLoss
}

ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh
}


# =========================
# Callback System
# =========================

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

            print(f"Saved best model → {self.path}")


# =========================
# Neural Network
# =========================

class NeuralNetwork(nn.Module):

    def __init__(self, n_inputs, cost="cce", lr=0.001, device=None):
        super().__init__()

        self.n_inputs = n_inputs
        self.layers = []

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if cost not in COST_FUNCTIONS:
            raise ValueError(f"Unknown cost function: {cost}")

        self.criterion = COST_FUNCTIONS[cost]()
        self.lr = lr

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

    # =========================
    # Architecture
    # =========================

    def add_layer(self, n_units, activation=None):

        self.layers.append(nn.Linear(self.n_inputs, n_units))

        if activation:
            if activation not in ACTIVATIONS:
                raise ValueError(f"Unknown activation: {activation}")

            self.layers.append(ACTIVATIONS[activation]())

        self.n_inputs = n_units

    def build(self):

        self.model = nn.Sequential(*self.layers).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

    # =========================
    # Utilities
    # =========================

    def _to_tensor(self, X, y=None):

        X = torch.tensor(X, dtype=torch.float32)

        if y is None:
            return X.to(self.device)

        if isinstance(self.criterion, nn.CrossEntropyLoss):
            y = torch.tensor(y, dtype=torch.long)
        else:
            y = torch.tensor(y, dtype=torch.float32)

        return X.to(self.device), y.to(self.device)

    def _accuracy(self, outputs, y):

        if isinstance(self.criterion, nn.CrossEntropyLoss):
            preds = torch.argmax(outputs, dim=1)

        elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
            preds = (torch.sigmoid(outputs) > 0.5).float()

        else:
            return None

        correct = (preds == y).sum().item()
        total = y.size(0)

        return correct, total

    def _evaluate(self, X, y):

        self.model.eval()

        with torch.no_grad():

            outputs = self.model(X)

            loss = self.criterion(outputs, y).item()

            acc = self._accuracy(outputs, y)

            if acc is None:
                return loss, None

            correct, total = acc
            return loss, correct / total

    # =========================
    # Training
    # =========================

    def fit(
        self,
        X,
        y,
        epochs=100,
        batch_size=64,
        val_split=0.0,
        scheduler=None,
        callbacks=None
    ):

        callbacks = callbacks or []

        if val_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_split
            )
        else:
            X_train, y_train = X, y
            X_val = y_val = None

        X_train, y_train = self._to_tensor(X_train, y_train)

        dataset = TensorDataset(X_train, y_train)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        if X_val is not None:
            X_val, y_val = self._to_tensor(X_val, y_val)

        if scheduler:
            self.scheduler = scheduler(self.optimizer)

        for cb in callbacks:
            cb.on_train_begin(self)

        for epoch in tqdm(range(epochs), desc="Training"):

            if self.stop_training:
                break

            self.model.train()

            total_loss = 0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in loader:

                self.optimizer.zero_grad()

                outputs = self.model(X_batch)

                loss = self.criterion(outputs, y_batch)

                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

                acc = self._accuracy(outputs, y_batch)

                if acc is not None:
                    correct, total = acc
                    train_correct += correct
                    train_total += total

            if self.scheduler:
                self.scheduler.step()

            train_loss = total_loss / len(loader)
            train_acc = train_correct / train_total if train_total > 0 else None

            val_loss = None
            val_acc = None

            if X_val is not None:
                val_loss, val_acc = self._evaluate(X_val, y_val)

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

    # =========================
    # Prediction
    # =========================

    def predict(self, X):

        self.model.eval()

        X = self._to_tensor(X)

        with torch.no_grad():
            outputs = self.model(X)

        return outputs.cpu().numpy()

    # =========================
    # Model Summary
    # =========================

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

    # =========================
    # Save / Load
    # =========================

    def save(self, path):

        torch.save(self.model.state_dict(), path)

    def load(self, path):

        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )

        self.model.eval()

def process_dates(df):
    df = df.copy(deep=False)
    
    df["date_time"] = pd.to_datetime(df["date_time"], format="%d-%m-%y %H:%M", errors="coerce")
    df["srch_ci"] = pd.to_datetime(df["srch_ci"], format="%d-%m-%y", errors="coerce")
    df["srch_co"] = pd.to_datetime(df["srch_co"], format="%d-%m-%y", errors="coerce")
    
    df = df.assign(
        search_month = df["date_time"].dt.month,
        stay_days = (df["srch_co"] - df["srch_ci"]).dt.days.clip(lower=0),
        booking_lead = (df["srch_ci"] - df["date_time"]).dt.days.clip(lower=0),
        search_weekday = df["date_time"].dt.weekday
    )

    df = df.drop(["date_time", "srch_ci", "srch_co"], axis=1)

    return df

def fill_na_median(df_train, df_test):
    df_train = df_train.copy(deep=False)
    df_test = df_test.copy(deep=False)
    
    for col in df_train.columns:
        if pd.api.types.is_numeric_dtype(df_train[col]):
            median = df_train[col].median()
            df_train[col] = df_train[col].fillna(median)
            df_test[col] = df_test[col].fillna(median)
            
    return df_train, df_test

def log_normalize(df, cols):
    df = df.copy(deep=False)

    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
            df[col] = np.log1p(np.clip(df[col], a_min=0, a_max=None))

    return df


def standardization(df_train, df_test, cols):
    df_train = df_train.copy(deep=False)
    df_test = df_test.copy(deep=False)

    for col in cols:
        if pd.api.types.is_numeric_dtype(df_train[col]):

            mean = df_train[col].mean()
            std = df_train[col].std()

            if std == 0:
                std = 1.0

            df_train[col] = (df_train[col] - mean) / std
            df_test[col] = (df_test[col] - mean) / std

    return df_train, df_test
    
def read_data(train_path, test_path, dest_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_dest = pd.read_csv(dest_path)
    
    df_train = df_train.merge(df_dest, on="srch_destination_id", how="left")
    df_test = df_test.merge(df_dest, on="srch_destination_id", how="left")
    
    df_train = process_dates(df_train)
    df_test = process_dates(df_test)

    y_train = df_train["hotel_cluster"]

    df_train = df_train.drop(columns=["hotel_cluster", "user_id"])
    df_test = df_test.drop(columns=["id", "user_id"])
    
    log_cols = ["orig_destination_distance", "cnt"]
    df_train = log_normalize(df_train, log_cols)
    df_test = log_normalize(df_test, log_cols)
    
    std_cols = [
        "orig_destination_distance",
        "srch_adults_cnt",
        "srch_children_cnt",
        "srch_rm_cnt",
        "cnt"
    ]
    
    df_train, df_test = standardization(df_train, df_test, std_cols)
    
    df_train, df_test = fill_na_median(df_train, df_test)
        
    df_test = df_test.reindex(columns=df_train.columns)
        
    X_train = df_train.astype("float32").to_numpy()
    y_train = y_train.to_numpy()
    X_test = df_test.astype("float32").to_numpy()
        
    return X_train, y_train, X_test    
            
def main():
    X_train, y_train, X_test = read_data("new_dataset/train.csv", "new_dataset/test.csv", "new_dataset/destinations.csv")
        
    print("data preprocessing completed")
    
    num_classes = 100
    y_train = np.eye(num_classes)[y_train]
    
    model = NeuralNetwork(
        n_inputs=X_train.shape[1],
        cost_function=CCE()
    )
    
    model.add_layer(256, Relu())
    model.add_layer(128, Relu())
    model.add_layer(100, Softmax())
    
    # retain the nn
    model.load(f"nn_data/nn_numpy_0.npz")
    
    model.fit_with_validation(X_train, y_train, check_every=5, n_iterations=1000, test_size=0.05, lr=0.5)
    
    y_test = model.predict(X_test)
    
    i = 0
    nn_data_path = "nn_data"
    
    os.makedirs(nn_data_path, exist_ok=True)

    while os.path.exists(f"{nn_data_path}/nn_numpy_{i}.npz"):
        i += 1

    model.save(f"{nn_data_path}/nn_numpy_{i}.npz")

    top5 = np.argsort(-y_test, axis=1)[:, :5]
    labels = np.apply_along_axis(lambda x: " ".join(map(str, x)), 1, top5)
        
    df = pd.DataFrame({
        "id": np.arange(0, len(labels)),
        "hotel_cluster": labels
    })
    
    output_path = "outputs/nn"
    
    os.makedirs(output_path, exist_ok=True)

    i = 0
    while os.path.exists(f"{output_path}/nn_numpy_{i}.csv"):
        i += 1

    df.to_csv(f"{output_path}/nn_numpy_{i}.csv", index=False)
        
if __name__ == '__main__':
    main()
    print("completed")