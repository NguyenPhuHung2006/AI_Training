import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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