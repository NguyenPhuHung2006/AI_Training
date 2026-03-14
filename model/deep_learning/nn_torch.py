import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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


class NeuralNetwork(nn.Module):

    def __init__(self, n_inputs, cost, lr=0.001, device=None):
        super().__init__()

        self.n_inputs = n_inputs
        self.layers = []

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.lr = lr

        if cost not in COST_FUNCTIONS:
            raise ValueError(f"Unknown cost function: {cost}")

        self.criterion = COST_FUNCTIONS[cost]()

        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

    # -------------------------
    # Architecture
    # -------------------------

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

    # -------------------------
    # Utilities
    # -------------------------

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

    # -------------------------
    # Training
    # -------------------------

    def fit(
        self,
        X,
        y,
        epochs=100,
        batch_size=64,
        check_every=1,
        val_split=0.0,
        early_stopping=None,
        scheduler=None,
        checkpoint_path=None
    ):

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

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in tqdm(range(epochs), desc="Training"):

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

            if epoch % check_every == 0:

                msg = f"{epoch} | train_loss:{train_loss:.4f}"

                if train_total > 0:
                    train_acc = train_correct / train_total
                    msg += f" | train_acc:{train_acc*100:.2f}%"

                if X_val is not None:

                    val_loss, val_acc = self._evaluate(X_val, y_val)

                    msg += f" | val_loss:{val_loss:.4f}"

                    if val_acc is not None:
                        msg += f" | val_acc:{val_acc*100:.2f}%"

                    if early_stopping:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            if checkpoint_path:
                                self.save(checkpoint_path)
                        else:
                            patience_counter += 1

                            if patience_counter >= early_stopping:
                                print("Early stopping triggered")
                                break

                tqdm.write(msg)
                
            self.history["train_loss"].append(train_loss)

            if train_total > 0:
                self.history["train_acc"].append(train_acc)

            if X_val is not None:
                self.history["val_loss"].append(val_loss)

                if val_acc is not None:
                    self.history["val_acc"].append(val_acc)

    # -------------------------
    # Prediction
    # -------------------------

    def predict(self, X):

        self.model.eval()

        X = self._to_tensor(X)

        with torch.no_grad():
            outputs = self.model(X)

        return outputs.cpu().numpy()

    # -------------------------
    # Model Summary
    # -------------------------

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

    # -------------------------
    # Save / Load
    # -------------------------

    def save(self, path):

        torch.save(self.model.state_dict(), path)

    def load(self, path):

        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )

        self.model.eval()