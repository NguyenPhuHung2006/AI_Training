import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

    def build(self, params):
        self.optimizer = optim.Adam(params, weight_decay=self.weight_decay)
        return self
    
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

    def fit(self, X, y, epochs=10, batch_size=64, val_split=0.1, callbacks=None):
        callbacks = callbacks or []

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            if isinstance(self.criterion, nn.CrossEntropyLoss):
                y = torch.tensor(y, dtype=torch.long)
            else:
                y = torch.tensor(y, dtype=torch.float32)

        full_dataset = NumpyDataset(X, y)

        if val_split > 0:
            indices = list(range(len(X)))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_split,
                random_state=42,
                stratify=y.cpu().numpy() if isinstance(self.criterion, nn.CrossEntropyLoss) else None
            )
            train_dataset = Subset(full_dataset, train_idx)
            val_dataset = Subset(full_dataset, val_idx)
        else:
            train_dataset = full_dataset
            val_dataset = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )

        for cb in callbacks:
            cb.on_train_begin(self)

        for epoch in tqdm(range(epochs), desc="Training"):
            if self.stop_training:
                break

            self.train()

            total_loss = 0
            train_correct = 0
            train_total = 0

            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

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

            val_loss = None
            val_acc = None

            if val_loader is not None:
                self.eval()
                v_loss = 0
                v_correct = 0
                v_total = 0

                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device, non_blocking=True)
                        yb = yb.to(self.device, non_blocking=True)

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

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss is not None else train_loss)
                else:
                    self.scheduler.step()

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
        if self.optimizer is None:
            raise RuntimeError("Model not built. Call build() first.")
        
        self.eval()

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        X = X.to(self.device)

        with torch.no_grad():
            outputs = self(X)

        return outputs.cpu().numpy()

    def save(self, path):
        save_dict = {
            "model_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "history": self.history
        }
        if self.scheduler:
            save_dict["scheduler_state"] = self.scheduler.state_dict()
        torch.save(save_dict, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        self.eval()