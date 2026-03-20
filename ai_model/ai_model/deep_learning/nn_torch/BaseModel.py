import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

    def _apply_utils(self, module, activation, dropout, dropout_type, norm, norm_dim, init):
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
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return self

    def _to_tensor(self, X, y=None):
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        if y is None:
            return X
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            y_dtype = torch.long
        else:
            y_dtype = torch.float32
        return X, torch.as_tensor(y, dtype=y_dtype).to(self.device)
    
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

    def fit(self, X, y, epochs=10, batch_size=64, val_split=0.1, callbacks=None):
        callbacks = callbacks or []
        if val_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_split
            )
        else:
            X_train, y_train = X, y
            X_val = y_val = None
            
        X_train, y_train = self._to_tensor(X_train, y_train)
        if X_val is not None:
            X_val, y_val = self._to_tensor(X_val, y_val)
        
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )
        
        for cb in callbacks:
            cb.on_train_begin(self)

        for epoch in tqdm(range(epochs), desc="Training"):
            if self.stop_training: 
                break
            
            self.model.train()
            
            total_loss = 0
            train_correct = 0
            train_total = 0
            
            for xb, yb in loader:
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
                acc = self._accuracy(out, yb)
                if acc is not None:
                    correct, total = acc
                    train_correct += correct
                    train_total += total
                    
            if self.scheduler is not None:
                self.scheduler.step()
                
            train_loss = total_loss / len(loader)
            train_acc = train_correct / train_total if train_total > 0 else None

            val_loss = None
            val_acc = None
            if X_val is not None:
                val_loss, val_acc = self._evaluate(X_val, y_val)
                
            # callback
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
            outputs = self.model(X)
            
        return outputs.cpu().numpy()
    
    def save(self, path):
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "history": self.history
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        
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