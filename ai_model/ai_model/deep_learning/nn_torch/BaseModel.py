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

    def fit(self, X, y, epochs=10, batch_size=64, val_split=0.1, callbacks=None):
        callbacks = callbacks or []
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split)
        X_train, y_train = self._to_tensor(X_train, y_train)
        X_val, y_val = self._to_tensor(X_val, y_val)
        
        dataset = TensorDataset(X_train, y_train)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )

        for epoch in tqdm(range(epochs), desc="Training"):
            if self.stop_training: 
                break
            self.model.train()
            total_loss = 0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            val_loss = self._evaluate(X_val, y_val)
            print(f"Epoch {epoch} | Train Loss: {total_loss/len(loader):.4f} | Val Loss: {val_loss:.4f}")
            for cb in callbacks: cb.on_epoch_end(self, {"val_loss": val_loss})

    def _evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            return self.criterion(self.model(X), y).item()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad(): return self.model(self._to_tensor(X)).cpu().numpy()