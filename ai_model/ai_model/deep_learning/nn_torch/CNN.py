from .BaseModel import BaseModel
import torch.nn as nn

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

    def add_filter(self, out_channels, kernel_size=3, stride=1, padding=1,
                  activation="relu", dropout=0.0, norm=None, init="he"):
        conv = nn.Conv2d(self.current_channels, out_channels, kernel_size, stride, padding)
        
        block = self._apply_utils(conv, activation, dropout, "standard", norm, out_channels, init)
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

    def add_fc(self, out_features, activation="relu", dropout=0.0):
        if self.flatten_dim is None:
            raise ValueError("Call add_flatten() before add_fc()")
        fc = nn.Linear(self.flatten_dim, out_features)
        block = self._apply_utils(fc, activation, dropout, "standard", None, out_features, "he")
        self.layers.append(block)
        self.flatten_dim = out_features
        return self

    def forward(self, x):
        return self.model(x)