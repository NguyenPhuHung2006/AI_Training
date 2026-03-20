from .BaseModel import BaseModel
import torch.nn as nn

class CNN(BaseModel):
    def __init__(self, in_channels, **kwargs):
        super().__init__(**kwargs)
        self.current_channels = in_channels

    def add_layer(self, out_channels, kernel_size=3, stride=1, padding=1, 
                  activation="relu", dropout=0.0, norm=None, init="he"):
        conv = nn.Conv2d(self.current_channels, out_channels, kernel_size, stride, padding)
        # Use batch2d for CNNs
        block = self._apply_utils(conv, activation, dropout, "standard", norm, out_channels, init)
        self.layers.append(block)
        self.current_channels = out_channels
        return self
        
    def add_pool(self, pool_type="max", kernel_size=2, stride=2):
        pool_layer = self.pool_types[pool_type](kernel_size=kernel_size, stride=stride)
        self.layers.append(pool_layer)
        return self
        
    def add_flatten(self):
        self.layers.append(nn.Flatten())
        return self