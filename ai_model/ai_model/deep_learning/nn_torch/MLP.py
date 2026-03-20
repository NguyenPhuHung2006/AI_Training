from .BaseModel import BaseModel
import torch.nn as nn

class MLP(BaseModel):
    def __init__(self, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.current_dim = input_dim

    def add_layer(self, n_units, activation="relu", dropout=0.0, norm=None, init="he"):
        linear = nn.Linear(self.current_dim, n_units)
        block = self._apply_utils(linear, activation, dropout, "standard", norm, n_units, init)
        self.layers.append(block)
        self.current_dim = n_units
        return self