from .BaseModel import BaseModel
import torch.nn as nn

class MLP(BaseModel):
    def __init__(self, input_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.current_dim = input_dim
        
    def add_embedding(self, vocab_size, embed_dim, padding_idx=None):
        embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.layers.append(embedding)
        self.current_dim = embed_dim
        return self

    def add_layer(self, n_units, activation="relu", dropout=0.0, norm=None, init="he"):
        if self.current_dim is None:
            raise ValueError("You must provide input_dim in __init__ or call add_embedding first.")
        
        linear = nn.Linear(self.current_dim, n_units)
        block = self._apply_utils(linear, activation, dropout, "standard", norm, n_units, init)
        self.layers.append(block)
        self.current_dim = n_units
        return self