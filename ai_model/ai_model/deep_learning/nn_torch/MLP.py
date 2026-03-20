from .BaseModel import BaseModel
import torch
import torch.nn as nn

class MLP(BaseModel):
    def __init__(self, input_dim=0, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = nn.ModuleDict()
        self.cat_cols = []
        self.num_cols = []
        self.total_embed_dim = 0
        self.current_dim = input_dim

    def add_embedding(self, col_index, vocab_size, embed_dim):
        self.embeddings[str(col_index)] = nn.Embedding(vocab_size, embed_dim)
        self.cat_cols.append(col_index)
        self.total_embed_dim += embed_dim
        self.current_dim = self.total_embed_dim + len(self.num_cols)
        return self

    def set_numerical_cols(self, col_indices):
        self.num_cols = col_indices
        self.current_dim = self.total_embed_dim + len(self.num_cols)
        return self

    def add_layer(self, n_units, **kwargs):
        layer = nn.Linear(self.current_dim, n_units)
        self.layers.append(self._apply_utils(layer, **kwargs, norm_dim=n_units))
        self.current_dim = n_units
        return self

    def forward(self, x):
        features = []
        for col in self.cat_cols:
            features.append(self.embeddings[str(col)](x[:, col].long()))
        
        if self.num_cols:
            features.append(x[:, self.num_cols].float())
        
        x_combined = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        return self.model(x_combined)