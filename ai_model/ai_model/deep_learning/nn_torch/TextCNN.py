from .BaseModel import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(BaseModel):
    def __init__(self, in_channels=None, **kwargs):
        super().__init__(**kwargs)

        self.embedding = None
        self.pad_token = None
        self.in_channels = in_channels

        self.conv_blocks = nn.ModuleList()
        self.pool_types = {
            "max": lambda x: F.max_pool1d(x, x.size(2)).squeeze(2),
            "avg": lambda x: F.avg_pool1d(x, x.size(2)).squeeze(2),
        }
        self.pool_type = "max"

        self.flatten_dim = 0
        self.pool_dropout = 0

    def add_embedding(self, vocab_size, embed_dim):
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.pad_token)
        self.in_channels = embed_dim
        return self
    
    def set_embedding_matrix(self, embedding_matrix, freeze=False):
        if not isinstance(embedding_matrix, torch.Tensor):
            raise TypeError("embedding_matrix must be a torch.Tensor")

        vocab_size, embed_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze,
            padding_idx=self.pad_token
        )

        self.in_channels = embed_dim
        return self

    def add_filter(self, out_channels, kernel_size, **kwargs):
        conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        )

        block = self._apply_utils(conv, **kwargs, norm_dim=out_channels)

        self.conv_blocks.append(block)
        self.flatten_dim += out_channels

        return self

    def add_pool(self, pool_type, pool_dropout=0):
        if pool_type not in self.pool_types:
            raise ValueError(f"Invalid pool type: {pool_type}")
        self.pool_type = pool_type
        self.pool_dropout = pool_dropout
        return self

    def add_fc(self, n_units, **kwargs):
        fc = nn.Linear(self.flatten_dim, n_units)
        block = self._apply_utils(fc, **kwargs, norm_dim=n_units)

        self.layers.append(block)
        self.flatten_dim = n_units
        return self

    def build(self):
        self.to(self.device)
        embed_params = list(self.embedding.parameters()) if self.embedding else []
        embed_ids = set(map(id, embed_params))
        
        main_params = [p for p in self.parameters() if id(p) not in embed_ids]
        
        if embed_params:
            params = [
                {'params': main_params, 'lr': self.lr},
                {'params': embed_params, 'lr': self.lr * 0.1}
            ]
        else:
            params = [{'params': main_params, 'lr': self.lr}]
        
        super().build(params)

    # conv1d: 
    # -> input:  (batch_size, in_channels, sequence_length)
    # -> output: (batch, out_channels, new_seq_len)
    # x: 
    # -> (batch, seq_len) or (batch, seq_len, embed_dim) if embedded
    # embedding:
    # -> input:  (batch, seq_len)
    # -> output: (batch, seq_len, embed_dim)
    def forward(self, x):
        if self.embedding is not None:
            # (batch, seq_len) -> (batch, seq_len, embed_dim)
            x = self.embedding(x.long())

        # -> (batch, embed_dim, seq_len)
        x = x.transpose(1, 2)

        conv_outputs = []

        if len(self.conv_blocks) == 0:
            raise ValueError("No convolution filters added")
        for block in self.conv_blocks:
            c = block(x)  # (batch, out_channels, L)
            
            # global pooling
            c = self.pool_types[self.pool_type](c)  # (batch, out_channels)

            conv_outputs.append(c)

        # concat all branches
        out = torch.cat(conv_outputs, dim=1)  # (batch, total_channels)
        
        if self.pool_dropout > 0:
            out = F.dropout(out, p=self.pool_dropout, training=self.training)

        # fully connected layers
        for layer in self.layers:
            out = layer(out)

        return out