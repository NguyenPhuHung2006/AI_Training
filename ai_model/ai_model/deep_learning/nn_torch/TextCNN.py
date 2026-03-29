from .BaseModel import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(BaseModel):
    def __init__(self, input_size=None, **kwargs):
        super().__init__(**kwargs)

        self.embedding = None
        self.embed_dim = None
        self.pad_id = None

        self.flatten_dim = None

    def add_embedding(self, vocab_size, embed_dim, pad_id=None):
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.embed_dim = embed_dim
        self.pad_id = pad_id
        return self

    def add_filter(self, out_channels, kernel_size, **kwargs):
        if self.embed_dim is None:
            raise ValueError("Call add_embedding() before add_filter()")

        conv = nn.Conv1d(
            in_channels=self.embed_dim,
            out_channels=out_channels,
            kernel_size=kernel_size
        )

        block = self._apply_utils(conv, **kwargs, norm_dim=out_channels)
        self.conv_blocks.append(block)

        return self

    # ✅ Pool control (optional)
    def add_pool(self, pool_type="max"):
        if pool_type != "max":
            raise ValueError("TextCNN only supports global max pooling")
        self.use_pool = True
        return self

    # ✅ Build conv output size
    def build_conv_output(self):
        total_channels = 0

        for block in self.conv_blocks:
            conv = block[0] if isinstance(block, nn.Sequential) else block
            total_channels += conv.out_channels

        self.flatten_dim = total_channels
        return self

    # ✅ FC layer (reuse your style)
    def add_fc(self, n_units, **kwargs):
        if self.flatten_dim is None:
            raise ValueError("Call build_conv_output() before add_fc()")

        fc = nn.Linear(self.flatten_dim, n_units)
        block = self._apply_utils(fc, **kwargs, norm_dim=n_units)

        self.layers.append(block)
        self.flatten_dim = n_units
        return self

    def build(self):
        self.conv_blocks = nn.ModuleList(self.conv_blocks)
        super().build()

    def forward(self, x):
        # x: (batch, seq_len)

        if self.embedding is None:
            raise ValueError("Embedding not defined")

        x = self.embedding(x)          # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)          # (batch, embed_dim, seq_len)

        conv_outputs = []

        for block in self.conv_blocks:
            c = block(x)               # (batch, out_channels, L)
            c = F.relu(c)

            if self.use_pool:
                c = F.max_pool1d(c, c.size(2))  # global max pool

            c = c.squeeze(2)           # (batch, out_channels)
            conv_outputs.append(c)

        out = torch.cat(conv_outputs, dim=1)  # (batch, total_channels)

        for layer in self.layers:
            out = layer(out)

        return out