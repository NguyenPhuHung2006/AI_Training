import torch
import torch.nn as nn
from .BaseModel import BaseModel

class RNN(BaseModel):
    def __init__(self, input_size=None, rnn_type="lstm", mode="many_to_one", **kwargs):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.mode = mode

        self.rnn_types = {
            "lstm": nn.LSTM,
            "gru": nn.GRU,
            "rnn": nn.RNN
        }
        self.rnn_class = self.rnn_types[rnn_type]

        self.rnn = None
        self.fc_layers = nn.ModuleList()

        self.embeddings = None
        self.output_size = None

        self.pad_token = None

    def add_embedding(self, vocab_size, embed_dim):
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=self.pad_token)
        self.input_size = embed_dim
        return self

    def add_rnn(self, hidden_size, num_layers=1, dropout=0.0, bidirectional=False):
        self.rnn = self.rnn_class(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.input_size = hidden_size * (2 if bidirectional else 1)
        return self

    def add_fc(self, out_features, **kwargs):
        layer = nn.Linear(self.input_size, out_features)
        block = self._apply_utils(layer, **kwargs, norm_dim=out_features)

        self.fc_layers.append(block)
        self.input_size = out_features
        self.output_size = out_features
        return self

    # def forward(self, x):
    #     x = x.to(self.device)
    #     lengths = (x != self.pad_token).sum(dim=1).cpu()

    #     # Embedding
    #     if self.embeddings is not None and x.dim() == 2:
    #         x = self.embeddings(x.long())

    #     x_packed = nn.utils.rnn.pack_padded_sequence(
    #         x, lengths, batch_first=True, enforce_sorted=False
    #     )

    #     _, hidden = self.rnn(x_packed)

    #     if isinstance(hidden, tuple):
    #         h_n = hidden[0]
    #     else:
    #         h_n = hidden

    #     batch_size = h_n.size(1)

    #     if self.rnn.bidirectional:
    #         h_n = h_n.view(self.rnn.num_layers, 2, batch_size, self.rnn.hidden_size)
    #         x = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=1)
    #     else:
    #         x = h_n[-1]
            
    #     for fc in self.fc_layers:
    #         x = fc(x)

    #     return x
    
    def forward(self, x):
        x = x.to(self.device)

        # Embedding
        if self.embeddings is not None and x.dim() == 2:
            x = self.embeddings(x.long())

        # RNN
        x, hidden = self.rnn(x)

        if self.mode == "many_to_one":
            if isinstance(hidden, tuple):  # LSTM
                h = hidden[0]
            else:
                h = hidden

            if self.rnn.bidirectional:
                h = h.view(self.rnn.num_layers, 2, x.size(0), self.rnn.hidden_size)
                x = torch.cat([h[-1, 0], h[-1, 1]], dim=1)
            else:
                x = h[-1]

        # FC layers
        for fc in self.fc_layers:
            x = fc(x)

        return x