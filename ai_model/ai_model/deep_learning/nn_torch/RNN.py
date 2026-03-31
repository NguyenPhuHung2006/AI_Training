import torch
import torch.nn as nn
from .BaseModel import BaseModel

class RNN(BaseModel):
    def __init__(self, 
                 input_size=None, 
                 rnn_type="lstm", 
                 mode="many_to_one", 
                 use_packing=False, 
                 pad_id=None,
                 **kwargs
                ):
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

        self.embedding = None
        self.output_size = None
        
        self.attention = None

        self.pad_id = pad_id
        self.use_packing = use_packing

    def add_embedding(self, vocab_size, embed_dim):
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.pad_id)
        self.input_size = embed_dim
        return self
    
    def set_embedding_matrix(self, embedding_matrix, freeze=False):
        if not isinstance(embedding_matrix, torch.Tensor):
            raise TypeError("embedding_matrix must be a torch.Tensor")

        vocab_size, embed_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze,
            padding_idx=self.pad_id
        )

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
    
    def add_attention(self):
        self.attention = nn.Linear(self.input_size, 1)
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

    def forward(self, x):
        x = x.to(self.device)
        
        mask = (x != self.pad_id)  # (batch, seq_len)
        lengths = mask.sum(dim=1).cpu()

        if self.embedding is not None and x.dim() == 2:
            x = self.embedding(x.long())

        if self.use_packing:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            outputs_packed, hidden = self.rnn(x_packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs_packed, batch_first=True)
        else:
            outputs, hidden = self.rnn(x)

        if self.mode == "many_to_one":
            if self.attention is not None:
                # Attention
                scores = self.attention(outputs).squeeze(-1)  # (batch, seq_len)
                scores = scores.masked_fill(~mask, float('-inf'))

                weights = torch.softmax(scores, dim=1)
                weights = weights.unsqueeze(-1) # (batch, seq_len, 1)

                x = torch.sum(weights * outputs, dim=1)

            else:
                if isinstance(hidden, tuple):  # LSTM
                    h = hidden[0]
                else:
                    h = hidden

                if self.rnn.bidirectional:
                    batch_size = h.size(1)
                    h = h.view(self.rnn.num_layers, 2, batch_size, self.rnn.hidden_size)
                    x = torch.cat([h[-1, 0], h[-1, 1]], dim=1)
                else:
                    x = h[-1]

        # ---- FC ----
        for fc in self.fc_layers:
            x = fc(x)

        return x