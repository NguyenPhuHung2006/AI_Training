import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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

        # seq2seq
        self.encoder = None
        self.decoder = None

        self.enc_bidirectional = False
        self.bridge_h = None
        self.bridge_c = None

        self.embeddings = None
        self.output_size = None
        
    def add_embedding(self, vocab_size, embed_dim, padding_idx=0):
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
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

    def add_seq2seq(self, hidden_size, num_layers=1, bidirectional_encoder=True):
        self.encoder = self.rnn_class(
            self.input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional_encoder
        )

        self.decoder = self.rnn_class(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.enc_bidirectional = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        if bidirectional_encoder:
            self.bridge_h = nn.Linear(hidden_size * 2, hidden_size)
            if self.rnn_class == nn.LSTM:
                self.bridge_c = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.bridge_h = None
            self.bridge_c = None

        self.input_size = hidden_size
        return self

    def add_fc(self, out_features, **kwargs):
        layer = nn.Linear(self.input_size, out_features)
        block = self._apply_utils(layer, **kwargs, norm_dim=out_features)

        self.fc_layers.append(block)
        self.input_size = out_features
        self.output_size = out_features
        return self

    def build(self):
        self.to(self.device)

        main_params = []

        if self.rnn:
            main_params += list(self.rnn.parameters())

        if self.encoder:
            main_params += list(self.encoder.parameters())

        if self.decoder:
            main_params += list(self.decoder.parameters())

        if self.bridge_h:
            main_params += list(self.bridge_h.parameters())

        if self.bridge_c:
            main_params += list(self.bridge_c.parameters())

        main_params += list(self.fc_layers.parameters())

        embed_params = list(self.embeddings.parameters()) if self.embeddings else []

        if embed_params:
            params_to_train = [
                {'params': main_params, 'lr': self.lr},
                {'params': embed_params, 'lr': self.lr * 10}
            ]
        else:
            params_to_train = [{'params': main_params, 'lr': self.lr}]

        self.optimizer = torch.optim.Adam(params_to_train, weight_decay=self.weight_decay)

        self.model = self
        return self

    def forward(self, x, lengths=None, target=None, hidden=None):
        # ---- Embedding ----
        if self.embeddings is not None and x.dim() == 2:
            x = self.embeddings(x)

        # ---- SEQ2SEQ ----
        if self.mode == "seq2seq":
            enc_out, hidden = self.encoder(x)

            if self.enc_bidirectional:
                B = x.size(0)

                # LSTM
                if isinstance(hidden, tuple):
                    h, c = hidden

                    h = h.view(self.num_layers, 2, B, self.hidden_size)
                    c = c.view(self.num_layers, 2, B, self.hidden_size)

                    h = torch.cat([h[:, 0], h[:, 1]], dim=-1)
                    c = torch.cat([c[:, 0], c[:, 1]], dim=-1)

                    h = self.bridge_h(h)
                    c = self.bridge_c(c)

                    hidden = (h, c)

                # GRU / RNN
                else:
                    h = hidden

                    h = h.view(self.num_layers, 2, B, self.hidden_size)
                    h = torch.cat([h[:, 0], h[:, 1]], dim=-1)

                    h = self.bridge_h(h)

                    hidden = h

            # ---- Decoder ----
            if target is not None:
                if self.embeddings is not None and target.dim() == 2:
                    target = self.embeddings(target)
                x, _ = self.decoder(target, hidden)
            else:
                x, _ = self.decoder(x, hidden)

        # ---- STANDARD RNN ----
        else:
            if lengths is not None:
                x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

            if self.rnn is not None:
                x, hidden = self.rnn(x, hidden) if hidden is not None else self.rnn(x)

            if lengths is not None:
                x, _ = pad_packed_sequence(x, batch_first=True)

        # ---- MODE HANDLING ----
        if self.mode == "many_to_one":
            x = x[:, -1, :]

        elif self.mode in ["one_to_many", "many_to_many", "seq2seq"]:
            pass

        # ---- FC ----
        for fc in self.fc_layers:
            x = fc(x)

        return x

    def summary(self):
        total_params = 0

        print("\nRNN Model Summary")
        print("-" * 50)

        if self.embeddings:
            params = sum(p.numel() for p in self.embeddings.parameters())
            print(f"{'Embedding':15} | params: {params}")
            total_params += params

        if self.rnn:
            params = sum(p.numel() for p in self.rnn.parameters())
            print(f"{'RNN':15} | params: {params}")
            total_params += params

        if self.encoder:
            params = sum(p.numel() for p in self.encoder.parameters())
            print(f"{'Encoder':15} | params: {params}")
            total_params += params

        if self.decoder:
            params = sum(p.numel() for p in self.decoder.parameters())
            print(f"{'Decoder':15} | params: {params}")
            total_params += params

        if self.bridge_h:
            params = sum(p.numel() for p in self.bridge_h.parameters())
            print(f"{'Bridge_h':15} | params: {params}")
            total_params += params

        if self.bridge_c:
            params = sum(p.numel() for p in self.bridge_c.parameters())
            print(f"{'Bridge_c':15} | params: {params}")
            total_params += params

        for i, fc in enumerate(self.fc_layers):
            params = sum(p.numel() for p in fc.parameters())
            print(f"{f'FC {i}':15} | params: {params}")
            total_params += params

        print("-" * 50)
        print(f"Total parameters: {total_params}\n")