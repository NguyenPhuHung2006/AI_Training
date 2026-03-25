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

    def add_seq2seq(self, hidden_size, num_layers=1):
        self.encoder = self.rnn_class(
            self.input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.decoder = self.rnn_class(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

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
        """
        x: (B, T) or (B, T, F)
        target: for seq2seq (teacher forcing)
        """

        # ---- Embedding ----
        if self.embeddings is not None and x.dim() == 2:
            x = self.embeddings(x)

        # ---- SEQ2SEQ ----
        if self.mode == "seq2seq":
            enc_out, hidden = self.encoder(x)

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

        # -------------------------------------------------
        # MODE HANDLING
        # -------------------------------------------------

        if self.mode == "many_to_one":
            x = x[:, -1, :]  # (B, H)

        elif self.mode == "one_to_many":
            # keep full sequence
            pass

        elif self.mode == "many_to_many":
            # keep full sequence
            pass

        elif self.mode == "seq2seq":
            # already handled
            pass

        # -------------------------------------------------
        # FC layers
        # -------------------------------------------------
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

        for i, fc in enumerate(self.fc_layers):
            params = sum(p.numel() for p in fc.parameters())
            print(f"{f'FC {i}':15} | params: {params}")
            total_params += params

        print("-" * 50)
        print(f"Total parameters: {total_params}\n")
        
    def generate(self, start_tokens, max_len=20, temperature=1.0):
        self.eval()

        if not torch.is_tensor(start_tokens):
            start_tokens = torch.tensor(start_tokens, device=self.device)

        x = start_tokens.unsqueeze(0)
        hidden = None

        outputs = start_tokens.tolist()

        for _ in range(max_len):
            logits = self(x)
            logits = logits[:, -1, :] / temperature

            probs = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1)

            outputs.append(token.item())
            x = torch.cat([x, token], dim=1)

        return outputs