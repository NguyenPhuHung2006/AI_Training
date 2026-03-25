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
        self.rnn_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        self.embeddings = None
        self.output_size = None  # needed for generation

    # -------------------------------------------------
    # Embedding
    # -------------------------------------------------
    def add_embedding(self, vocab_size, embed_dim, padding_idx=0):
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.input_size = embed_dim
        return self

    # -------------------------------------------------
    # RNN layers
    # -------------------------------------------------
    def add_rnn(self, hidden_size, num_layers=1, dropout=0.0, bidirectional=False):
        rnn = self.rnn_class(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.rnn_layers.append(rnn)
        self.input_size = hidden_size * (2 if bidirectional else 1)
        return self

    # -------------------------------------------------
    # FC layers (stack)
    # -------------------------------------------------
    def add_fc(self, out_features, **kwargs):
        layer = nn.Linear(self.input_size, out_features)
        block = self._apply_utils(layer, **kwargs, norm_dim=out_features)

        self.fc_layers.append(block)
        self.input_size = out_features
        self.output_size = out_features
        return self

    # -------------------------------------------------
    # Build
    # -------------------------------------------------
    def build(self):
        self.model = self

        main_params = list(self.rnn_layers.parameters()) + list(self.fc_layers.parameters())
        embed_params = list(self.embeddings.parameters()) if self.embeddings else []

        if embed_params:
            params_to_train = [
                {'params': main_params, 'lr': self.lr},
                {'params': embed_params, 'lr': self.lr * 10}
            ]
        else:
            params_to_train = [{'params': main_params, 'lr': self.lr}]

        self.optimizer = torch.optim.Adam(params_to_train, weight_decay=self.weight_decay)
        return self

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(self, x, lengths=None, hidden=None):
        """
        x: 
            - (B, T) if token ids
            - (B, T, F) if already embedded
        lengths: (B,) for variable-length sequences
        """

        # ---- Embedding ----
        if self.embeddings is not None and x.dim() == 2:
            x = self.embeddings(x)  # (B, T, E)

        # ---- Pack (for variable length) ----
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # ---- RNN ----
        for rnn in self.rnn_layers:
            x, hidden = rnn(x, hidden) if hidden is not None else rnn(x)

        # ---- Unpack ----
        if lengths is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)

        # ---- Mode handling ----
        if self.mode == "many_to_one":
            if lengths is not None:
                # take last valid timestep
                idx = (lengths - 1).view(-1, 1, 1).expand(x.size(0), 1, x.size(2))
                x = x.gather(1, idx).squeeze(1)
            else:
                x = x[:, -1, :]

        # ---- FC stack ----
        for fc in self.fc_layers:
            x = fc(x)

        return x

    # -------------------------------------------------
    # Generate (autoregressive)
    # -------------------------------------------------
    def generate(self, start_tokens, max_len=20, temperature=1.0):
        """
        start_tokens: list[int] or tensor (T,)
        """

        self.eval()

        if not torch.is_tensor(start_tokens):
            start_tokens = torch.tensor(start_tokens, device=self.device)

        x = start_tokens.unsqueeze(0)  # (1, T)
        hidden = None

        outputs = start_tokens.tolist()

        for _ in range(max_len):
            logits = self(x)  # (1, T, vocab)
            logits = logits[:, -1, :] / temperature

            probs = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)  # sampling

            outputs.append(token.item())

            x = torch.cat([x, token], dim=1)

        return outputs

    # -------------------------------------------------
    # Generate with teacher forcing (for debugging)
    # -------------------------------------------------
    def generate_step(self, x, hidden=None):
        """
        One-step forward for custom loops
        """
        if self.embeddings is not None and x.dim() == 1:
            x = self.embeddings(x).unsqueeze(1)

        for rnn in self.rnn_layers:
            x, hidden = rnn(x, hidden)

        for fc in self.fc_layers:
            x = fc(x)

        return x, hidden