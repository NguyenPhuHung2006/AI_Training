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

        self.encoder = None
        self.decoder = None

        self.enc_bidirectional = False
        self.bridge_h = None
        self.bridge_c = None

        self.embeddings = None
        self.output_size = None

        self.sos_token = None
        self.eos_token = None
        self.pad_token = None

    def add_embedding(self, vocab_size, embed_dim):
        pad_id = self.pad_token if self.pad_token is not None else 0
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
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

        self.input_size = hidden_size
        return self

    def add_fc(self, out_features, **kwargs):
        layer = nn.Linear(self.input_size, out_features)
        block = self._apply_utils(layer, **kwargs, norm_dim=out_features)

        self.fc_layers.append(block)
        self.input_size = out_features
        self.output_size = out_features
        return self

    def forward(self, x, target=None, hidden=None, teacher_forcing_ratio=1.0):
        x = x.to(self.device)
        lengths = (x != self.pad_token).sum(dim=1)
        if target is not None:
            target = target.to(self.device)

        if self.embeddings is not None and x.dim() == 2:
            x = self.embeddings(x.long())
            
        if self.mode == "seq2seq":
            if self.eos_token is None:
                raise ValueError("eos token is not defined")
            if self.sos_token is None:
                raise ValueError("sos token is not defined")
            
            enc_out, hidden = self.encoder(x)
            B = x.size(0)

            if self.enc_bidirectional:
                if isinstance(hidden, tuple):
                    h, c = hidden
                    h = h.view(self.num_layers, 2, B, self.hidden_size)
                    c = c.view(self.num_layers, 2, B, self.hidden_size)

                    h = torch.cat([h[:, 0], h[:, 1]], dim=-1)
                    c = torch.cat([c[:, 0], c[:, 1]], dim=-1)

                    h = self.bridge_h(h)
                    c = self.bridge_c(c) if self.bridge_c else c

                    hidden = (h, c)
                else:
                    h = hidden.view(self.num_layers, 2, B, self.hidden_size)
                    h = torch.cat([h[:, 0], h[:, 1]], dim=-1)
                    h = self.bridge_h(h)
                    hidden = h

            T = target.size(1) if target is not None else x.size(1)

            input_t = torch.full((B, 1), self.sos_token, device=self.device)

            if self.embeddings is not None:
                input_t = self.embeddings(input_t)
            else:
                input_t = input_t.float().unsqueeze(-1)

            outputs = []

            for t in range(T):
                out, hidden = self.decoder(input_t, hidden)

                logits = out
                for fc in self.fc_layers:
                    logits = fc(logits)

                outputs.append(logits)

                # ---- Teacher forcing ----
                if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                    next_input = target[:, t].unsqueeze(1)

                    if self.embeddings is not None:
                        next_input = self.embeddings(next_input)
                    else:
                        next_input = next_input.float().unsqueeze(-1)

                    input_t = next_input
                else:
                    next_token = torch.argmax(logits, dim=-1)

                    if self.embeddings is not None:
                        input_t = self.embeddings(next_token)
                    else:
                        input_t = next_token.float().unsqueeze(-1)

            x = torch.cat(outputs, dim=1)

        # =========================
        # STANDARD RNN
        # =========================
        else:
            if lengths is not None:
                x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

            if self.rnn:
                if hidden is not None:
                    x, hidden = self.rnn(x, hidden)
                else:
                    x, hidden = self.rnn(x)

            if lengths is not None:
                x, _ = pad_packed_sequence(x, batch_first=True)

        # ---- Mode handling ----
        if self.mode == "many_to_one":
            if lengths is None:
                x = x[:, -1]
            else:
                x = x[torch.arange(x.size(0)), lengths - 1]

        # ---- FC ----
        if self.mode != "seq2seq":
            for fc in self.fc_layers:
                x = fc(x)

        return x

    # ----------------------------
    # GENERATE
    # ----------------------------

    def generate(self, start_tokens, max_len=20, temperature=1.0):
        if self.eos_token is None:
            raise ValueError("eos token is not defined")
        if self.sos_token is None:
            raise ValueError("sos token is not defined")
        
        self.eval()

        if not torch.is_tensor(start_tokens):
            start_tokens = torch.tensor(start_tokens, device=self.device)

        x = start_tokens.unsqueeze(0).to(self.device)

        if self.embeddings is not None:
            x = self.embeddings(x)

        enc_out, hidden = self.encoder(x)

        # ---- Bidirectional fix ----
        if self.enc_bidirectional:
            B = x.size(0)

            if isinstance(hidden, tuple):
                h, c = hidden
                h = h.view(self.num_layers, 2, B, self.hidden_size)
                c = c.view(self.num_layers, 2, B, self.hidden_size)

                h = torch.cat([h[:, 0], h[:, 1]], dim=-1)
                c = torch.cat([c[:, 0], c[:, 1]], dim=-1)

                h = self.bridge_h(h)
                c = self.bridge_c(c) if self.bridge_c else c

                hidden = (h, c)
            else:
                h = hidden.view(self.num_layers, 2, B, self.hidden_size)
                h = torch.cat([h[:, 0], h[:, 1]], dim=-1)
                h = self.bridge_h(h)
                hidden = h

        input_t = torch.tensor([[self.sos_token]], device=self.device)

        if self.embeddings is not None:
            input_t = self.embeddings(input_t)
        else:
            input_t = input_t.float().unsqueeze(-1)

        outputs = []

        for _ in range(max_len):
            out, hidden = self.decoder(input_t, hidden)

            logits = out
            for fc in self.fc_layers:
                logits = fc(logits)

            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

            token = torch.multinomial(probs, 1)
            token_id = token.item()

            outputs.append(token_id)

            if token_id == self.eos_token:
                break

            if self.embeddings is not None:
                input_t = self.embeddings(token)
            else:
                input_t = token.float().unsqueeze(-1)

        return outputs