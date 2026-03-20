from .BaseModel import BaseModel
import torch.nn as nn

class RNN(BaseModel):
    def __init__(self, input_size, rnn_type="lstm", **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.rnn_types = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}
        self.rnn_class = self.rnn_types[rnn_type]

    def add_layer(self, hidden_size, num_layers=1, dropout=0.0, bidirectional=False):
        rnn = self.rnn_class(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.layers.append(rnn)
        self.input_size = hidden_size * (2 if bidirectional else 1)
        return self

    def forward(self, x):
        # RNNs need a custom forward because they return (output, state)
        for layer in self.model:
            if isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
                x, _ = layer(x)
            else:
                x = layer(x)
        return x