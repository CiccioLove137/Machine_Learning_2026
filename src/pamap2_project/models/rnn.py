from __future__ import annotations

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(
        self,
        rnn_type: str,          # "gru" | "lstm"
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()

        rnn_type = rnn_type.lower()
        if rnn_type not in {"gru", "lstm"}:
            raise ValueError(f"rnn_type deve essere 'gru' o 'lstm', trovato: {rnn_type}")

        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        return logits: (B, C)
        """
        out, h = self.rnn(x)

        # prendo l'output dell'ultimo timestep
        last_out = out[:, -1, :]   # (B, hidden*(1|2))
        logits = self.classifier(last_out)
        return logits
