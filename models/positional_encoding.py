"""
PositionalEncoding class has been taken from PyTorch tutorials.
<Source>: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, dropout, length):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(length, model_dimension)  # (length, model_dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(
            1
        )  # (length, 1)
        div_term = torch.exp(
            torch.arange(0, model_dimension, 2).float()
            * (-math.log(10000.0) / model_dimension)
        )  # ([model_dim//2])
        pe[:, 0::2] = torch.sin(position * div_term)  # (length, model_dim//2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (length, model_dim//2)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (length, 1, model_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # x: (length, B, dim)
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)