from typing import List, Tuple, Optional
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Minimal causal TCN blocks (adapted pattern used in many TCN implementations)
# ------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, padding, dropout=0.0):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class SimpleTCN(nn.Module):
    """
    A compact TCN usable as Uncoupler / Recoupler.
    Input shapes:
      - If processing univariate series: (batch, seq_len, channels_in) or (batch, channels_in, seq_len).
    This TCN returns features in (batch, seq_len, out_channels) (transposed).
    """
    def __init__(self, in_channels:int, num_levels:int=4, hidden_channels:int=32, kernel_size:int=3, dropout:float=0.0):
        super().__init__()
        layers = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else hidden_channels
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_ch, hidden_channels, kernel_size, dilation, padding, dropout))
        self.network = nn.Sequential(*layers)
        # final conv to produce out channels per time step
        self.final = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)

    def forward(self, x):
        """
        强制要求输入为 (B, T, C)
        """
        if x.dim() != 3:
            raise ValueError(f"SimpleTCN expected input (B, T, C), got {x.shape}")

        B, T, C = x.shape
        x = x.transpose(1, 2)     # (B, C, T)

        y = self.network(x)       # (B, hidden, T)
        y = self.final(y)         # (B, hidden, T)
        return y.transpose(1, 2)  # (B, T, hidden)