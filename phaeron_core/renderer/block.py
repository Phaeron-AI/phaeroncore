from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .adaln import AdaLNModulation, modulate

class DiTBlock(nn.Module):
  def __init__(self, hidden: int, num_heads: int = 12, mlp_ratio: float = 4.0)-> None:
    super().__init__()

    self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
    self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
    self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
    self.mlp = nn.Sequential(
      nn.Linear(hidden, int(hidden * mlp_ratio)),
      nn.GELU(approximate="tanh"),
      nn.Linear(int(hidden * mlp_ratio), hidden),
    )
    self.modulation = AdaLNModulation(hidden)
  
  def forward(self, x: Tensor, emb: Tensor)-> Tensor:
    s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = self.modulation(emb)
    h = modulate(self.norm1(x), s_msa, sc_msa)

    x = x + g_msa[:, None, :] * self.attn(h, h, h, need_weights=False)[0]

    h = modulate(self.norm2(x), s_mlp, sc_mlp)

    x = x + g_mlp[:, None] * self.mlp(h)
    return x