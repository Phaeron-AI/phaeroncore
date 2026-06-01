from __future__ import annotations
from typing import Optional

import math
import torch
import torch.nn as nn
from torch import Tensor

class TimeStepEmbedding(nn.Module):
  def __init__(self, hidden: int, vector_dim: Optional[int] = None, freq_dim: int = 256)-> None:
    super().__init__()

    if freq_dim % 2 != 0:
      raise ValueError(f"freq_dim must be even, got {freq_dim}")

    self.freq_dim = freq_dim
    self.mlp = nn.Sequential(
      nn.Linear(freq_dim, hidden),
      nn.SiLU(),
      nn.Linear(hidden, hidden)
    )

    self.vector_proj = nn.Linear(vector_dim, hidden) if vector_dim is not None else None

  def _sinusoidal(self, t: Tensor):
    half_dim = self.freq_dim // 2

    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)

    args = t[:, None] * emb[None, :]
    features = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return features

  def forward(self, t: Tensor, vector: Optional[Tensor]):
    emb = self.mlp(self._sinusoidal(t))

    if self.vector_proj is not None:
      assert vector is not None, "spec has vector_dim but no vector given"
      emb = emb + self.vector_proj(vector)
    else:
      assert vector is None, "vector given but spec declares no vector_dim"
    
    return emb

