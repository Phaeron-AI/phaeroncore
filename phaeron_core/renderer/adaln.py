from __future__ import annotations
import torch.nn as nn
from torch import Tensor

class AdaLNModulation(nn.Module):
  def __init__(self, hidden: int)-> None:
    super().__init__()

    self.linear = nn.Sequential(
      nn.SiLU(),
      nn.Linear(hidden, 6 * hidden)
    )

    nn.init.zeros_(self.linear.weight)  # type: ignore
    nn.init.zeros_(self.linear.bias)  # type: ignore
  
  def forward(self, emb: Tensor)-> tuple[Tensor, ...]:
    out = self.linear(emb)
    return out.chunk(6, dim=-1)
  
def modulate(x: Tensor, shift: Tensor, scale: Tensor)-> Tensor:
  return x * (1.0 + scale[:, None, :]) + shift[:, None, :]