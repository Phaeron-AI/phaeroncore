from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

class PatchEmbed(nn.Module):
  def __init__(self, in_channels: int, hidden: int, patch_size: int = 2)-> None:
    super().__init__()
    self.patch_size = patch_size

    self.proj = nn.Conv2d(in_channels, hidden, kernel_size=patch_size, stride=patch_size)
  
  def forward(self, x: Tensor)-> Tensor:
    x = self.proj(x)

    x = x.flatten(2).transpose(1, 2)

    return x

class Unpatchify(nn.Module):
  def __init__(self, hidden: int, out_channels: int, patch_size: int = 2)-> None:
    super().__init__()
    self.patch_size = patch_size
    self.out_channels = out_channels
    self.proj = nn.Linear(hidden, out_channels * patch_size * patch_size)

  def forward(self, x: Tensor, h: int, w: int)-> Tensor:
    B, N, _ = x.shape
    p = self.patch_size

    h_patch, w_patch = h // p, w // p
    if N != h_patch * w_patch:
      raise ValueError(f"Token count {N} does not match expected grid {(h_patch, w_patch)} for target {(h, w)}")
    
    x = self.proj(x)
    x = x.view(B, h_patch, w_patch, p, p, self.out_channels)
    
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, self.out_channels, h, w)
    
    return x