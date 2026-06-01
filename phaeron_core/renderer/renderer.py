from __future__ import annotations
from torch import Tensor
import torch.nn as nn

from typing import Optional

from conditioning import Conditioning, ConditioningSpec

class FlowRenderer(nn.Module):
  def __init__(self, cond_spec: ConditioningSpec, latent_channels: int, hidden: int = 768, depth: int = 12)-> None:
    super().__init__()

    self.cond_spec = cond_spec
    