from __future__ import annotations
from dataclasses import dataclass
from torch import Tensor

from typing import Optional

@dataclass(frozen=True)
class ConditioningSpec:
  channels: int
  vector_dim: Optional[int] = None

@dataclass
class Conditioning:
  spatial: Optional[Tensor] = None
  vector: Optional[Tensor] = None

  def __post_init__(self)-> None:
    if self.spatial is not None and self.spatial.ndim != 4:
      raise ValueError(f"spatial must be [B,C,h,w], got ndim={self.spatial.ndim}")
    if self.vector is not None and self.vector.ndim != 2:
      raise ValueError(f"spatial must be [B,D], got ndim={self.vector.ndim}")
  
  def matches(self, spec: ConditioningSpec)-> bool:
    if self.spatial is not None and self.spatial.shape[1] != spec.channels:
      return False
    if spec.vector_dim is None:
      return self.vector is None
    return self.vector is not None and self.vector.shape[1] == spec.vector_dim