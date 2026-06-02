from __future__ import annotations
from typing import Optional, Protocol
from contextlib import contextmanager
import torch
from torch.amp.autocast_mode import autocast
from torch import Tensor
import torch.nn as nn

from .renderer.renderer import FlowRenderer


class ExtraLossFn(Protocol):
  def __call__(self, pred: Tensor, batch: dict) -> Tensor: ...


@contextmanager
def _autocast_ctx(device_type: str = "cuda"):
  with autocast(device_type=device_type, dtype=torch.bfloat16):
    yield


def flow_matching_loss(renderer: FlowRenderer, batch: dict) -> tuple[Tensor, Tensor]:
  x_0 = batch['x_0']            # [B, C, H, W] - noise / source
  x_1 = batch['x_1']            # [B, C, H, W] - data / target
  cond = batch['cond']
  B = x_0.shape[0]

  # fp32 timestep: precision matters in the sinusoidal embedding (matches sample()).
  t = torch.rand((B,), device=x_0.device, dtype=torch.float32)
  t_col = t[:, None, None, None]

  z_t = (1.0 - t_col) * x_0 + t_col * x_1
  target_velocity = x_1 - x_0

  # instance method, not the class
  pred_velocity = renderer.velocity(z_t, t, cond)

  base_loss = torch.mean((pred_velocity - target_velocity) ** 2)
  return base_loss, pred_velocity


def train_step(renderer: FlowRenderer, batch: dict, optimizer: torch.optim.Optimizer, *,
               extra_loss_fn: ExtraLossFn | None = None,
               extra_weight: float = 1.0,
               device_type: str = "cuda") -> dict[str, float]:
  """One optimization step. Returns scalar metrics for logging."""
  optimizer.zero_grad(set_to_none=True)

  metrics: dict[str, float] = {}

  with _autocast_ctx(device_type):
    base_loss, pred_velocity = flow_matching_loss(renderer, batch)
    metrics["train/base_loss"] = base_loss.item()

    total_loss = base_loss
    if extra_loss_fn is not None:
      extra_loss = extra_loss_fn(pred_velocity, batch)
      metrics["train/extra_loss"] = extra_loss.item()
      total_loss = total_loss + (extra_weight * extra_loss)

  # Loss reductions run in fp32 under autocast; ensure fp32 before backward.
  total_loss = total_loss.float()
  metrics["train/total_loss"] = total_loss.item()

  total_loss.backward()

  grad_norm = torch.nn.utils.clip_grad_norm_(renderer.parameters(), max_norm=1.0)
  metrics["train/grad_norm"] = float(grad_norm)

  for param_group in optimizer.param_groups:
    metrics["train/lr"] = param_group["lr"]
    break

  optimizer.step()
  return metrics