from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from diffusers import AutoencoderKL # type: ignore


class FrozenVAE(nn.Module):
  latent_channels: int
  scale_factor: float

  def __init__(self, checkpoint: str = "stabilityai/sd-vae-ft-mse", scale_factor: float = 0.18215) -> None:
    super().__init__()
    self.scale_factor = scale_factor

    self.vae = AutoencoderKL.from_pretrained(checkpoint)

    self.vae.requires_grad_(False)
    self.vae.eval()

    self.latent_channels = self.vae.config.latent_channels  # type: ignore

  @torch.no_grad()
  def encode(self, image: Tensor) -> Tensor:
    device = next(self.vae.parameters()).device
    dtype = next(self.vae.parameters()).dtype
    x = image.to(device=device, dtype=dtype)

    posterior = self.vae.encode(x).latent_dist  # type: ignore
    latent = posterior.sample()

    return latent * self.scale_factor

  @torch.no_grad()
  def decode(self, latent: Tensor) -> Tensor:
    """[B,C,h,w] -> [B,3,H,W] image."""
    device = next(self.vae.parameters()).device
    dtype = next(self.vae.parameters()).dtype
    z = latent.to(device=device, dtype=dtype)

    z = z / self.scale_factor
    
    out = self.vae.decode(z).sample  # type: ignore
    return out