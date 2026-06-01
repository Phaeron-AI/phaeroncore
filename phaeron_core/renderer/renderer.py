from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor

from ..conditioning import Conditioning, ConditioningSpec
from .embeddings import TimeStepEmbedding
from .patchify import PatchEmbed, Unpatchify
from .adaln import AdaLNModulation, modulate
from .block import DiTBlock


class FlowRenderer(nn.Module):
  def __init__(self, cond_spec: ConditioningSpec, latent_channels: int, hidden: int = 768, depth: int = 12, num_heads: int = 12, patch_size: int = 2, max_tokens: int = 1024) -> None:
    super().__init__()

    self.cond_spec = cond_spec
    self.latent_channels = latent_channels

    in_ch = latent_channels + cond_spec.channels
    self.patch_embed = PatchEmbed(in_ch, hidden, patch_size)
    self.t_embed = TimeStepEmbedding(hidden, cond_spec.vector_dim)
    self.blocks = nn.ModuleList(
        [DiTBlock(hidden, num_heads) for _ in range(depth)]
    )
    self.final_mod = AdaLNModulation(hidden)
    self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
    self.unpatchify = Unpatchify(hidden, latent_channels, patch_size)

    nn.init.zeros_(self.unpatchify.proj.weight)
    nn.init.zeros_(self.unpatchify.proj.bias)

    self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, hidden))
    nn.init.normal_(self.pos_embed, std=0.02)

  def velocity(self, z_t: Tensor, t: Tensor, cond: Conditioning) -> Tensor:
    assert cond.matches(self.cond_spec), "conditioning does not match spec"
    B, C, h, w = z_t.shape

    x = torch.cat([z_t, cond.spatial], dim=1) if cond.spatial is not None else z_t

    tokens = self.patch_embed(x)
    num_tokens = tokens.shape[1]

    if num_tokens > self.pos_embed.shape[1]:
      raise ValueError(
        f"Spatial resolution {(h, w)} yields {num_tokens} tokens, "
        f"exceeding pos_embed limit of {self.pos_embed.shape[1]}"
      )

    tokens = tokens + self.pos_embed[:, :num_tokens, :]

    emb = self.t_embed(t, cond.vector)

    for blk in self.blocks:
      tokens = blk(tokens, emb)

    shift, scale, _g, _sm, _scm, _gm = self.final_mod(emb)
    tokens = modulate(self.final_norm(tokens), shift, scale)

    v_pred = self.unpatchify(tokens, h, w)
    return v_pred

  def sample(self, z_init: Tensor, cond: Conditioning, n_steps: int = 1) -> Tensor:
    z = z_init.clone()
    dt = 1.0 / n_steps

    with torch.no_grad():
      for i in range(n_steps):
        t_scalar = i * dt

        t = torch.full((z.shape[0],), t_scalar, device=z.device, dtype=torch.float32)

        v = self.velocity(z, t, cond)
        z = z + dt * v

    return z