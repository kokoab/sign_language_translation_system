"""Conditional diffusion over residual v17 transition motion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from .schema_v17 import NUM_CHANNELS, NUM_NODES


@dataclass(frozen=True)
class TransitionResidualDiffusionV17Config:
    frames: int = 32
    nodes: int = NUM_NODES
    channels: int = NUM_CHANNELS
    spatial_channels: int = 3
    dim: int = 192
    depth: int = 4
    heads: int = 6
    dropout: float = 0.10
    timesteps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02

    @property
    def condition_dim(self) -> int:
        return self.nodes * self.channels

    @property
    def residual_dim(self) -> int:
        return self.nodes * self.spatial_channels

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor, dim: int
) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if embedding.shape[1] < dim:
        embedding = torch.nn.functional.pad(embedding, (0, dim - embedding.shape[1]))
    return embedding


class TransitionResidualDiffusionV17(nn.Module):
    """Denoise stochastic motion residuals around a boundary-safe mean trajectory."""

    def __init__(self, config: TransitionResidualDiffusionV17Config | None = None):
        super().__init__()
        self.config = config or TransitionResidualDiffusionV17Config()
        self.input_projection = nn.Linear(
            self.config.condition_dim + self.config.residual_dim + 1,
            self.config.dim,
        )
        self.style_projection = nn.Sequential(
            nn.Linear(self.config.condition_dim * 2, self.config.dim),
            nn.GELU(),
            nn.LayerNorm(self.config.dim),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(self.config.dim, self.config.dim * 2),
            nn.SiLU(),
            nn.Linear(self.config.dim * 2, self.config.dim),
        )
        self.position = nn.Parameter(
            torch.zeros(1, self.config.frames, self.config.dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.dim,
            nhead=self.config.heads,
            dim_feedforward=self.config.dim * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence = nn.TransformerEncoder(layer, self.config.depth)
        self.output_norm = nn.LayerNorm(self.config.dim)
        self.output_projection = nn.Linear(self.config.dim, self.config.residual_dim)
        nn.init.trunc_normal_(self.position, std=0.02)

        betas = torch.linspace(
            self.config.beta_start, self.config.beta_end, self.config.timesteps
        )
        alphas = 1.0 - betas
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def visible_style(
        self, flattened: torch.Tensor, transition_mask: torch.Tensor
    ) -> torch.Tensor:
        visible = (~transition_mask).to(flattened.dtype).unsqueeze(-1)
        count = visible.sum(dim=1).clamp_min(1.0)
        mean = (flattened * visible).sum(dim=1) / count
        variance = (
            (flattened - mean.unsqueeze(1)).square() * visible
        ).sum(dim=1) / count
        return self.style_projection(torch.cat((mean, variance.sqrt()), dim=-1))

    def forward(
        self,
        mean_features: torch.Tensor,
        transition_mask: torch.Tensor,
        noisy_residual: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        expected = (
            len(mean_features), self.config.frames, self.config.nodes,
            self.config.channels,
        )
        if tuple(mean_features.shape) != expected:
            raise ValueError(f"unexpected mean tensor {tuple(mean_features.shape)}")
        if tuple(noisy_residual.shape) != expected[:-1] + (self.config.spatial_channels,):
            raise ValueError("unexpected noisy residual tensor")
        if transition_mask.shape != expected[:2] or transition_mask.dtype != torch.bool:
            raise ValueError("transition mask must be boolean [batch, frames]")
        if timesteps.shape != (len(mean_features),):
            raise ValueError("timesteps must have one value per batch row")
        flattened = mean_features.flatten(2)
        style = self.visible_style(flattened, transition_mask)
        time = self.time_projection(
            sinusoidal_timestep_embedding(timesteps, self.config.dim)
        )
        tokens = self.input_projection(torch.cat((
            flattened,
            noisy_residual.flatten(2),
            transition_mask.to(mean_features.dtype).unsqueeze(-1),
        ), dim=-1))
        tokens = tokens + self.position + style[:, None] + time[:, None]
        tokens = self.sequence(tokens)
        predicted_noise = self.output_projection(
            self.output_norm(tokens)
        ).reshape(noisy_residual.shape)
        return torch.where(
            transition_mask[:, :, None, None],
            predicted_noise,
            torch.zeros_like(predicted_noise),
        )

    def q_sample(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps].to(clean.dtype)[:, None, None, None]
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    @torch.inference_mode()
    def sample_normalized_residual(
        self,
        mean_features: torch.Tensor,
        transition_mask: torch.Tensor,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        sampling_steps: int | None = None,
    ) -> torch.Tensor:
        shape = mean_features.shape[:-1] + (self.config.spatial_channels,)
        residual = torch.randn(
            shape,
            device=mean_features.device,
            dtype=mean_features.dtype,
            generator=generator,
        ) * temperature
        residual = residual * transition_mask[:, :, None, None]
        step_count = min(sampling_steps or self.config.timesteps, self.config.timesteps)
        schedule = torch.linspace(
            self.config.timesteps - 1, 0, step_count
        ).round().to(torch.long).unique_consecutive().tolist()
        for position, timestep in enumerate(schedule):
            times = torch.full(
                (len(mean_features),), timestep,
                device=mean_features.device, dtype=torch.long,
            )
            predicted_noise = self(
                mean_features, transition_mask, residual, times
            )
            alpha_bar = self.alpha_bars[timestep].to(residual.dtype)
            clean = (
                residual - (1.0 - alpha_bar).sqrt() * predicted_noise
            ) / alpha_bar.sqrt()
            clean = clean.clamp(-6.0, 6.0)
            if position + 1 < len(schedule):
                previous = self.alpha_bars[schedule[position + 1]].to(residual.dtype)
                residual = (
                    previous.sqrt() * clean
                    + (1.0 - previous).sqrt() * predicted_noise
                )
            else:
                residual = clean
            residual = residual * transition_mask[:, :, None, None]
        return residual

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
