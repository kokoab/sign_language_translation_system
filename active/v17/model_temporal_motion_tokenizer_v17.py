"""Temporal discrete tokenizer for genuine v17 landmark trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .schema_v17 import NUM_NODES


@dataclass(frozen=True)
class TemporalMotionTokenizerV17Config:
    frames: int = 128
    hidden_dim: int = 192
    latent_dim: int = 64
    codebook_size: int = 128
    commitment_weight: float = 0.25
    codebook_decay: float = 0.99
    downsample_factor: int = 4

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TemporalMotionTokenizerV17(nn.Module):
    def __init__(self, config: TemporalMotionTokenizerV17Config):
        super().__init__()
        if config.downsample_factor not in (2, 4) or config.frames % config.downsample_factor:
            raise ValueError("downsample factor must be two or four and divide frames")
        self.config = config
        width = NUM_NODES * 5
        encoder = [
            nn.Conv1d(width, config.hidden_dim, 5, padding=2),
            nn.GELU(),
        ]
        if config.downsample_factor == 4:
            encoder += [
                nn.Conv1d(config.hidden_dim, config.hidden_dim, 4, stride=2, padding=1),
                nn.GELU(),
            ]
        encoder.append(nn.Conv1d(
            config.hidden_dim, config.latent_dim, 4, stride=2, padding=1
        ))
        self.encoder = nn.Sequential(*encoder)
        self.encoder_norm = nn.LayerNorm(config.latent_dim)
        self.codebook = nn.Embedding(config.codebook_size, config.latent_dim)
        self.codebook.weight.requires_grad_(False)
        self.register_buffer("codebook_initialized", torch.tensor(False))
        self.register_buffer("ema_count", torch.ones(config.codebook_size))
        self.register_buffer("ema_sum", torch.zeros(config.codebook_size, config.latent_dim))
        decoder = [
            nn.ConvTranspose1d(
                config.latent_dim, config.hidden_dim, 4, stride=2, padding=1
            ),
            nn.GELU(),
        ]
        if config.downsample_factor == 4:
            decoder += [
                nn.ConvTranspose1d(
                    config.hidden_dim, config.hidden_dim, 4, stride=2, padding=1
                ),
                nn.GELU(),
            ]
        self.decoder = nn.Sequential(*decoder)
        self.xyz = nn.Conv1d(config.hidden_dim, NUM_NODES * 3, 3, padding=1)
        self.presence = nn.Conv1d(config.hidden_dim, NUM_NODES, 3, padding=1)
        self.confidence = nn.Conv1d(config.hidden_dim, NUM_NODES, 3, padding=1)
        nn.init.uniform_(self.codebook.weight, -1 / config.codebook_size, 1 / config.codebook_size)

    @property
    def code_steps(self) -> int:
        return self.config.frames // self.config.downsample_factor

    def encode_latent(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1:] != (self.config.frames, NUM_NODES, 5):
            raise ValueError("features must be [batch,frames,61,5]")
        latent = self.encoder(features.flatten(2).transpose(1, 2)).transpose(1, 2)
        return self.encoder_norm(latent)

    def quantize(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and not bool(self.codebook_initialized):
            flat = latent.detach().flatten(0, 1)
            if len(flat) < self.config.codebook_size:
                raise ValueError("first training batch is too small to initialize codebook")
            selected = flat[torch.randperm(len(flat), device=flat.device)[:self.config.codebook_size]]
            with torch.no_grad():
                self.codebook.weight.copy_(selected)
                self.ema_sum.copy_(selected)
                self.codebook_initialized.fill_(True)
        distance = (
            latent.square().sum(dim=-1, keepdim=True)
            - 2 * latent @ self.codebook.weight.T
            + self.codebook.weight.square().sum(dim=-1)
        )
        codes = distance.argmin(dim=-1)
        quantized = self.codebook(codes)
        if self.training:
            with torch.no_grad():
                flat_codes = codes.flatten()
                flat_latent = latent.detach().flatten(0, 1)
                counts = torch.bincount(
                    flat_codes, minlength=self.config.codebook_size
                ).to(latent.dtype)
                sums = torch.zeros_like(self.ema_sum)
                sums.index_add_(0, flat_codes, flat_latent)
                decay = self.config.codebook_decay
                self.ema_count.mul_(decay).add_(counts, alpha=1 - decay)
                self.ema_sum.mul_(decay).add_(sums, alpha=1 - decay)
                self.codebook.weight.copy_(
                    self.ema_sum / self.ema_count[:, None].clamp(min=1e-5)
                )
        return codes, quantized

    def encode_codes(self, features: torch.Tensor) -> torch.Tensor:
        return self.quantize(self.encode_latent(features))[0]

    def decode_quantized(self, quantized: torch.Tensor) -> dict[str, torch.Tensor]:
        decoded = self.decoder(quantized.transpose(1, 2))
        batch = len(decoded)
        xyz = self.xyz(decoded).transpose(1, 2).reshape(
            batch, self.config.frames, NUM_NODES, 3
        )
        return {
            "xyz": xyz,
            "presence_logits": self.presence(decoded).transpose(1, 2).contiguous(),
            "confidence_logits": self.confidence(decoded).transpose(1, 2).contiguous(),
            "log_duration": xyz.new_zeros(batch),
        }

    def decode_codes(self, codes: torch.Tensor) -> dict[str, torch.Tensor]:
        if codes.ndim != 2 or codes.shape[1] != self.code_steps:
            raise ValueError("codes must be [batch,code_steps]")
        return self.decode_quantized(self.codebook(codes))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode_latent(features)
        codes, quantized = self.quantize(latent)
        straight_through = latent + (quantized - latent).detach()
        output = self.decode_quantized(straight_through)
        output["codes"] = codes
        output["quantization_loss"] = self.config.commitment_weight * F.mse_loss(
            latent, quantized.detach()
        )
        counts = torch.bincount(codes.flatten(), minlength=self.config.codebook_size).float()
        probabilities = counts / counts.sum().clamp(min=1)
        output["codebook_perplexity"] = torch.exp(
            -(probabilities * probabilities.clamp(min=1e-12).log()).sum()
        )
        return output
