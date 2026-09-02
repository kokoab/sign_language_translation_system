"""Gloss-conditioned whole-utterance v17 landmark generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .schema_v17 import NUM_NODES


@dataclass(frozen=True)
class FullTrajectoryV17Config:
    vocabulary_size: int
    source_families: int
    frames: int = 128
    maximum_tokens: int = 42
    model_dim: int = 128
    heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 3
    feedforward_dim: int = 384
    dropout: float = 0.1
    motion_latent_dim: int = 32
    trajectory_encoder_layers: int = 2

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FullTrajectoryGeneratorV17(nn.Module):
    def __init__(self, config: FullTrajectoryV17Config):
        super().__init__()
        self.config = config
        d = config.model_dim
        self.token = nn.Embedding(config.vocabulary_size, d, padding_idx=0)
        self.token_position = nn.Embedding(config.maximum_tokens, d)
        self.source = nn.Embedding(config.source_families, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d, config.heads, config.feedforward_dim, config.dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.encoder_layers)
        self.frame_query = nn.Parameter(torch.randn(config.frames, d) * 0.02)
        self.trajectory_input = nn.Linear(NUM_NODES * 5, d)
        self.trajectory_position = nn.Embedding(config.frames, d)
        trajectory_layer = nn.TransformerEncoderLayer(
            d, config.heads, config.feedforward_dim, config.dropout,
            batch_first=True, norm_first=True,
        )
        self.trajectory_encoder = nn.TransformerEncoder(
            trajectory_layer, config.trajectory_encoder_layers
        )
        self.motion_mean = nn.Linear(d, config.motion_latent_dim)
        self.motion_log_variance = nn.Linear(d, config.motion_latent_dim)
        self.motion_to_query = nn.Linear(config.motion_latent_dim, d)
        decoder_layer = nn.TransformerDecoderLayer(
            d, config.heads, config.feedforward_dim, config.dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.output_norm = nn.LayerNorm(d)
        self.xyz = nn.Linear(d, NUM_NODES * 3)
        self.presence = nn.Linear(d, NUM_NODES)
        self.confidence = nn.Linear(d, NUM_NODES)
        self.duration = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(
        self, tokens: torch.Tensor, token_valid: torch.Tensor,
        source_ids: torch.Tensor, *,
        target_features: torch.Tensor | None = None,
        motion_latent: torch.Tensor | None = None,
        sample_latent: bool = False,
    ) -> dict[str, torch.Tensor]:
        if tokens.shape != token_valid.shape or tokens.ndim != 2:
            raise ValueError("tokens and validity must align as [batch,tokens]")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        source = self.source(source_ids)[:, None]
        memory = self.token(tokens) + self.token_position(positions)[None] + source
        memory = self.encoder(memory, src_key_padding_mask=~token_valid)
        denominator = token_valid.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (memory * token_valid[..., None]).sum(dim=1) / denominator
        motion_mean = motion_log_variance = None
        if target_features is not None:
            if target_features.shape[1:] != (
                self.config.frames, NUM_NODES, 5
            ):
                raise ValueError("target features must be [batch,frames,61,5]")
            trajectory = self.trajectory_input(target_features.flatten(2))
            trajectory = trajectory + self.trajectory_position(
                torch.arange(self.config.frames, device=tokens.device)
            )[None]
            trajectory = self.trajectory_encoder(trajectory)
            trajectory = trajectory.mean(dim=1)
            motion_mean = self.motion_mean(trajectory)
            motion_log_variance = self.motion_log_variance(trajectory).clamp(-8, 8)
            motion_latent = motion_mean
            if sample_latent:
                motion_latent = motion_mean + torch.randn_like(motion_mean) * torch.exp(
                    0.5 * motion_log_variance
                )
        elif motion_latent is None:
            motion_latent = torch.zeros(
                (len(tokens), self.config.motion_latent_dim),
                dtype=memory.dtype, device=memory.device,
            )
        if motion_latent.shape != (len(tokens), self.config.motion_latent_dim):
            raise ValueError("motion latent has the wrong shape")
        query = (
            self.frame_query[None].expand(len(tokens), -1, -1)
            + source + self.motion_to_query(motion_latent)[:, None]
        )
        decoded = self.output_norm(self.decoder(
            query, memory, memory_key_padding_mask=~token_valid
        ))
        output = {
            "xyz": self.xyz(decoded).reshape(
                len(tokens), self.config.frames, NUM_NODES, 3
            ),
            "presence_logits": self.presence(decoded),
            "confidence_logits": self.confidence(decoded),
            "log_duration": self.duration(pooled).squeeze(1),
        }
        if motion_mean is not None:
            output["motion_mean"] = motion_mean
            output["motion_log_variance"] = motion_log_variance
        return output


def observation_from_prediction(
    prediction: dict[str, torch.Tensor], threshold: float = 0.5,
) -> torch.Tensor:
    xyz = prediction["xyz"]
    presence = torch.sigmoid(prediction["presence_logits"]) >= threshold
    confidence = torch.sigmoid(prediction["confidence_logits"])
    output = torch.zeros(xyz.shape[:-1] + (5,), dtype=xyz.dtype, device=xyz.device)
    output[..., :3] = xyz * presence[..., None]
    output[..., 3] = presence
    output[..., 4] = confidence * presence
    return output
