"""Gloss-conditioned autoregressive prior over temporal motion codes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TemporalCodePriorV17Config:
    vocabulary_size: int
    source_families: int
    codebook_size: int = 128
    code_steps: int = 64
    maximum_tokens: int = 42
    model_dim: int = 128
    heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 3
    feedforward_dim: int = 384
    dropout: float = 0.1
    history_dropout: float = 0.2
    text_anchor_loss_weight: float = 0.0
    text_guidance_scale: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TemporalCodePriorV17(nn.Module):
    def __init__(self, config: TemporalCodePriorV17Config):
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
        self.code = nn.Embedding(config.codebook_size + 1, d)
        self.code_position = nn.Embedding(config.code_steps, d)
        decoder_layer = nn.TransformerDecoderLayer(
            d, config.heads, config.feedforward_dim, config.dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.norm = nn.LayerNorm(d)
        self.code_logits = nn.Linear(d, config.codebook_size)
        self.side_logits = nn.Linear(d, 2)
        if config.text_anchor_loss_weight > 0:
            self.text_code_anchor = nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                nn.Linear(d, config.codebook_size),
            )
            self.text_side_anchor = nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2),
            )
        self.duration = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def encode_text(self, tokens, token_valid, source_ids):
        if tokens.shape != token_valid.shape or tokens.ndim != 2:
            raise ValueError("tokens and validity must align as [batch,tokens]")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        source = self.source(source_ids)[:, None]
        memory = self.token(tokens) + self.token_position(positions)[None] + source
        memory = self.encoder(memory, src_key_padding_mask=~token_valid)
        pooled = (memory * token_valid[..., None]).sum(dim=1) / token_valid.sum(
            dim=1, keepdim=True
        ).clamp(min=1)
        return memory, pooled, source

    def aligned_gloss_timeline(self, memory, token_valid):
        """Monotonically align code positions to BOS/EOS-delimited gloss embeddings."""
        lengths = (token_valid.sum(dim=1) - 2).clamp(min=1)
        time = torch.linspace(
            0, 1, self.config.code_steps, device=memory.device
        )[None]
        position = time * (lengths[:, None] - 1)
        lower = position.floor().long() + 1
        upper = torch.minimum(lower + 1, lengths[:, None])
        fraction = (position - position.floor())[..., None]
        gather = lambda index: memory.gather(
            1, index[..., None].expand(-1, -1, memory.shape[-1])
        )
        return gather(lower) * (1 - fraction) + gather(upper) * fraction

    def decode_prefix(self, input_codes, memory, token_valid, source, text_timeline):
        if input_codes.ndim != 2 or input_codes.shape[1] > self.config.code_steps:
            raise ValueError("input codes must be [batch,steps]")
        positions = torch.arange(input_codes.shape[1], device=input_codes.device)
        target = (
            self.code(input_codes) + self.code_position(positions)[None]
            + source + text_timeline[:, :input_codes.shape[1]]
        )
        causal = nn.Transformer.generate_square_subsequent_mask(
            input_codes.shape[1], device=input_codes.device
        )
        state = self.norm(self.decoder(
            target, memory, tgt_mask=causal,
            memory_key_padding_mask=~token_valid,
        ))
        return self.code_logits(state), self.side_logits(state)

    def forward(self, tokens, token_valid, source_ids, target_codes):
        if target_codes.shape[1] != self.config.code_steps:
            raise ValueError("target code length does not match the prior")
        memory, pooled, source = self.encode_text(tokens, token_valid, source_ids)
        text_timeline = self.aligned_gloss_timeline(memory, token_valid)
        bos = torch.full(
            (len(target_codes), 1), self.config.codebook_size,
            dtype=torch.long, device=target_codes.device,
        )
        input_codes = torch.cat((bos, target_codes[:, :-1]), dim=1)
        if self.training and self.config.history_dropout > 0:
            drop = torch.rand_like(input_codes, dtype=torch.float) < self.config.history_dropout
            drop[:, 0] = False
            input_codes = torch.where(drop, bos.expand_as(input_codes), input_codes)
        code_logits, side_logits = self.decode_prefix(
            input_codes, memory, token_valid, source, text_timeline,
        )
        output = {
            "code_logits": code_logits,
            "side_logits": side_logits,
            "log_duration": self.duration(pooled).squeeze(1),
        }
        if hasattr(self, "text_code_anchor"):
            output["text_code_logits"] = self.text_code_anchor(text_timeline)
            output["text_side_logits"] = self.text_side_anchor(text_timeline)
            output["text_anchor_loss_weight"] = self.config.text_anchor_loss_weight
        return output

    @torch.inference_mode()
    def generate(
        self, tokens, token_valid, source_ids, *, temperature=1.0, top_k=16,
        sample=True,
    ):
        if temperature <= 0 or not 1 <= top_k <= self.config.codebook_size:
            raise ValueError("temperature and top-k must be positive and valid")
        memory, pooled, source = self.encode_text(tokens, token_valid, source_ids)
        text_timeline = self.aligned_gloss_timeline(memory, token_valid)
        prefix = torch.full(
            (len(tokens), 1), self.config.codebook_size,
            dtype=torch.long, device=tokens.device,
        )
        codes = []
        sides = []
        text_code_logits = (
            self.text_code_anchor(text_timeline)
            if hasattr(self, "text_code_anchor") else None
        )
        text_side_logits = (
            self.text_side_anchor(text_timeline)
            if hasattr(self, "text_side_anchor") else None
        )
        for step in range(self.config.code_steps):
            logits, side_logits = self.decode_prefix(
                prefix, memory, token_valid, source, text_timeline
            )
            logits = logits[:, -1]
            current_sides = side_logits[:, -1]
            if text_code_logits is not None:
                scale = self.config.text_guidance_scale
                logits = logits + scale * text_code_logits[:, step]
                current_sides = current_sides + scale * text_side_logits[:, step]
            logits = logits / temperature
            values, indices = logits.topk(top_k, dim=-1)
            if sample:
                choice = torch.multinomial(values.softmax(dim=-1), 1)
                next_code = indices.gather(1, choice)
            else:
                next_code = indices[:, :1]
            codes.append(next_code)
            sides.append(current_sides)
            prefix = torch.cat((prefix, next_code), dim=1)
        return {
            "codes": torch.cat(codes, dim=1),
            "side_logits": torch.stack(sides, dim=1),
            "log_duration": self.duration(pooled).squeeze(1),
        }


def apply_temporal_hand_mask(observation, side_logits, downsample_factor, threshold=0.5):
    """Apply predicted linguistic hand participation without creating either hand."""
    if observation.ndim != 4 or observation.shape[-2:] != (61, 5):
        raise ValueError("observation must be [batch,frames,61,5]")
    if side_logits.ndim != 3 or side_logits.shape[-1] != 2:
        raise ValueError("side logits must be [batch,steps,2]")
    sides = (torch.sigmoid(side_logits) >= threshold).repeat_interleave(
        downsample_factor, dim=1
    )
    if sides.shape[:2] != observation.shape[:2]:
        raise ValueError("side timeline does not match observation frames")
    output = observation.clone()
    output[:, :, :21] *= sides[..., 0, None, None]
    output[:, :, 21:42] *= sides[..., 1, None, None]
    return output
