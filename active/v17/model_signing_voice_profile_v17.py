"""Content-preserving statistical style profiles for a v17 signing voice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema_v17 import NUM_NODES


PROFILE_FRAMES = 32
PROFILE_DIM = NUM_NODES * 3 + PROFILE_FRAMES * 3


@dataclass(frozen=True)
class SigningVoiceProfileV17:
    node_offset: np.ndarray
    frame_curve: np.ndarray

    def validate(self) -> None:
        if self.node_offset.shape != (NUM_NODES, 3):
            raise ValueError("node offset must be [61, 3]")
        if self.frame_curve.shape != (PROFILE_FRAMES, 3):
            raise ValueError("frame curve must be [32, 3]")
        if not np.isfinite(self.node_offset).all() or not np.isfinite(self.frame_curve).all():
            raise ValueError("voice profile contains non-finite values")

    def vector(self) -> np.ndarray:
        self.validate()
        return np.concatenate((self.node_offset.reshape(-1), self.frame_curve.reshape(-1)))

    @classmethod
    def from_vector(cls, value: np.ndarray) -> "SigningVoiceProfileV17":
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (PROFILE_DIM,):
            raise ValueError("unexpected signing-voice profile vector")
        split = NUM_NODES * 3
        profile = cls(
            vector[:split].reshape(NUM_NODES, 3),
            vector[split:].reshape(PROFILE_FRAMES, 3),
        )
        profile.validate()
        return profile


def estimate_voice_profile(
    landmarks: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    prototypes: np.ndarray,
) -> SigningVoiceProfileV17:
    """Estimate content-independent signer habits from multiple distinct glosses."""
    rows = np.asarray(indices, dtype=np.int64)
    if not len(rows) or len(set(targets[rows].tolist())) < 2:
        raise ValueError("a voice profile requires at least two distinct glosses")
    real = landmarks[rows].astype(np.float32)
    base = prototypes[targets[rows]].astype(np.float32)
    valid = ((real[..., 3] > 0) & (base[..., 3] > 0))[..., None]
    node_count = valid.sum(axis=1).clip(min=1)
    clip_node_offset = ((real[..., :3] - base[..., :3]) * valid).sum(axis=1) / node_count
    node_offset = np.median(clip_node_offset, axis=0).astype(np.float32)
    residual = real[..., :3] - base[..., :3] - node_offset[None, None]
    frame_count = valid.sum(axis=2).clip(min=1)
    clip_frame_curve = (residual * valid).sum(axis=2) / frame_count
    frame_curve = np.median(clip_frame_curve, axis=0).astype(np.float32)
    frame_curve -= frame_curve.mean(axis=0, keepdims=True)
    return SigningVoiceProfileV17(node_offset, frame_curve)


def fit_profile_latent(
    profiles: list[SigningVoiceProfileV17], latent_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.stack([profile.vector() for profile in profiles]).astype(np.float32)
    if not 1 <= latent_dim < min(values.shape):
        raise ValueError("profile latent dimension must be below voices and features")
    mean = values.mean(axis=0)
    _, _, right = np.linalg.svd(values - mean, full_matrices=False)
    components = right[:latent_dim].astype(np.float32)
    latents = ((values - mean) @ components.T).astype(np.float32)
    return mean.astype(np.float32), components, latents


def encode_profile(
    profile: SigningVoiceProfileV17,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    return ((profile.vector() - mean) @ components.T).astype(np.float32)


def decode_profile(
    latent: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> SigningVoiceProfileV17:
    return SigningVoiceProfileV17.from_vector(
        mean + np.asarray(latent, dtype=np.float32) @ components
    )


def apply_voice_profile(
    prototype: np.ndarray,
    profile: SigningVoiceProfileV17,
    *,
    profile_strength: float = 1.0,
    curve_strength: float = 0.25,
) -> np.ndarray:
    profile.validate()
    value = np.asarray(prototype, dtype=np.float32)
    if value.shape != (PROFILE_FRAMES, NUM_NODES, 5):
        raise ValueError("unexpected content prototype")
    if not 0.0 <= profile_strength <= 1.0 or not 0.0 <= curve_strength <= 1.0:
        raise ValueError("profile and curve strengths must be in [0, 1]")
    output = value.copy()
    spatial = (
        value[..., :3]
        + profile_strength * profile.node_offset[None]
        + curve_strength * profile.frame_curve[:, None]
    )
    present = value[..., 3:4] > 0
    output[..., :3] = np.where(present, spatial, 0.0)
    return output
