"""Accuracy-first isolated-sign model for the v17 landmark contract.

The public model input remains the extractor's compact ``[B, 32, 61, 5]`` tensor.
Masked motion and hand-shape distances are derived inside the network so training,
evaluation, and the eventual Core ML export cannot disagree about preprocessing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .schema_v17 import BODY_START, FACE_START, NUM_CHANNELS, NUM_NODES


@dataclass(frozen=True)
class Stage1V17Config:
    num_classes: int = 100
    dim: int = 256
    depth: int = 4
    heads: int = 8
    conv_kernel: int = 15
    dropout: float = 0.12
    head_dropout: float = 0.25
    drop_path: float = 0.08
    use_pairwise: bool = True
    input_modality: str = "all"
    spatial_encoder: str = "flat"
    graph_node_dim: int = 64
    graph_layers: int = 2
    graph_heads: int = 4
    temporal_encoder: str = "global"
    part_depth: int = 1
    use_bone_features: bool = False
    use_hand_angle_features: bool = False
    use_keypoint_temporal_gate: bool = False
    use_articulated_pose_embedding: bool = False
    static_hand_token: str = "none"
    use_attention_score_mixing: bool = False
    canonicalize_camera_roll: bool = False
    use_part_auxiliary: bool = False
    phonology_head_sizes: tuple[tuple[str, int], ...] = ()

    def validate(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if self.dim < 32 or self.depth < 1:
            raise ValueError("dim must be >= 32 and depth must be positive")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if self.conv_kernel < 3 or self.conv_kernel % 2 == 0:
            raise ValueError("conv_kernel must be odd and at least 3")
        if self.input_modality not in ("all", "hands", "face", "mouth"):
            raise ValueError("input_modality must be all, hands, face, or mouth")
        if self.spatial_encoder not in ("flat", "graph_parts", "flat_graph_residual"):
            raise ValueError(
                "spatial_encoder must be flat, graph_parts, or flat_graph_residual"
            )
        if self.graph_node_dim < 16 or self.graph_layers < 1:
            raise ValueError("graph_node_dim must be >= 16 and graph_layers positive")
        if self.graph_node_dim % self.graph_heads:
            raise ValueError("graph_node_dim must be divisible by graph_heads")
        if self.temporal_encoder not in ("global", "partwise_global"):
            raise ValueError("temporal_encoder must be global or partwise_global")
        if self.part_depth < 1:
            raise ValueError("part_depth must be positive")
        if self.temporal_encoder == "partwise_global" and self.spatial_encoder != "flat":
            raise ValueError("partwise_global currently requires the flat spatial encoder")
        if self.temporal_encoder == "partwise_global" and self.dim % 4:
            raise ValueError("partwise_global requires dim divisible by four")
        if self.use_part_auxiliary and self.temporal_encoder != "partwise_global":
            raise ValueError("part auxiliary heads require partwise_global")
        if self.use_bone_features and self.spatial_encoder == "graph_parts":
            raise ValueError("bone features require a flat or partwise input path")
        if self.use_hand_angle_features and self.spatial_encoder == "graph_parts":
            raise ValueError("hand angle features require a flat or partwise input path")
        if self.use_keypoint_temporal_gate and self.temporal_encoder != "partwise_global":
            raise ValueError("keypoint temporal gating requires partwise_global")
        if self.use_articulated_pose_embedding and self.temporal_encoder != "partwise_global":
            raise ValueError("articulated pose embedding requires partwise_global")
        if self.static_hand_token not in ("none", "quality", "low_motion"):
            raise ValueError("static_hand_token must be none, quality, or low_motion")
        if self.static_hand_token != "none" and self.temporal_encoder != "partwise_global":
            raise ValueError("static hand token requires partwise_global")
        if self.use_attention_score_mixing and self.temporal_encoder != "partwise_global":
            raise ValueError("attention score mixing requires partwise_global")
        names = [name for name, _ in self.phonology_head_sizes]
        if len(names) != len(set(names)) or any(size < 2 for _, size in self.phonology_head_sizes):
            raise ValueError("phonology heads must have unique names and at least two classes")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def masked_temporal_features(features: torch.Tensor) -> torch.Tensor:
    """Append causal XYZ velocity/acceleration without crossing missing joints.

    A derivative exists only when both observations used by that difference exist.
    This prevents a landmark appearing/disappearing at zero from creating a large,
    physically meaningless motion spike.
    """
    if features.ndim != 4 or features.shape[-2:] != (NUM_NODES, NUM_CHANNELS):
        raise ValueError(
            f"expected [B, T, {NUM_NODES}, {NUM_CHANNELS}], got {tuple(features.shape)}"
        )
    xyz = features[..., :3]
    presence = features[..., 3:4].clamp(0.0, 1.0)
    confidence = features[..., 4:5].clamp(0.0, 1.0) * presence
    xyz = xyz * presence

    velocity = torch.zeros_like(xyz)
    velocity_valid = presence[:, 1:] * presence[:, :-1]
    velocity[:, 1:] = (xyz[:, 1:] - xyz[:, :-1]) * velocity_valid

    acceleration = torch.zeros_like(xyz)
    acceleration_valid = velocity_valid[:, 1:] * velocity_valid[:, :-1]
    acceleration[:, 2:] = (
        velocity[:, 2:] - velocity[:, 1:-1]
    ) * acceleration_valid
    return torch.cat(
        (xyz, presence, confidence, velocity, acceleration), dim=-1
    )


def canonicalize_camera_roll_v17(features: torch.Tensor) -> torch.Tensor:
    """Align each clip's shoulder/eye axis while preserving the v17 contract.

    Camera roll is constant at clip scale. A confidence-weighted mean of normalized
    left-to-right shoulder directions provides the primary axis; the pupil axis is a
    fallback when shoulders are unavailable. Clips with neither reference pass
    through unchanged. The operation is differentiable and Core ML traceable.
    """
    if features.ndim != 4 or features.shape[-2:] != (NUM_NODES, NUM_CHANNELS):
        raise ValueError(
            f"expected [B, T, {NUM_NODES}, {NUM_CHANNELS}], got {tuple(features.shape)}"
        )

    def reference(first: int, second: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Anatomical left appears on image right for upright, unmirrored input.
        # Use right-to-left image direction so the canonical upright axis is +X.
        vector = features[..., first, :2] - features[..., second, :2]
        valid = (
            (features[..., first, 3] > 0.5)
            & (features[..., second, 3] > 0.5)
        )
        length = torch.linalg.vector_norm(vector, dim=-1)
        valid &= length > 1e-6
        unit = vector / length.clamp_min(1e-6).unsqueeze(-1)
        confidence = torch.minimum(
            features[..., first, 4], features[..., second, 4]
        ).clamp(0.0, 1.0)
        weight = valid.to(features.dtype) * confidence
        direction = (unit * weight.unsqueeze(-1)).sum(dim=1)
        weight_sum = weight.sum(dim=1)
        return direction, weight_sum

    shoulder, shoulder_weight = reference(BODY_START, BODY_START + 1)
    eyes, eye_weight = reference(FACE_START, FACE_START + 1)
    use_shoulders = shoulder_weight > 0
    direction = torch.where(use_shoulders.unsqueeze(-1), shoulder, eyes)
    available = use_shoulders | (eye_weight > 0)
    norm = torch.linalg.vector_norm(direction, dim=-1)
    available &= norm > 1e-6
    cosine = torch.where(available, direction[:, 0] / norm.clamp_min(1e-6), 1.0)
    sine = torch.where(available, direction[:, 1] / norm.clamp_min(1e-6), 0.0)

    output = features.clone()
    presence = (output[..., 3:4] > 0.5).to(output.dtype)
    x_coordinate = output[..., 0].clone()
    y_coordinate = output[..., 1].clone()
    cosine = cosine.view(-1, 1, 1)
    sine = sine.view(-1, 1, 1)
    output[..., 0] = (x_coordinate * cosine + y_coordinate * sine) * presence[..., 0]
    output[..., 1] = (-x_coordinate * sine + y_coordinate * cosine) * presence[..., 0]
    output[..., :3] *= presence
    output[..., 4:5] *= presence
    return output


# Directed hand/arm bones. Face anchors are not a physical mesh and are therefore
# deliberately excluded rather than inventing facial bones from sparse points.
BONE_PARENT_INDEX = tuple(
    [59, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    + [60, 21, 22, 23, 24, 21, 26, 27, 28, 21, 30, 31, 32, 21, 34, 35, 36, 21, 38, 39, 40]
    + list(range(42, 57))
    + [58, 58, 57, 58]
)
BONE_HAS_PARENT = tuple(
    [True] * 21
    + [True] * 21
    + [False] * 15
    + [True, False, True, True]
)


def masked_bone_features(features: torch.Tensor) -> torch.Tensor:
    """Derive directed bone vectors and bone motion without crossing missing data."""
    if features.ndim != 4 or features.shape[-2:] != (NUM_NODES, NUM_CHANNELS):
        raise ValueError(
            f"expected [B, T, {NUM_NODES}, {NUM_CHANNELS}], got {tuple(features.shape)}"
        )
    xyz = features[..., :3]
    presence = features[..., 3:4] > 0.5
    parents = torch.tensor(BONE_PARENT_INDEX, device=features.device)
    has_parent = torch.tensor(
        BONE_HAS_PARENT, device=features.device, dtype=features.dtype
    ).view(1, 1, NUM_NODES, 1)
    parent_xyz = xyz.index_select(2, parents)
    parent_presence = presence.index_select(2, parents)
    valid = presence.to(features.dtype) * parent_presence.to(features.dtype) * has_parent
    bone = (xyz - parent_xyz) * valid

    motion = torch.zeros_like(bone)
    motion_valid = valid[:, 1:] * valid[:, :-1]
    motion[:, 1:] = (bone[:, 1:] - bone[:, :-1]) * motion_valid
    return torch.cat((bone, motion), dim=-1)


HAND_FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)

HAND_BONE_EDGES = tuple(
    (parent, child)
    for chain in HAND_FINGER_CHAINS
    for parent, child in zip(chain[:-1], chain[1:])
)


def wrist_relative_hand_features(features: torch.Tensor) -> torch.Tensor:
    """Return wrist-relative XYZ and validity for both 21-joint hands."""
    if features.ndim != 4 or features.shape[-2:] != (NUM_NODES, NUM_CHANNELS):
        raise ValueError(
            f"expected [B, T, {NUM_NODES}, {NUM_CHANNELS}], got {tuple(features.shape)}"
        )
    output = []
    for offset in (0, 21):
        hand_xyz = features[..., offset : offset + 21, :3]
        hand_presence = features[..., offset : offset + 21, 3:4] > 0.5
        wrist_xyz = features[..., offset : offset + 1, :3]
        wrist_presence = features[..., offset : offset + 1, 3:4] > 0.5
        valid = hand_presence & wrist_presence
        relative = (hand_xyz - wrist_xyz) * valid.to(features.dtype)
        output.append(torch.cat((relative, valid.to(features.dtype)), dim=-1).flatten(2))
    return torch.cat(output, dim=-1)


def masked_hand_bone_geometry(features: torch.Tensor) -> torch.Tensor:
    """Return 40 unit bone directions, lengths, and validity flags per frame."""
    if features.ndim != 4 or features.shape[-2:] != (NUM_NODES, NUM_CHANNELS):
        raise ValueError(
            f"expected [B, T, {NUM_NODES}, {NUM_CHANNELS}], got {tuple(features.shape)}"
        )
    xyz = features[..., :3]
    presence = features[..., 3] > 0.5
    bones = []
    for offset in (0, 21):
        for parent, child in HAND_BONE_EDGES:
            parent += offset
            child += offset
            vector = xyz[..., child, :] - xyz[..., parent, :]
            length = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
            valid = (
                presence[..., parent]
                & presence[..., child]
                & (length[..., 0] > 1e-8)
            ).unsqueeze(-1)
            direction = vector / length.clamp_min(1e-8)
            bones.append(torch.cat(
                (
                    direction * valid.to(features.dtype),
                    length * valid.to(features.dtype),
                    valid.to(features.dtype),
                ),
                dim=-1,
            ))
    return torch.stack(bones, dim=-2)


def static_hand_frame_weights(
    features: torch.Tensor, mode: str, *, top_k: int = 3
) -> torch.Tensor:
    """Select reliable hand frames by quality alone or quality plus low motion.

    The output is ``[B, T, 2]`` for left/right hands. Each observed hand sums to one;
    a completely absent hand remains zero. The low-motion treatment and quality-only
    control use the same number of frames and differ only in their ranking score.
    """
    if features.ndim != 4 or features.shape[-2:] != (NUM_NODES, NUM_CHANNELS):
        raise ValueError(
            f"expected [B, T, {NUM_NODES}, {NUM_CHANNELS}], got {tuple(features.shape)}"
        )
    if mode not in ("quality", "low_motion"):
        raise ValueError("static hand frame mode must be quality or low_motion")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    batch, frames = features.shape[:2]
    outputs = []
    for offset in (0, 21):
        hand = features[..., offset : offset + 21, :]
        present = hand[..., 3] > 0.5
        count = present.sum(dim=-1)
        quality = (
            hand[..., 4].clamp(0.0, 1.0) * present.to(features.dtype)
        ).sum(dim=-1) / count.clamp_min(1).to(features.dtype)
        reliable = count >= 12
        score = quality
        if mode == "low_motion":
            transition = present[:, 1:] & present[:, :-1]
            transition_count = transition.sum(dim=-1)
            speed = torch.linalg.vector_norm(
                hand[:, 1:, :, :3] - hand[:, :-1, :, :3], dim=-1
            )
            speed = (speed * transition.to(features.dtype)).sum(dim=-1)
            speed = speed / transition_count.clamp_min(1).to(features.dtype)
            motion = torch.full_like(quality, torch.inf)
            motion[:, 1:] = speed
            motion_reliable = reliable.clone()
            motion_reliable[:, 0] = False
            motion_reliable[:, 1:] &= transition_count >= 12
            # Fall back to quality-ranked frames only when no reliable transition
            # exists for a hand, as happens in very short or heavily occluded clips.
            has_motion = motion_reliable.any(dim=1, keepdim=True)
            reliable = torch.where(has_motion, motion_reliable, reliable)
            finite_motion = torch.where(reliable, motion, torch.zeros_like(motion))
            maximum = finite_motion.amax(dim=1, keepdim=True)
            minimum = torch.where(
                reliable, motion, torch.full_like(motion, torch.inf)
            ).amin(dim=1, keepdim=True)
            minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
            normalized_motion = (motion - minimum) / (maximum - minimum).clamp_min(1e-6)
            normalized_motion = torch.where(
                torch.isfinite(normalized_motion), normalized_motion, torch.ones_like(motion)
            )
            score = torch.where(has_motion, quality - normalized_motion, quality)
        score = score.masked_fill(~reliable, torch.finfo(score.dtype).min)
        selected_count = min(top_k, frames)
        indices = score.topk(selected_count, dim=1).indices
        weights = torch.zeros(batch, frames, device=features.device, dtype=features.dtype)
        weights.scatter_(1, indices, 1.0)
        weights *= reliable.to(features.dtype)
        weights /= weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        outputs.append(weights)
    return torch.stack(outputs, dim=-1)


def articulated_bone_distance(
    first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    """Missing-aware length-weighted angular distance between hand bone sets.

    Inputs are outputs of :func:`masked_hand_bone_geometry` and may contain
    broadcast dimensions. Distances are zero when no corresponding bone is valid.
    """
    if first.shape[-2:] != (40, 5) or second.shape[-2:] != (40, 5):
        raise ValueError("articulated bone geometry must end in [40, 5]")
    valid = (first[..., 4] > 0.5) & (second[..., 4] > 0.5)
    cosine = (first[..., :3] * second[..., :3]).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    weight = 0.5 * (first[..., 3] + second[..., 3]) * valid.to(first.dtype)
    return (angle * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1e-8)


def masked_hand_angle_features(features: torch.Tensor) -> torch.Tensor:
    """Return cosine flexion angles at 30 internal hand joints.

    Cosines avoid the unstable derivative of ``acos`` near straight fingers. A value
    is zero unless the parent, center, and child observations are all present.
    """
    if features.ndim != 4 or features.shape[-2:] != (NUM_NODES, NUM_CHANNELS):
        raise ValueError(
            f"expected [B, T, {NUM_NODES}, {NUM_CHANNELS}], got {tuple(features.shape)}"
        )
    xyz = features[..., :3]
    presence = features[..., 3] > 0.5
    angles = torch.zeros(
        *features.shape[:-1], 1, device=features.device, dtype=features.dtype
    )
    for offset in (0, 21):
        for chain in HAND_FINGER_CHAINS:
            for parent, center, child in zip(chain[:-2], chain[1:-1], chain[2:]):
                parent += offset
                center += offset
                child += offset
                valid = presence[..., parent] & presence[..., center] & presence[..., child]
                incoming = xyz[..., parent, :] - xyz[..., center, :]
                outgoing = xyz[..., child, :] - xyz[..., center, :]
                denominator = (
                    torch.linalg.vector_norm(incoming, dim=-1)
                    * torch.linalg.vector_norm(outgoing, dim=-1)
                )
                valid = valid & (denominator > 1e-8)
                cosine = (incoming * outgoing).sum(dim=-1) / denominator.clamp_min(1e-8)
                angles[..., center, 0] = cosine.clamp(-1.0, 1.0) * valid.to(features.dtype)
    return angles


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return value
        keep = 1.0 - self.probability
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        mask = torch.empty(shape, device=value.device, dtype=value.dtype).bernoulli_(keep)
        return value * mask / keep


class SqueezeformerBlockV17(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        conv_kernel: int,
        dropout: float,
        drop_path: float,
        attention_score_mixing: bool = False,
    ):
        super().__init__()
        self.ff1_norm = nn.LayerNorm(dim)
        self.ff1 = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = (
            ScoreMixingMultiheadAttentionV17(dim, heads, dropout)
            if attention_score_mixing
            else nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(
                dim, dim, conv_kernel, padding=conv_kernel // 2, groups=dim
            ),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
            nn.Dropout(dropout),
        )
        self.ff2_norm = nn.LayerNorm(dim)
        self.ff2 = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.drop_path(0.5 * self.ff1(self.ff1_norm(value)))
        normalized = self.attention_norm(value)
        if isinstance(self.attention, ScoreMixingMultiheadAttentionV17):
            attended = self.attention(normalized)
        else:
            attended, _ = self.attention(
                normalized, normalized, normalized, need_weights=False
            )
        value = value + self.drop_path(self.attention_dropout(attended))
        convolved = self.conv(self.conv_norm(value).transpose(1, 2)).transpose(1, 2)
        value = value + self.drop_path(convolved)
        value = value + self.drop_path(0.5 * self.ff2(self.ff2_norm(value)))
        return self.output_norm(value)


class ScoreMixingMultiheadAttentionV17(nn.Module):
    """MHA plus one zero-initialized depthwise 3x3 residual over score maps."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.base = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        # Creating the extra layer must not change initialization of later baseline
        # modules. Restore the CPU RNG after construction, then start as exact zero.
        rng_state = torch.random.get_rng_state()
        self.score_mixer = nn.Conv2d(
            heads, heads, kernel_size=3, padding=1, groups=heads, bias=False
        )
        torch.random.set_rng_state(rng_state)
        nn.init.zeros_(self.score_mixer.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, frames, _ = value.shape
        projected = F.linear(
            value, self.base.in_proj_weight, self.base.in_proj_bias
        )
        query, key, content = projected.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, frames, self.heads, self.head_dim).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        content = split_heads(content)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores + self.score_mixer(scores)
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.base.dropout, training=self.training)
        attended = torch.matmul(weights, content).transpose(1, 2).reshape(
            batch, frames, self.dim
        )
        return self.base.out_proj(attended)


class KeypointTemporalGateV17(nn.Module):
    """Learn a light, independent temporal importance curve for every keypoint.

    The gate sees only detector confidence, XYZ speed, and XYZ acceleration for one
    node at a time. Its final depthwise convolution is zero-initialized, so enabling
    the component starts as an exact identity rather than perturbing the baseline
    before it has learned useful temporal emphasis.
    """

    SIGNALS_PER_NODE = 3

    def __init__(self) -> None:
        super().__init__()
        self.temporal = nn.Conv1d(
            NUM_NODES * self.SIGNALS_PER_NODE,
            NUM_NODES,
            kernel_size=3,
            padding=1,
            groups=NUM_NODES,
        )
        self.output = nn.Conv1d(
            NUM_NODES, NUM_NODES, kernel_size=1, groups=NUM_NODES
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, model_input: torch.Tensor, derived: torch.Tensor
    ) -> torch.Tensor:
        if model_input.ndim != 4 or model_input.shape[2] != NUM_NODES:
            raise ValueError(
                f"expected model input [B, T, {NUM_NODES}, C], "
                f"got {tuple(model_input.shape)}"
            )
        if derived.shape[:-1] != model_input.shape[:-1] or derived.shape[-1] != 11:
            raise ValueError(
                f"expected matching derived [B, T, {NUM_NODES}, 11], "
                f"got {tuple(derived.shape)}"
            )
        confidence = derived[..., 4]
        speed = torch.linalg.vector_norm(derived[..., 5:8], dim=-1)
        acceleration = torch.linalg.vector_norm(derived[..., 8:11], dim=-1)
        signals = torch.stack((confidence, speed, acceleration), dim=-1)
        batch, frames, nodes, channels = signals.shape
        signals = signals.permute(0, 2, 3, 1).reshape(
            batch, nodes * channels, frames
        )
        logits = self.output(F.gelu(self.temporal(signals)))
        gate = 1.0 + torch.tanh(logits).transpose(1, 2).unsqueeze(-1)
        return model_input * gate


class ArticulatedPoseEmbeddingV17(nn.Module):
    """Compact frame-wise hand-geometry branch for distance pretraining."""

    INPUT_DIM = 2 * 21 * 4
    OUTPUT_DIM = 64

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.OUTPUT_DIM),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedding = self.network(wrist_relative_hand_features(features))
        active = features[..., :42, 3].amax(dim=-1, keepdim=True) > 0.5
        embedding = F.normalize(embedding, dim=-1, eps=1e-8)
        return embedding * active.to(embedding.dtype)


class PartWiseTemporalEncoderV17(nn.Module):
    """Temporally encode isolated anatomical streams before whole-body fusion."""

    PARTS = {
        "left_hand": (0, 21),
        "right_hand": (21, 42),
        "face": (42, 57),
        "body": (57, 61),
    }

    def __init__(
        self,
        output_dim: int,
        heads: int,
        conv_kernel: int,
        dropout: float,
        drop_path: float,
        depth: int,
        *,
        use_pairwise: bool,
        node_channels: int,
        attention_score_mixing: bool = False,
    ):
        super().__init__()
        part_dim = output_dim // len(self.PARTS)
        if part_dim % heads:
            raise ValueError("part dimension must be divisible by attention heads")
        pairwise_per_hand = 33 if use_pairwise else 0
        input_dims = {
            "left_hand": 21 * node_channels + pairwise_per_hand,
            "right_hand": 21 * node_channels + pairwise_per_hand,
            "face": 15 * node_channels,
            "body": 4 * node_channels,
        }
        self.use_pairwise = use_pairwise
        self.part_dim = part_dim
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(input_dims[name], part_dim),
                    nn.LayerNorm(part_dim),
                    nn.Dropout(dropout),
                )
                for name in self.PARTS
            }
        )
        self.positions = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(1, 32, part_dim))
                for name in self.PARTS
            }
        )
        for position in self.positions.values():
            nn.init.trunc_normal_(position, std=0.02)
        self.blocks = nn.ModuleDict(
            {
                name: nn.ModuleList(
                    SqueezeformerBlockV17(
                        part_dim,
                        heads,
                        conv_kernel,
                        dropout,
                        drop_path * index / max(depth - 1, 1),
                        attention_score_mixing,
                    )
                    for index in range(depth)
                )
                for name in self.PARTS
            }
        )
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def project_parts(
        self, derived: torch.Tensor, pairwise: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        projected: dict[str, torch.Tensor] = {}
        for name, (start, end) in self.PARTS.items():
            part = derived[:, :, start:end].flatten(2)
            if self.use_pairwise and name in ("left_hand", "right_hand"):
                if pairwise is None or pairwise.shape[-1] != 66:
                    raise ValueError("part-wise hand encoding requires 66 pairwise features")
                pair_start = 0 if name == "left_hand" else 33
                part = torch.cat((part, pairwise[..., pair_start : pair_start + 33]), dim=-1)
            projected[name] = self.projections[name](part)
        return projected

    def encode_parts(
        self, derived: torch.Tensor, pairwise: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        projected = self.project_parts(derived, pairwise)
        outputs: dict[str, torch.Tensor] = {}
        for name in self.PARTS:
            value = projected[name] + self.positions[name][:, : derived.shape[1]]
            for block in self.blocks[name]:
                value = block(value)
            outputs[name] = value
        return outputs

    def fuse_parts(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if tuple(outputs) != tuple(self.PARTS):
            raise ValueError(f"expected ordered parts {tuple(self.PARTS)}, got {tuple(outputs)}")
        return self.fusion(torch.cat([outputs[name] for name in self.PARTS], dim=-1))

    def forward(
        self, derived: torch.Tensor, pairwise: torch.Tensor | None
    ) -> torch.Tensor:
        return self.fuse_parts(self.encode_parts(derived, pairwise))


def anatomical_adjacency_v17() -> torch.Tensor:
    """Return a fixed physical graph for the 61-node v17 schema.

    Global learned attention is modeled separately. Keeping this matrix physical
    prevents the common failure mode where a learnable residual silently turns an
    anatomical graph into a dense graph at initialization.
    """
    adjacency = torch.eye(NUM_NODES, dtype=torch.float32)

    hand_chains = (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    )
    for offset in (0, 21):
        for chain in hand_chains:
            for first, second in zip(chain, chain[1:]):
                adjacency[offset + first, offset + second] = 1.0
                adjacency[offset + second, offset + first] = 1.0
        for first, second in zip((5, 9, 13), (9, 13, 17)):
            adjacency[offset + first, offset + second] = 1.0
            adjacency[offset + second, offset + first] = 1.0

    # Face topology: eyes/brows, nose, lips, and jaw contour.
    face_edges = (
        (42, 44), (42, 45), (43, 46), (43, 47),
        (44, 45), (46, 47), (44, 56), (47, 56),
        (56, 48), (48, 49), (48, 50), (49, 51), (50, 51),
        (49, 52), (50, 54), (52, 53), (53, 54), (53, 48),
    )
    body_edges = ((57, 58), (57, 59), (58, 60), (59, 0), (60, 21))
    for first, second in face_edges + body_edges:
        adjacency[first, second] = 1.0
        adjacency[second, first] = 1.0

    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


class GraphSpatialBlockV17(nn.Module):
    """Fuse physical graph messages with input-sensitive global joint attention."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.physical_norm = nn.LayerNorm(dim)
        self.physical_projection = nn.Linear(dim, dim)
        self.global_norm = nn.LayerNorm(dim)
        self.global_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Linear(dim * 2, dim)
        self.output_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, value: torch.Tensor, presence: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        physical_input = self.physical_norm(value)
        physical = torch.einsum("nm,bmd->bnd", adjacency, physical_input)
        physical = self.physical_projection(physical)

        safe_presence = presence.clone()
        empty = ~safe_presence.any(dim=1)
        if empty.any():
            safe_presence[empty, 0] = True
        normalized = self.global_norm(value)
        global_value, _ = self.global_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~safe_presence,
            need_weights=False,
        )
        mix = torch.sigmoid(self.gate(torch.cat((physical, global_value), dim=-1)))
        fused = mix * physical + (1.0 - mix) * global_value
        value = value + fused
        value = value + self.feed_forward(self.output_norm(value))
        return value * presence.unsqueeze(-1).to(value.dtype)


class GraphPartEncoderV17(nn.Module):
    """Encode anatomical nodes, then preserve hands/face/body as explicit parts."""

    PART_RANGES = ((0, 21), (21, 42), (42, 57), (57, 61), (0, 61))

    def __init__(
        self, output_dim: int, node_dim: int, layers: int, heads: int, dropout: float
    ):
        super().__init__()
        self.register_buffer("adjacency", anatomical_adjacency_v17())
        self.node_projection = nn.Sequential(
            nn.Linear(11, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
        )
        self.node_embedding = nn.Parameter(torch.zeros(1, NUM_NODES, node_dim))
        nn.init.trunc_normal_(self.node_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            GraphSpatialBlockV17(node_dim, heads, dropout) for _ in range(layers)
        )
        self.part_attention = nn.Linear(node_dim, 1)
        self.output_projection = nn.Sequential(
            nn.Linear(node_dim * len(self.PART_RANGES), output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, derived: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        batch, frames, nodes, _ = derived.shape
        value = self.node_projection(derived) + self.node_embedding
        value = value * presence.unsqueeze(-1).to(value.dtype)
        value = value.reshape(batch * frames, nodes, -1)
        flat_presence = presence.reshape(batch * frames, nodes)
        for block in self.blocks:
            value = block(value, flat_presence, self.adjacency)

        scores = self.part_attention(value).squeeze(-1)
        pooled_parts = []
        for start, end in self.PART_RANGES:
            part_presence = flat_presence[:, start:end]
            safe_presence = part_presence.clone()
            empty = ~safe_presence.any(dim=1)
            if empty.any():
                safe_presence[empty, 0] = True
            part_scores = scores[:, start:end].masked_fill(
                ~safe_presence, torch.finfo(scores.dtype).min
            )
            weights = F.softmax(part_scores, dim=1)
            pooled = (value[:, start:end] * weights.unsqueeze(-1)).sum(dim=1)
            pooled_parts.append(pooled * (~empty).unsqueeze(-1).to(pooled.dtype))
        output = self.output_projection(torch.cat(pooled_parts, dim=-1))
        return output.reshape(batch, frames, -1)


class SLTStage1V17(nn.Module):
    """Squeezeformer classifier with missing-aware internal feature derivation."""

    HAND_PAIRS = (
        (4, 8), (4, 12), (4, 16), (4, 20),
        (8, 12), (8, 16), (8, 20), (12, 16), (12, 20), (16, 20),
        (4, 2), (8, 5), (12, 9), (16, 13), (20, 17),
        (5, 9), (9, 13), (13, 17), (5, 17),
        (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
        (4, 5), (4, 9), (4, 13),
        (5, 20), (0, 9), (7, 11), (11, 15), (15, 19), (3, 7),
    )
    MODALITY_RANGES = {
        "hands": ((0, 42),),
        "face": ((42, 57),),
        "mouth": ((49, 53),),
    }

    def __init__(self, config: Stage1V17Config | None = None):
        super().__init__()
        self.config = config or Stage1V17Config()
        self.config.validate()
        pairwise_dim = len(self.HAND_PAIRS) * 2 if self.config.use_pairwise else 0
        node_channels = (
            11
            + (6 if self.config.use_bone_features else 0)
            + (1 if self.config.use_hand_angle_features else 0)
        )
        if self.config.temporal_encoder == "partwise_global":
            self.input_projection = None
            self.graph_encoder = None
            self.pairwise_projection = None
            self.register_parameter("graph_residual_scale", None)
            self.part_temporal_encoder = PartWiseTemporalEncoderV17(
                self.config.dim,
                self.config.heads,
                self.config.conv_kernel,
                self.config.dropout,
                self.config.drop_path,
                self.config.part_depth,
                use_pairwise=self.config.use_pairwise,
                node_channels=node_channels,
                attention_score_mixing=self.config.use_attention_score_mixing,
            )
        elif self.config.spatial_encoder in ("flat", "flat_graph_residual"):
            self.part_temporal_encoder = None
            input_dim = NUM_NODES * node_channels + pairwise_dim
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, self.config.dim),
                nn.LayerNorm(self.config.dim),
                nn.Dropout(self.config.dropout),
            )
            if self.config.spatial_encoder == "flat_graph_residual":
                self.graph_encoder = GraphPartEncoderV17(
                    self.config.dim,
                    self.config.graph_node_dim,
                    self.config.graph_layers,
                    self.config.graph_heads,
                    self.config.dropout,
                )
                # Zero makes a hybrid checkpoint exactly equal to its flat warm start.
                # tanh keeps the learned correction bounded without blocking its gradient.
                self.graph_residual_scale = nn.Parameter(torch.zeros(()))
            else:
                self.graph_encoder = None
                self.register_parameter("graph_residual_scale", None)
            self.pairwise_projection = None
        else:
            self.part_temporal_encoder = None
            self.input_projection = None
            self.graph_encoder = GraphPartEncoderV17(
                self.config.dim,
                self.config.graph_node_dim,
                self.config.graph_layers,
                self.config.graph_heads,
                self.config.dropout,
            )
            self.pairwise_projection = (
                nn.Linear(pairwise_dim, self.config.dim) if pairwise_dim else None
            )
            self.register_parameter("graph_residual_scale", None)
        self.keypoint_temporal_gate = (
            KeypointTemporalGateV17()
            if self.config.use_keypoint_temporal_gate
            else None
        )
        self.articulated_pose_embedding = (
            ArticulatedPoseEmbeddingV17()
            if self.config.use_articulated_pose_embedding
            else None
        )
        self.articulated_pose_fusion = (
            nn.Sequential(
                nn.Linear(
                    self.config.dim + ArticulatedPoseEmbeddingV17.OUTPUT_DIM,
                    self.config.dim,
                ),
                nn.LayerNorm(self.config.dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            )
            if self.config.use_articulated_pose_embedding
            else None
        )
        self.static_hand_projection = (
            nn.Sequential(
                nn.Linear(self.config.dim // 2, self.config.dim),
                nn.LayerNorm(self.config.dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            )
            if self.config.static_hand_token != "none"
            else None
        )
        if self.config.static_hand_token != "none":
            self.static_hand_residual_scale = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("static_hand_residual_scale", None)
        self.position = nn.Parameter(torch.zeros(1, 32, self.config.dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        rates = torch.linspace(0.0, self.config.drop_path, self.config.depth).tolist()
        self.blocks = nn.ModuleList(
            SqueezeformerBlockV17(
                self.config.dim,
                self.config.heads,
                self.config.conv_kernel,
                self.config.dropout,
                rates[index],
                self.config.use_attention_score_mixing,
            )
            for index in range(self.config.depth)
        )
        self.frame_attention = nn.Sequential(
            nn.Linear(self.config.dim, self.config.dim // 4),
            nn.GELU(),
            nn.Linear(self.config.dim // 4, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.config.dim),
            nn.Linear(self.config.dim, self.config.dim * 2),
            nn.GELU(),
            nn.Dropout(self.config.head_dropout),
            nn.Linear(self.config.dim * 2, self.config.num_classes),
        )
        self.phonology_heads = nn.ModuleDict(
            {
                name: nn.Linear(self.config.dim, size)
                for name, size in self.config.phonology_head_sizes
            }
        )
        part_dim = self.config.dim // 4
        self.part_auxiliary_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(part_dim),
                    nn.Linear(part_dim, self.config.num_classes),
                )
                for name in PartWiseTemporalEncoderV17.PARTS
            }
            if self.config.use_part_auxiliary
            else {}
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _pairwise(self, raw: torch.Tensor) -> torch.Tensor:
        xyz = raw[..., :3]
        presence = raw[..., 3]
        output = []
        for offset in (0, 21):
            for first, second in self.HAND_PAIRS:
                valid = presence[:, :, offset + first] * presence[:, :, offset + second]
                distance = torch.linalg.vector_norm(
                    xyz[:, :, offset + first] - xyz[:, :, offset + second], dim=-1
                )
                output.append(distance * valid)
        return torch.stack(output, dim=-1)

    def _encode_internal(
        self, features: torch.Tensor, *, return_parts: bool = False
    ) -> tuple[
        torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None
    ]:
        if features.shape[1] > self.position.shape[1]:
            raise ValueError(f"at most {self.position.shape[1]} frames are supported")
        if self.config.canonicalize_camera_roll:
            features = canonicalize_camera_roll_v17(features)
        if self.config.input_modality != "all":
            masked = torch.zeros_like(features)
            for start, end in self.MODALITY_RANGES[self.config.input_modality]:
                masked[:, :, start:end] = features[:, :, start:end]
            features = masked
        derived = masked_temporal_features(features)
        input_features = [derived]
        if self.config.use_bone_features:
            input_features.append(masked_bone_features(features))
        if self.config.use_hand_angle_features:
            input_features.append(masked_hand_angle_features(features))
        model_input = torch.cat(input_features, dim=-1)
        if self.keypoint_temporal_gate is not None:
            model_input = self.keypoint_temporal_gate(model_input, derived)
        if self.config.temporal_encoder == "partwise_global":
            if self.part_temporal_encoder is None:
                raise RuntimeError("part-wise temporal encoder is unavailable")
            pairwise = self._pairwise(features) if self.config.use_pairwise else None
            if return_parts:
                part_outputs = self.part_temporal_encoder.encode_parts(model_input, pairwise)
                encoded = self.part_temporal_encoder.fuse_parts(part_outputs)
            else:
                part_outputs = None
                encoded = self.part_temporal_encoder(model_input, pairwise)
            if self.articulated_pose_embedding is not None:
                if self.articulated_pose_fusion is None:
                    raise RuntimeError("articulated pose fusion is unavailable")
                geometry = self.articulated_pose_embedding(features)
                encoded = self.articulated_pose_fusion(
                    torch.cat((encoded, geometry), dim=-1)
                )
        elif self.config.spatial_encoder in ("flat", "flat_graph_residual"):
            part_outputs = None
            per_frame = model_input.flatten(2)
            if self.config.use_pairwise:
                per_frame = torch.cat((per_frame, self._pairwise(features)), dim=-1)
            if self.input_projection is None:
                raise RuntimeError("flat input projection is unavailable")
            encoded = self.input_projection(per_frame)
            if self.config.spatial_encoder == "flat_graph_residual":
                if self.graph_encoder is None or self.graph_residual_scale is None:
                    raise RuntimeError("graph residual encoder is unavailable")
                presence = features[..., 3] > 0.5
                graph = self.graph_encoder(derived, presence)
                encoded = encoded + torch.tanh(self.graph_residual_scale) * graph
        else:
            part_outputs = None
            if self.graph_encoder is None:
                raise RuntimeError("graph encoder is unavailable")
            presence = features[..., 3] > 0.5
            encoded = self.graph_encoder(derived, presence)
            if self.pairwise_projection is not None:
                encoded = encoded + self.pairwise_projection(self._pairwise(features))
        encoded = encoded + self.position[:, : features.shape[1]]
        for block in self.blocks:
            encoded = block(encoded)
        hand_activity = features[:, :, :42, 3].amax(dim=-1) > 0.5
        return encoded, hand_activity, part_outputs

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, hand_activity, _ = self._encode_internal(features)
        return encoded, hand_activity

    @staticmethod
    def _masked_mean(value: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        weights = active.to(value.dtype).unsqueeze(-1)
        return (value * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _pool(self, features: torch.Tensor) -> torch.Tensor:
        needs_parts = self.config.static_hand_token != "none"
        encoded, active, part_outputs = self._encode_internal(
            features, return_parts=needs_parts
        )
        scores = self.frame_attention(encoded).squeeze(-1)
        has_active = active.any(dim=1, keepdim=True)
        usable = torch.where(has_active, active, torch.ones_like(active))
        scores = scores.masked_fill(~usable, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=1)
        pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        if needs_parts:
            if (
                part_outputs is None
                or self.static_hand_projection is None
                or self.static_hand_residual_scale is None
            ):
                raise RuntimeError("static hand token modules are unavailable")
            static_weights = static_hand_frame_weights(
                features, self.config.static_hand_token
            )
            left = (
                part_outputs["left_hand"] * static_weights[..., 0:1]
            ).sum(dim=1)
            right = (
                part_outputs["right_hand"] * static_weights[..., 1:2]
            ).sum(dim=1)
            static = self.static_hand_projection(torch.cat((left, right), dim=-1))
            pooled = pooled + torch.tanh(self.static_hand_residual_scale) * static
        return pooled

    def forward(
        self, features: torch.Tensor, return_embeddings: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pooled = self._pool(features)
        logits = self.classifier(pooled)
        return (logits, pooled) if return_embeddings else logits

    def forward_multitask(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        pooled = self._pool(features)
        logits = self.classifier(pooled)
        auxiliary = {name: head(pooled) for name, head in self.phonology_heads.items()}
        return logits, pooled, auxiliary

    def forward_part_auxiliary(
        self, features: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        if not self.part_auxiliary_heads:
            raise RuntimeError("part auxiliary heads are disabled")
        encoded, active, part_outputs = self._encode_internal(
            features, return_parts=True
        )
        if part_outputs is None:
            raise RuntimeError("part-wise outputs are unavailable")
        scores = self.frame_attention(encoded).squeeze(-1)
        has_active = active.any(dim=1, keepdim=True)
        usable = torch.where(has_active, active, torch.ones_like(active))
        scores = scores.masked_fill(~usable, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=1)
        pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        auxiliary: dict[str, torch.Tensor] = {}
        part_valid: dict[str, torch.Tensor] = {}
        for name, (start, end) in PartWiseTemporalEncoderV17.PARTS.items():
            part_active = features[:, :, start:end, 3].amax(dim=-1) > 0.5
            part_pooled = self._masked_mean(part_outputs[name], part_active)
            auxiliary[name] = self.part_auxiliary_heads[name](part_pooled)
            part_valid[name] = part_active.any(dim=1)
        return self.classifier(pooled), pooled, auxiliary, part_valid

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def make_stage1_checkpoint(
    model: SLTStage1V17,
    state_dict: dict[str, torch.Tensor],
    *,
    epoch: int,
    validation_metrics: dict[str, float],
    label_to_index: dict[str, int],
    manifest_sha256: str,
    schema_fingerprint: str,
) -> dict[str, object]:
    return {
        "format": "slt_stage1_v17",
        "epoch": int(epoch),
        "model_config": model.config.to_dict(),
        "model_state_dict": state_dict,
        "validation_metrics": validation_metrics,
        "label_to_index": label_to_index,
        "manifest_sha256": manifest_sha256,
        "schema_fingerprint": schema_fingerprint,
    }
