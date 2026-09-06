"""Shared causal observation and data contract for continuous v17 recognition.

Every corpus, including isolated replay, goes through the same rolling observation.
Only published/manual intervals define hard background. Local phrase boundaries are
unknown and are deliberately left to the sequence objective.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from active.v17.geometry_v17 import resample_features
from active.v17.model_streaming_stage1_head_v17 import Stage1EvidenceBlock
from active.v17.train_stage_1_reel_emission_v17 import asllrp_annotations
from active.v17.train_streaming_tcn_ctc_v17 import refuse_protected, restore_source_frames


@dataclass(frozen=True)
class ContinuousConfig:
    windows: tuple[int, ...] = (8, 16, 32)
    stride: int = 4
    fps: float = 30.0
    stage1_dim: int = 256
    num_glosses: int = 100
    hidden_dim: int = 192
    blocks: int = 3
    dropout: float = 0.15

    def __post_init__(self):
        if not self.windows or min(self.windows) < 2 or self.stride < 1 or self.fps <= 0:
            raise ValueError("invalid rolling observation contract")

    @property
    def scale_dim(self):
        return self.stage1_dim + self.num_glosses

    @property
    def input_dim(self):
        return len(self.windows) * self.scale_dim

    @property
    def other_index(self):
        return self.num_glosses + 1

    def to_dict(self):
        return asdict(self)


def observation_ends(frames: int, stride: int, *, final: bool = True) -> list[int]:
    if frames < 1 or stride < 1:
        raise ValueError("positive frame count and stride required")
    ends = list(range(stride, frames + 1, stride))
    if final and (not ends or ends[-1] != frames):
        ends.append(frames)
    return ends


def observation_windows(frames: np.ndarray, end: int, windows: tuple[int, ...]) -> np.ndarray:
    if not 1 <= end <= len(frames):
        raise ValueError("observation endpoint outside available frames")
    return np.stack([
        resample_features(frames[max(0, end - width):end], 32).astype(np.float32)
        for width in windows
    ])


def annotated_gaps(frames: int, intervals: list[tuple[int, int, int]], minimum: int = 4):
    """Complement of ALL lexical intervals, including OTHER and overlapping signs."""
    cursor = 0
    for left, right, _ in sorted(intervals):
        left, right = max(0, left), min(frames, right)
        if left - cursor >= minimum:
            yield cursor, left
        cursor = max(cursor, right)
    if frames - cursor >= minimum:
        yield cursor, frames


def interval_targets(ends: list[int], intervals: list[tuple[int, int, int]], width: int):
    """Supervise only windows wholly inside one interval or a true interior gap.

Windows crossing a lexical boundary are ignored. CTC aligns these rather than
forcing sign fragments to blank or inventing an emission step for every annotation.
"""
    labels = np.full(len(ends), -100, np.int64)
    if not intervals:
        return labels
    for i, end in enumerate(ends):
        start = max(0, end - width)
        overlapping = [(l, r, t) for l, r, t in intervals if l < end and r > start]
        if len(overlapping) == 1:
            left, right, target = overlapping[0]
            if left <= start and end <= right:
                labels[i] = target
        elif not overlapping:
            labels[i] = 0
    return labels


@dataclass
class ContinuousSample:
    frames: np.ndarray
    targets: tuple[int, ...]
    source: str
    identity: str
    signer: str
    intervals: list[tuple[int, int, int]]
    path: str


def load_continuous_samples(root: Path, role: str, labels: dict[str, int]):
    refuse_protected((root,))
    if role not in {"train", "validation"}:
        raise ValueError("only development train/validation roles allowed")
    ncslgr = {
        row["source_item_id"]: row
        for row in json.loads(Path("active/v17/ncslgr_supervised_manifest_v17.json").read_text())["rows"]
    }
    asllrp = asllrp_annotations(
        Path("data/local/asllrp_contiguous_phrases_v17/manifest.json"),
        Path("data/local/asllrp_segmented_citizen100_v17/manifest.json"),
    )
    samples = []
    for path in sorted((root / role).glob("*/*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata_json"].item()))
            if meta["role"] != role:
                raise ValueError(f"split mismatch: {path}")
            frames = restore_source_frames(archive["landmarks"], archive["window_source_ranges"])
            targets = tuple(int(t) + 1 for t in archive["target_indices"])
        identity, source = str(meta["source_item_id"]), str(meta["source"])
        intervals = []
        if source == "ncslgr_strict":
            row = ncslgr[identity]
            if row["role"] != role:
                raise ValueError("NCSLGR timed annotation split mismatch")
            intervals = [
                (int(e["source_start_frame"]), int(e["source_end_frame_exclusive"]),
                 labels.get(e["canonical_label"], len(labels)) + 1)
                for e in row["events"]
            ]
        elif source == "asllrp_contiguous":
            span, signs = asllrp[identity]
            crop = int(span["utterance_start_frame_global"]) + int(span["crop_start_frame_local"])
            intervals = [
                (int(s["sign_start_frame"]) - crop, int(s["sign_end_frame"]) - crop + 1,
                 labels[str(s["canonical_label"])] + 1) for s in signs
            ]
            if tuple(t for _, _, t in intervals) != targets:
                raise ValueError("ASLLRP timed labels disagree with sequence")
        signer = str(meta.get("signer_id", meta.get("participant_id", "")))
        if source == "ncslgr_strict":
            signer = str(ncslgr[identity]["participant_id"])
        sample = ContinuousSample(frames, targets, source, identity, signer, intervals, str(path))
        samples.append(sample)
        if intervals:
            for left, right in annotated_gaps(len(frames), intervals):
                samples.append(ContinuousSample(
                    frames[left:right], (), f"blank:{source}", f"{identity}:gap:{left}:{right}",
                    signer, [(0, right - left, 0)], str(path),
                ))
    if not samples:
        raise ValueError(f"no continuous samples in {root / role}")
    return samples


def load_isolated_streams(root: Path, labels: dict[str, int], source: str):
    refuse_protected((root,))
    samples = []
    for label, target in sorted(labels.items()):
        for path in sorted((root / label).glob("*.v17.npz")):
            with np.load(path, allow_pickle=False) as archive:
                features = archive["features"].astype(np.float32)
                meta = json.loads(str(archive["metadata_json"].item()))
            # Restore the recorded approximate duration instead of treating every
            # isolated clip as one decoder step. Fine motion lost upstream remains lost.
            sampled = int(meta.get("sampled_frame_count", 32))
            decoded = int(meta.get("decoded_frame_count", sampled))
            processed = int(meta.get("source_frames_processed", 32))
            fps = float(meta.get("fps", 30)) or 30.0
            count = round(processed * max(1, decoded - 1) / max(1, sampled - 1) * 30 / fps)
            count = max(4, min(256, count))
            frames = resample_features(features, count).astype(np.float32)
            samples.append(ContinuousSample(
                frames, (target + 1,), f"isolated:{source}", str(path), "", [], str(path)
            ))
    if not samples:
        raise ValueError(f"no isolated replay under {root}")
    return samples


class ContinuousEvidenceModel(nn.Module):
    """Causal multi-duration fusion with a residual isolated recognition path."""

    def __init__(self, config: ContinuousConfig):
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(config.input_dim)
        self.projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.blocks = nn.ModuleList([
            Stage1EvidenceBlock(config.hidden_dim, 3, 2**i, config.dropout)
            for i in range(config.blocks)
        ])
        self.scale_gate = nn.Linear(config.hidden_dim, len(config.windows))
        self.output = nn.Linear(config.hidden_dim, config.num_glosses + 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.scale_gate.weight)
        nn.init.zeros_(self.scale_gate.bias)

    def forward(self, evidence: torch.Tensor, return_embeddings: bool = False):
        if evidence.ndim != 3 or evidence.shape[-1] != self.config.input_dim:
            raise ValueError("continuous evidence contract mismatch")
        value = F.gelu(self.projection(self.norm(evidence)))
        for block in self.blocks:
            value = block(value)
        scales = evidence.reshape(*evidence.shape[:2], len(self.config.windows), self.config.scale_dim)
        glosses = scales[..., self.config.stage1_dim:]
        gate = self.scale_gate(value).softmax(-1).unsqueeze(-1)
        residual = (glosses * gate).sum(-2)
        logits = self.output(value)
        logits = logits + F.pad(residual, (1, 1))
        return (logits, value) if return_embeddings else logits
