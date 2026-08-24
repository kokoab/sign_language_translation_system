#!/usr/bin/env python3
"""Train v17 Stage 1 on ASL Citizen's official signer-disjoint splits."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
)

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.model_v17 import (
        SLTStage1V17,
        Stage1V17Config,
        make_stage1_checkpoint,
    )
    from active.v17.schema_mediapipe_v17 import (
        MediaPipeV17Config,
        schema_fingerprint as mediapipe_schema_fingerprint,
    )
    from active.v17.schema_v17 import (
        CLIP_LENGTH,
        MOUTH_END,
        MOUTH_START,
        NUM_CHANNELS,
        NUM_NODES,
        V17Config,
        schema_fingerprint,
    )
else:
    from .model_v17 import SLTStage1V17, Stage1V17Config, make_stage1_checkpoint
    from .schema_mediapipe_v17 import (
        MediaPipeV17Config,
        schema_fingerprint as mediapipe_schema_fingerprint,
    )
    from .schema_v17 import (
        CLIP_LENGTH,
        MOUTH_END,
        MOUTH_START,
        NUM_CHANNELS,
        NUM_NODES,
        V17Config,
        schema_fingerprint,
    )


LOG = logging.getLogger("stage1_v17")
SPLITS = ("train", "val", "test")
EXPECTED_SHAPE = (CLIP_LENGTH, NUM_NODES, NUM_CHANNELS)
EXTRACTORS = ("apple", "mediapipe_t50")


def mask_mouth_nodes_v17(features: torch.Tensor) -> torch.Tensor:
    """Remove only lip/mouth evidence while retaining facial location geometry."""
    output = features.clone()
    output[..., MOUTH_START:MOUTH_END, :] = 0
    return output

# Horizontal reflection must exchange every anatomical left/right landmark, not
# only the hand ranges. Brow endpoints reverse because the face-contour direction
# is mirrored.
MIRROR_NODE_INDEX = tuple(
    list(range(21, 42))
    + list(range(0, 21))
    + [43, 42, 47, 46, 45, 44, 48, 50, 49, 51, 52, 55, 54, 53, 56]
    + [58, 57, 60, 59]
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_phonology_supervision(
    path: Path, manifest_path: Path, label_to_index: dict[str, int]
) -> dict[str, object]:
    """Load manifest-locked class-level ASL-LEX phonological targets."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "slt_v17_asllex_phonology_targets":
        raise ValueError("unsupported phonology target format")
    if payload.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("phonology targets do not match the current Citizen manifest")
    if int(payload.get("class_count", -1)) != len(label_to_index):
        raise ValueError("phonology target class count does not match the dataset")

    classes = payload.get("classes")
    if not isinstance(classes, list) or len(classes) != len(label_to_index):
        raise ValueError("phonology targets have an invalid class table")
    for item in classes:
        label = str(item["canonical_label"])
        if label_to_index.get(label) != int(item["class_index"]):
            raise ValueError("phonology target label mapping does not match the dataset")

    target_maps: dict[str, torch.Tensor] = {}
    head_sizes = []
    attributes = payload.get("attributes")
    if not isinstance(attributes, list) or not attributes:
        raise ValueError("phonology targets contain no attributes")
    for item in attributes:
        name = str(item["name"])
        values = item["values"]
        targets = torch.tensor(item["targets_by_class_index"], dtype=torch.long)
        if name in target_maps or len(values) < 2 or len(targets) != len(label_to_index):
            raise ValueError(f"invalid phonology target attribute: {name}")
        valid = targets != -100
        if valid.any() and (targets[valid].min() < 0 or targets[valid].max() >= len(values)):
            raise ValueError(f"phonology target index is out of range: {name}")
        target_maps[name] = targets
        head_sizes.append((name, len(values)))
    return {
        "head_sizes": tuple(head_sizes),
        "target_maps": target_maps,
        "sha256": sha256_file(path),
        "path": str(path),
        "attributes": [
            {
                "name": str(item["name"]),
                "classes": len(item["values"]),
                "annotated_classes": int(item["annotated_classes"]),
            }
            for item in attributes
        ],
    }


def phonology_auxiliary_loss(
    auxiliary_logits: dict[str, torch.Tensor],
    class_target_maps: dict[str, torch.Tensor],
    class_targets: torch.Tensor,
) -> torch.Tensor:
    """Average valid per-attribute losses without making missing labels a class."""
    losses = []
    for name, logits in auxiliary_logits.items():
        if name not in class_target_maps:
            raise ValueError(f"missing phonology targets for head: {name}")
        targets = class_target_maps[name].index_select(0, class_targets)
        if (targets != -100).any():
            losses.append(F.cross_entropy(logits, targets, ignore_index=-100))
    if not losses:
        if not auxiliary_logits:
            raise ValueError("phonology auxiliary loss requires at least one head")
        return next(iter(auxiliary_logits.values())).sum() * 0.0
    return torch.stack(losses).mean()


def part_auxiliary_loss(
    auxiliary_logits: dict[str, torch.Tensor],
    targets: torch.Tensor,
    part_valid: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Mean gloss loss across observed anatomical streams only."""
    expected = {"left_hand", "right_hand", "face", "body"}
    if set(auxiliary_logits) != expected or set(part_valid) != expected:
        raise ValueError(
            f"part auxiliary logits require {sorted(expected)}, "
            f"got logits={sorted(auxiliary_logits)} valid={sorted(part_valid)}"
        )
    losses = []
    for name in sorted(expected):
        valid = part_valid[name]
        if valid.ndim != 1 or len(valid) != len(targets) or valid.dtype != torch.bool:
            raise ValueError(f"invalid part availability mask: {name}")
        if valid.any():
            losses.append(F.cross_entropy(auxiliary_logits[name][valid], targets[valid]))
    if not losses:
        return next(iter(auxiliary_logits.values())).sum() * 0.0
    return torch.stack(losses).mean()


def initialize_flat_graph_residual(
    model: SLTStage1V17,
    checkpoint_path: Path,
    manifest_path: Path,
    expected_schema: str,
) -> dict[str, object]:
    """Warm-start a zero-gated graph residual without weakening provenance checks."""
    if model.config.spatial_encoder != "flat_graph_residual":
        raise ValueError("--initialize-from currently requires flat_graph_residual")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("initial checkpoint is not a v17 Stage 1 checkpoint")
    if checkpoint.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("initial checkpoint manifest mismatch")
    if checkpoint.get("schema_fingerprint") != expected_schema:
        raise ValueError("initial checkpoint extractor schema mismatch")
    source_config = Stage1V17Config(**checkpoint["model_config"])
    if source_config.spatial_encoder != "flat":
        raise ValueError("initial checkpoint must use the flat spatial encoder")
    shared_fields = (
        "num_classes", "dim", "depth", "heads", "conv_kernel", "dropout",
        "head_dropout", "drop_path", "use_pairwise", "input_modality",
    )
    mismatches = [
        name for name in shared_fields
        if getattr(source_config, name) != getattr(model.config, name)
    ]
    if mismatches:
        raise ValueError(f"initial checkpoint config mismatch: {mismatches}")
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_missing = {
        key for key in model.state_dict()
        if key == "graph_residual_scale"
        or key.startswith("graph_encoder.")
        or key.startswith("phonology_heads.")
    }
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if incompatible.unexpected_keys or unexpected_missing:
        raise ValueError(
            "unsafe partial initialization: "
            f"missing={sorted(unexpected_missing)} "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    if set(incompatible.missing_keys) != allowed_missing:
        raise ValueError("initial checkpoint unexpectedly supplied challenger-only keys")
    return {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "source_epoch": int(checkpoint.get("epoch", -1)),
        "source_validation_metrics": checkpoint.get("validation_metrics", {}),
        "loaded_shared_keys": len(checkpoint["model_state_dict"]),
        "new_keys": sorted(incompatible.missing_keys),
        "zero_initialized_residual": True,
    }


def initialize_exact_stage1_finetune(
    model: SLTStage1V17,
    checkpoint_path: Path,
    manifest_path: Path,
    expected_schema: str,
    label_to_index: dict[str, int],
) -> dict[str, object]:
    """Strictly restore a selected model for replay-based domain adaptation."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("fine-tune checkpoint is not a v17 Stage 1 checkpoint")
    if checkpoint.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("fine-tune checkpoint Citizen manifest mismatch")
    if checkpoint.get("schema_fingerprint") != expected_schema:
        raise ValueError("fine-tune checkpoint extractor schema mismatch")
    if checkpoint.get("label_to_index") != label_to_index:
        raise ValueError("fine-tune checkpoint label mapping mismatch")
    if checkpoint.get("model_config") != model.config.to_dict():
        raise ValueError("fine-tune checkpoint model config mismatch")
    provenance = checkpoint.get("training_data_provenance", {})
    if (
        checkpoint.get("test_evaluated") is True
        or provenance.get("citizen_test_accessed") is not False
        or provenance.get("semlex_test_accessed") is not False
    ):
        raise ValueError("fine-tune checkpoint lacks sealed-test provenance")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return {
        "mode": "exact_selected_checkpoint_replay_finetune",
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "source_epoch": int(checkpoint.get("epoch", -1)),
        "source_validation_metrics": checkpoint.get("validation_metrics", {}),
        "strict_state_dict": True,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
    }


def initialize_articulated_pose_embedding(
    model: SLTStage1V17,
    checkpoint_path: Path,
    manifest_path: Path,
    supplement_manifest_path: Path,
    expected_schema: str,
) -> dict[str, object]:
    """Load only the self-supervised geometry MLP into a capacity-matched branch."""
    if model.articulated_pose_embedding is None:
        raise ValueError("articulated pose pretraining requires the embedding branch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_v17_articulated_pose_pretrain":
        raise ValueError("not a v17 articulated-pose pretraining checkpoint")
    if checkpoint.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("articulated-pose pretraining Citizen manifest mismatch")
    if checkpoint.get("supplement_manifest_sha256") != sha256_file(
        supplement_manifest_path
    ):
        raise ValueError("articulated-pose pretraining supplement manifest mismatch")
    if checkpoint.get("schema_fingerprint") != expected_schema:
        raise ValueError("articulated-pose pretraining extractor schema mismatch")
    model.articulated_pose_embedding.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    return {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "format": checkpoint["format"],
        "epochs": int(checkpoint.get("epochs", -1)),
        "triplets": int(checkpoint.get("triplets", -1)),
        "objective": checkpoint.get("objective"),
    }


def initialize_masked_pose_encoder(
    model: SLTStage1V17,
    checkpoint_path: Path,
    manifest_path: Path,
    supplement_manifest_path: Path,
    expected_schema: str,
) -> dict[str, object]:
    """Strictly load only a masked-pose-pretrained part-wise temporal encoder."""
    if model.config.temporal_encoder != "partwise_global":
        raise ValueError("masked pose pretraining requires partwise_global")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_v17_masked_pose_pretrain":
        raise ValueError("not a v17 masked-pose pretraining checkpoint")
    if checkpoint.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("masked-pose pretraining Citizen manifest mismatch")
    if checkpoint.get("supplement_manifest_sha256") != sha256_file(
        supplement_manifest_path
    ):
        raise ValueError("masked-pose pretraining supplement manifest mismatch")
    if checkpoint.get("schema_fingerprint") != expected_schema:
        raise ValueError("masked-pose pretraining extractor schema mismatch")
    if checkpoint.get("model_config") != model.config.to_dict():
        raise ValueError("masked-pose pretraining model config mismatch")
    state = checkpoint.get("encoder_state_dict")
    if not isinstance(state, dict):
        raise ValueError("masked-pose checkpoint has no encoder state")
    prefixes = ("part_temporal_encoder.", "position", "blocks.")
    expected_keys = {
        key for key in model.state_dict() if key.startswith(prefixes)
    }
    if set(state) != expected_keys:
        raise ValueError("masked-pose encoder keys do not exactly match the model")
    current = model.state_dict()
    for key, value in state.items():
        if current[key].shape != value.shape:
            raise ValueError(f"masked-pose encoder tensor shape mismatch: {key}")
        current[key] = value
    model.load_state_dict(current, strict=True)
    return {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "format": checkpoint["format"],
        "epochs": int(checkpoint.get("epochs", -1)),
        "objective": checkpoint.get("objective"),
        "loaded_encoder_keys": len(state),
    }


def load_rejections(path: Path | None) -> set[tuple[str, str, str]]:
    if path is None or not path.exists():
        return set()
    rejected = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rejected.add((row["split"], row["canonical_label"], row["video"]))
    return rejected


def extractor_schema_fingerprint(extractor: str) -> str:
    if extractor == "apple":
        return schema_fingerprint(V17Config())
    if extractor == "mediapipe_t50":
        config = MediaPipeV17Config(
            minimum_hand_detection_confidence=0.50,
            minimum_hand_presence_confidence=0.50,
            minimum_hand_tracking_confidence=0.50,
        )
        return mediapipe_schema_fingerprint(config)
    raise ValueError(f"extractor must be one of {EXTRACTORS}")


class Citizen100V17Dataset(Dataset):
    """Strict loader for ``split/canonical-label/*.v17.npz`` archives."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        manifest_path: str | Path,
        rejection_path: str | Path | None = None,
        *,
        cache: bool = True,
        expected_schema: str | None = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.root = Path(root)
        self.split = split
        self.source_name = "citizen"
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        classes = sorted(manifest["classes"], key=lambda item: item["class_index"])
        indices = [int(item["class_index"]) for item in classes]
        if indices != list(range(len(classes))):
            raise ValueError("manifest class indices must be contiguous from zero")
        self.label_to_index = {
            str(item["canonical_label"]): int(item["class_index"])
            for item in classes
        }
        self.index_to_label = {value: key for key, value in self.label_to_index.items()}
        self.num_classes = len(classes)
        self.expected_schema = expected_schema or schema_fingerprint(V17Config())
        rejected = load_rejections(Path(rejection_path) if rejection_path else None)

        files: list[Path] = []
        targets: list[int] = []
        split_root = self.root / split
        for label, target in self.label_to_index.items():
            class_root = split_root / label
            if not class_root.is_dir():
                raise FileNotFoundError(f"missing class directory: {class_root}")
            selected = []
            for feature_path in sorted(class_root.glob("*.v17.npz")):
                video_name = feature_path.name.removesuffix(".v17.npz") + ".mp4"
                if (split, label, video_name) not in rejected:
                    selected.append(feature_path)
            if not selected:
                raise ValueError(f"no usable {split} samples for class {label}")
            files.extend(selected)
            targets.extend([target] * len(selected))

        self.files = files
        self.targets = torch.tensor(targets, dtype=torch.long)
        self._cached: list[torch.Tensor] | None = None
        if cache:
            self._cached = [self._load(path) for path in self.files]

    def _load(self, path: Path) -> torch.Tensor:
        return load_v17_archive(path, self.expected_schema)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._cached[index] if self._cached is not None else self._load(self.files[index])
        return features, self.targets[index]

    def balanced_subset(self, samples_per_class: int) -> Subset:
        remaining = {index: samples_per_class for index in range(self.num_classes)}
        selected = []
        for index, target in enumerate(self.targets.tolist()):
            if remaining[target] > 0:
                selected.append(index)
                remaining[target] -= 1
        if any(remaining.values()):
            raise ValueError("requested subset exceeds at least one class count")
        return Subset(self, selected)


def load_v17_archive(path: Path, expected_schema: str) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as payload:
        features = payload["features"]
        metadata = json.loads(str(payload["metadata_json"]))
    if tuple(features.shape) != EXPECTED_SHAPE:
        raise ValueError(f"{path}: expected {EXPECTED_SHAPE}, got {features.shape}")
    if metadata.get("schema_fingerprint") != expected_schema:
        raise ValueError(f"{path}: v17 schema fingerprint mismatch")
    features = features.astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise ValueError(f"{path}: features contain non-finite values")
    presence = features[..., 3]
    if not np.isin(presence, (0.0, 1.0)).all():
        raise ValueError(f"{path}: presence is not binary")
    return torch.from_numpy(features.copy())


class SemLexSupplementV17Dataset(Dataset):
    """Strict train-only loader for a reviewed SemLex selection manifest."""

    def __init__(
        self,
        root: str | Path,
        manifest_path: str | Path,
        label_to_index: dict[str, int],
        *,
        cache: bool = True,
        expected_schema: str | None = None,
    ):
        self.root = Path(root)
        self.source_name = "semlex"
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("split") != "train_only":
            raise ValueError("SemLex supplement manifest must be train_only")
        videos = manifest.get("videos")
        if not isinstance(videos, list) or not videos:
            raise ValueError("SemLex supplement manifest has no videos")
        if int(manifest.get("selected_clips", -1)) != len(videos):
            raise ValueError("SemLex supplement selected_clips does not match videos")
        self.expected_schema = expected_schema or schema_fingerprint(V17Config())
        self.label_to_index = dict(label_to_index)
        self.num_classes = len(label_to_index)

        files: list[Path] = []
        targets: list[int] = []
        seen: set[tuple[str, str]] = set()
        for row in videos:
            if row.get("semlex_split") != "train":
                raise ValueError("SemLex supplement contains a non-train clip")
            label = str(row.get("canonical_label", ""))
            if label not in self.label_to_index:
                raise ValueError(f"SemLex supplement has unknown class: {label}")
            video_id = str(row.get("semlex_video_id", ""))
            if not video_id or Path(video_id).name != video_id:
                raise ValueError(f"unsafe SemLex video id: {video_id!r}")
            key = (label, video_id)
            if key in seen:
                raise ValueError(f"duplicate SemLex selection: {label}/{video_id}")
            seen.add(key)
            feature_path = self.root / label / f"{video_id}.v17.npz"
            if not feature_path.is_file():
                raise FileNotFoundError(f"missing SemLex feature: {feature_path}")
            files.append(feature_path)
            targets.append(self.label_to_index[label])

        self.files = files
        self.targets = torch.tensor(targets, dtype=torch.long)
        self._cached: list[torch.Tensor] | None = None
        if cache:
            self._cached = [load_v17_archive(path, self.expected_schema) for path in files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = (
            self._cached[index]
            if self._cached is not None
            else load_v17_archive(self.files[index], self.expected_schema)
        )
        return features, self.targets[index]


class LocalReviewSupplementV17Dataset(Dataset):
    """Strict train-only loader for reviewed or official exact-variant supplements."""

    def __init__(
        self,
        root: str | Path,
        manifest_path: str | Path,
        label_to_index: dict[str, int],
        *,
        allowed_tiers: tuple[str, ...] = ("tier_a_dual_top1",),
        cache: bool = True,
        expected_schema: str | None = None,
        mask_mouth_nodes: bool = False,
    ):
        self.root = Path(root)
        self.source_name = "local"
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.manifest_format = str(manifest.get("format", ""))
        self.known_signers = tuple(
            sorted(
                str(value).strip()
                for value in manifest.get("signers", [])
                if str(value).strip()
            )
        )
        split_eligibility = manifest.get("split_eligibility")
        if split_eligibility not in (
            "train_only_after_human_review",
            "train_only_after_ASL_fluent_exact_variant_review",
            "train_only_official_asllex_signbank_cross_reference",
        ):
            raise ValueError("local supplement manifest must be approved train-only data")
        if self.manifest_format == "slt_v17_local_deep_clean_final_v1" and (
            manifest.get("split") != "train"
            or manifest.get("extraction_complete") is not True
            or manifest.get("extractor_schema_fingerprint")
            != (expected_schema or schema_fingerprint(V17Config()))
            or manifest.get("signer_disjoint") is not False
            or manifest.get("signer_overlap_user_approved") is not True
            or manifest.get("citizen_test_accessed") is not False
            or manifest.get("semlex_test_accessed") is not False
        ):
            raise ValueError("invalid finalized local training extraction contract")
        official_exact = (
            split_eligibility
            == "train_only_official_asllex_signbank_cross_reference"
        )
        if official_exact:
            official_format = self.manifest_format
            if official_format not in (
                "slt_v17_asllrp_asllex_exact_supplement",
                "slt_v17_asllvd_asllex_exact_supplement",
            ):
                raise ValueError("official exact supplement has an invalid format")
            if manifest.get("citizen_test_accessed") is not False:
                raise ValueError("official exact supplement must prove Citizen test isolation")
            if manifest.get("semlex_test_accessed") is not False:
                raise ValueError("official exact supplement must prove SemLex test isolation")
        videos = manifest.get("videos")
        if not isinstance(videos, list) or not videos:
            raise ValueError("local supplement manifest has no videos")
        if (
            self.manifest_format == "slt_v17_local_deep_clean_final_v1"
            and int(manifest.get("selected_clips", -1)) != len(videos)
        ):
            raise ValueError("finalized local training selected_clips mismatch")
        if not allowed_tiers or len(set(allowed_tiers)) != len(allowed_tiers):
            raise ValueError("local supplement tiers must be non-empty and unique")
        self.allowed_tiers = tuple(allowed_tiers)
        self.expected_schema = expected_schema or schema_fingerprint(V17Config())
        self.label_to_index = dict(label_to_index)
        self.num_classes = len(label_to_index)
        self.mask_mouth_nodes = bool(mask_mouth_nodes)

        files: list[Path] = []
        targets: list[int] = []
        seen: set[tuple[str, str]] = set()
        for row in videos:
            if row.get("consensus_tier") not in self.allowed_tiers:
                continue
            if self.manifest_format == "slt_v17_local_deep_clean_final_v1" and (
                row.get("local_split") != "train"
                or row.get("training_eligible") is not True
                or row.get("validation_eligible") is not False
            ):
                raise ValueError("finalized local training row violates split contract")
            if official_exact:
                annotation = str(row.get("signbank_annotation_id", "")).strip()
                common_invalid = (
                    row.get("training_eligible") is not True
                    or row.get("consensus_tier") != "official_asllex_signbank_exact"
                    or not str(row.get("citizen_asl_lex_code", "")).strip()
                    or not str(row.get("asllex_entry_id", "")).strip()
                    or not annotation
                )
                if official_format == "slt_v17_asllvd_asllex_exact_supplement":
                    variant_invalid = (
                        str(row.get("variant_gloss", "")).strip().rstrip("+")
                        != annotation
                        or not str(row.get("signer", "")).strip()
                    )
                else:
                    variant_invalid = (
                        str(row.get("asllrp_entry_variant", "")).strip() != annotation
                        or str(row.get("asllrp_occurrence", "")).strip().rstrip("+")
                        != annotation
                    )
                if common_invalid or variant_invalid:
                    raise ValueError("official exact supplement row violates variant contract")
            label = str(row.get("canonical_label", ""))
            if label not in self.label_to_index:
                raise ValueError(f"local supplement has unknown class: {label}")
            raw_path = Path(str(row.get("raw_path", "")))
            if not raw_path.name:
                raise ValueError("local supplement row has no raw filename")
            feature_path = self.root / label / f"{raw_path.stem}.v17.npz"
            declared_path = Path(str(row.get("feature_path", "")))
            if declared_path.name != feature_path.name or declared_path.parent.name != label:
                raise ValueError(f"local feature provenance mismatch: {declared_path}")
            key = (label, raw_path.name)
            if key in seen:
                raise ValueError(f"duplicate local selection: {label}/{raw_path.name}")
            seen.add(key)
            if not feature_path.is_file():
                raise FileNotFoundError(f"missing local feature: {feature_path}")
            files.append(feature_path)
            targets.append(self.label_to_index[label])
        if not files:
            raise ValueError("local supplement has no clips in the approved tiers")

        self.files = files
        self.targets = torch.tensor(targets, dtype=torch.long)
        self._cached: list[torch.Tensor] | None = None
        if cache:
            self._cached = [load_v17_archive(path, self.expected_schema) for path in files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = (
            self._cached[index]
            if self._cached is not None
            else load_v17_archive(self.files[index], self.expected_schema)
        )
        if self.mask_mouth_nodes:
            features = mask_mouth_nodes_v17(features)
        return features, self.targets[index]


class LocalValidationV17Dataset(Dataset):
    """Strict non-signer-disjoint validation loader for the local deep-clean run."""

    def __init__(
        self,
        root: str | Path,
        manifest_path: str | Path,
        label_to_index: dict[str, int],
        *,
        cache: bool = True,
        expected_schema: str | None = None,
        mask_mouth_nodes: bool = False,
    ):
        self.root = Path(root)
        self.source_name = "local_validation"
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest_format = manifest.get("format")
        if (
            manifest_format
            not in {
                "slt_v17_local_deep_clean_v1",
                "slt_v17_local_deep_clean_final_v1",
            }
            or manifest.get("split") != "val"
            or manifest.get("split_eligibility")
            != "validation_nonsigner_disjoint_user_approved"
            or manifest.get("signer_disjoint") is not False
            or manifest.get("signer_overlap_user_approved") is not True
            or manifest.get("citizen_test_accessed") is not False
            or manifest.get("semlex_test_accessed") is not False
        ):
            raise ValueError("invalid local deep-clean validation contract")
        if manifest_format == "slt_v17_local_deep_clean_final_v1" and (
            manifest.get("extraction_complete") is not True
            or manifest.get("extractor_schema_fingerprint")
            != (expected_schema or schema_fingerprint(V17Config()))
        ):
            raise ValueError("invalid finalized local validation extraction contract")
        videos = manifest.get("videos")
        if not isinstance(videos, list) or not videos:
            raise ValueError("local validation manifest has no videos")
        if int(manifest.get("selected_clips", -1)) != len(videos):
            raise ValueError("local validation selected_clips mismatch")
        self.expected_schema = expected_schema or schema_fingerprint(V17Config())
        self.label_to_index = dict(label_to_index)
        self.num_classes = len(label_to_index)
        self.mask_mouth_nodes = bool(mask_mouth_nodes)

        files: list[Path] = []
        targets: list[int] = []
        seen: set[tuple[str, str]] = set()
        for row in videos:
            if (
                row.get("local_split") != "val"
                or row.get("validation_eligible") is not True
                or row.get("training_eligible") is not False
            ):
                raise ValueError("local validation row violates split contract")
            label = str(row.get("canonical_label", ""))
            if label not in self.label_to_index:
                raise ValueError(f"local validation has unknown class: {label}")
            item_id = str(row.get("item_id", ""))
            if not item_id or Path(item_id).name != item_id:
                raise ValueError(f"unsafe local validation item id: {item_id!r}")
            key = (label, item_id)
            if key in seen:
                raise ValueError(f"duplicate local validation item: {label}/{item_id}")
            seen.add(key)
            feature_path = self.root / label / f"{item_id}.v17.npz"
            declared_path = Path(str(row.get("feature_path", "")))
            if declared_path.name != feature_path.name or declared_path.parent.name != label:
                raise ValueError(f"local validation feature provenance mismatch: {declared_path}")
            if not feature_path.is_file():
                raise FileNotFoundError(f"missing local validation feature: {feature_path}")
            files.append(feature_path)
            targets.append(self.label_to_index[label])
        self.files = files
        self.targets = torch.tensor(targets, dtype=torch.long)
        self._cached: list[torch.Tensor] | None = None
        if cache:
            self._cached = [load_v17_archive(path, self.expected_schema) for path in files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = (
            self._cached[index]
            if self._cached is not None
            else load_v17_archive(self.files[index], self.expected_schema)
        )
        if self.mask_mouth_nodes:
            features = mask_mouth_nodes_v17(features)
        return features, self.targets[index]


def dataset_targets_and_sources(dataset: Dataset) -> tuple[torch.Tensor, list[str]]:
    """Return aligned targets/source names through Subset and ConcatDataset wrappers."""
    if isinstance(dataset, Subset):
        targets, sources = dataset_targets_and_sources(dataset.dataset)
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        return targets.index_select(0, indices), [sources[index] for index in indices]
    if isinstance(dataset, ConcatDataset):
        parts = [dataset_targets_and_sources(part) for part in dataset.datasets]
        return (
            torch.cat([targets for targets, _ in parts]),
            [source for _, sources in parts for source in sources],
        )
    targets = getattr(dataset, "targets", None)
    source_name = getattr(dataset, "source_name", None)
    if not isinstance(targets, torch.Tensor) or not isinstance(source_name, str):
        raise TypeError("dataset must expose tensor targets and a source_name")
    return targets.clone().long(), [source_name] * len(targets)


def class_source_balanced_weights(
    targets: torch.Tensor,
    sources: list[str],
    num_classes: int,
    source_probabilities: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Balance class and source margins over existing class/source cells.

    Iterative proportional fitting assigns equal expected probability to every class
    and every source while respecting structural zeros (for example, Citizen-only
    classes). Each sample in a populated class/source cell receives equal weight.
    """
    if targets.ndim != 1 or len(targets) != len(sources) or not len(targets):
        raise ValueError("targets and sources must be non-empty and aligned")
    if int(targets.min()) < 0 or int(targets.max()) >= num_classes:
        raise ValueError("target outside configured class range")
    source_names = sorted(set(sources))
    source_to_index = {name: index for index, name in enumerate(source_names)}
    counts = torch.zeros(num_classes, len(source_names), dtype=torch.float64)
    for target, source in zip(targets.tolist(), sources):
        counts[int(target), source_to_index[source]] += 1.0
    if (counts.sum(dim=1) == 0).any():
        raise ValueError("balanced sampling requires every class to be represented")

    mass = (counts > 0).to(torch.float64)
    class_margin = torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float64)
    if source_probabilities is None:
        source_margin = torch.full(
            (len(source_names),), 1.0 / len(source_names), dtype=torch.float64
        )
    else:
        if set(source_probabilities) != set(source_names):
            raise ValueError("source probabilities must name every and only active source")
        values = [float(source_probabilities[name]) for name in source_names]
        if any(value <= 0.0 for value in values) or not np.isclose(sum(values), 1.0):
            raise ValueError("source probabilities must be positive and sum to one")
        source_margin = torch.tensor(values, dtype=torch.float64)
    for _ in range(200):
        mass *= (class_margin / mass.sum(dim=1)).unsqueeze(1)
        mass *= (source_margin / mass.sum(dim=0)).unsqueeze(0)
    if not torch.allclose(mass.sum(dim=1), class_margin, atol=1e-9, rtol=0.0):
        raise ValueError("class/source margins are structurally infeasible")
    if not torch.allclose(mass.sum(dim=0), source_margin, atol=1e-9, rtol=0.0):
        raise ValueError("class/source margins did not converge")

    weights = torch.empty(len(targets), dtype=torch.float64)
    for index, (target, source) in enumerate(zip(targets.tolist(), sources)):
        source_index = source_to_index[source]
        weights[index] = mass[int(target), source_index] / counts[int(target), source_index]
    summary: dict[str, object] = {
        "mode": "class_source_balanced_replacement",
        "expected_class_probability": 1.0 / num_classes,
        "expected_source_probabilities": {
            name: float(mass[:, source_to_index[name]].sum()) for name in source_names
        },
        "minimum_sample_weight": float(weights.min()),
        "maximum_sample_weight": float(weights.max()),
    }
    return weights, summary


def parse_source_probabilities(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None
    output: dict[str, float] = {}
    for item in value.split(","):
        name, separator, probability = item.partition("=")
        name = name.strip()
        if not separator or not name or name in output:
            raise ValueError("source probabilities must use unique NAME=VALUE entries")
        output[name] = float(probability)
    return output


def mirror_v17(features: torch.Tensor) -> torch.Tensor:
    """Reflect a batch and exchange all left/right anatomical nodes."""
    if features.ndim != 4 or features.shape[-2:] != (NUM_NODES, NUM_CHANNELS):
        raise ValueError(f"unexpected feature shape: {tuple(features.shape)}")
    index = torch.tensor(MIRROR_NODE_INDEX, device=features.device)
    mirrored = features.index_select(2, index).clone()
    mirrored[..., 0] = -mirrored[..., 0]
    return mirrored


def rotate_camera_roll_v17(
    features: torch.Tensor, angle_radians: torch.Tensor | float
) -> torch.Tensor:
    """Rotate camera-plane XY by arbitrary per-sample angles.

    v17 coordinates are isotropic before body-relative normalization, so this is a
    real camera-roll transform rather than an aspect-ratio distortion. Depth,
    presence, confidence, and exact missing zeros are preserved.
    """
    if features.ndim != 4 or features.shape[-2:] != (NUM_NODES, NUM_CHANNELS):
        raise ValueError(f"unexpected feature shape: {tuple(features.shape)}")
    batch = features.shape[0]
    angles = torch.as_tensor(
        angle_radians, dtype=features.dtype, device=features.device
    )
    if angles.ndim == 0:
        angles = angles.expand(batch)
    if angles.shape != (batch,):
        raise ValueError(f"camera-roll angles must have shape [{batch}]")
    output = features.clone()
    presence = (output[..., 3:4] > 0.5).to(output.dtype)
    cosine = torch.cos(angles).view(batch, 1, 1)
    sine = torch.sin(angles).view(batch, 1, 1)
    x_coordinate = output[..., 0].clone()
    y_coordinate = output[..., 1].clone()
    output[..., 0] = (x_coordinate * cosine - y_coordinate * sine) * presence[..., 0]
    output[..., 1] = (x_coordinate * sine + y_coordinate * cosine) * presence[..., 0]
    output[..., :3] *= presence
    output[..., 4:5] *= presence
    return output


def augment_v17(
    features: torch.Tensor,
    *,
    full_roll_probability: float = 0.35,
    maximum_roll_degrees: float = 180.0,
    mild_roll_degrees: float = 12.0,
) -> torch.Tensor:
    """Apply geometry-valid augmentation while preserving the missing-value contract."""
    if not 0.0 <= full_roll_probability <= 1.0:
        raise ValueError("full-roll probability must be in [0, 1]")
    if not 0.0 <= mild_roll_degrees <= maximum_roll_degrees <= 180.0:
        raise ValueError("roll degrees must satisfy 0 <= mild <= maximum <= 180")
    value = features.clone()
    batch, frames, nodes, _ = value.shape
    device = value.device

    mirror_mask = torch.rand(batch, device=device) < 0.5
    if mirror_mask.any():
        value[mirror_mask] = mirror_v17(value[mirror_mask])

    presence = value[..., 3:4]
    scale = 0.88 + 0.24 * torch.rand(batch, 1, 1, 1, device=device)
    mild = (torch.rand(batch, device=device) * 2.0 - 1.0) * mild_roll_degrees
    full = (torch.rand(batch, device=device) * 2.0 - 1.0) * maximum_roll_degrees
    use_full = torch.rand(batch, device=device) < full_roll_probability
    angle = torch.where(use_full, full, mild) * (np.pi / 180.0)
    value = rotate_camera_roll_v17(value, angle)
    value[..., :2] *= scale
    translation = (torch.rand(batch, 1, 1, 2, device=device) - 0.5) * 0.08
    value[..., :2] = value[..., :2] + translation * presence

    if torch.rand((), device=device) < 0.6:
        noise = torch.randn_like(value[..., :3]) * 0.008
        value[..., :3] = value[..., :3] + noise * presence

    # Nearest-neighbour temporal warping duplicates/skips real frames and never
    # invents fractional presence values or interpolates through missing joints.
    if torch.rand((), device=device) < 0.5:
        warped = value.clone()
        base = torch.linspace(0.0, 1.0, frames, device=device)
        for sample in range(batch):
            rate = float(0.82 + 0.36 * torch.rand((), device=device))
            positions = ((base - 0.5) * rate + 0.5).clamp(0.0, 1.0)
            indices = (positions * (frames - 1)).round().long()
            warped[sample] = value[sample].index_select(0, indices)
        value = warped

    if torch.rand((), device=device) < 0.25:
        for sample in range(batch):
            count = int(torch.randint(2, 7, (), device=device))
            indices = torch.randperm(nodes, device=device)[:count]
            value[sample, :, indices] = 0.0

    presence = (value[..., 3:4] > 0.5).to(value.dtype)
    value[..., 3:4] = presence
    value[..., :3] *= presence
    value[..., 4:5] = value[..., 4:5].clamp(0.0, 1.0) * presence
    return value


def partmix_hands_v17(
    features: torch.Tensor,
    targets: torch.Tensor,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace one complete hand with a non-self donor and mix the two labels."""
    if features.ndim != 4 or features.shape[-2:] != (NUM_NODES, NUM_CHANNELS):
        raise ValueError(f"unexpected feature shape: {tuple(features.shape)}")
    if targets.ndim != 1 or len(targets) != len(features):
        raise ValueError("features and targets must have aligned batch dimensions")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("partmix probability must be in [0, 1]")

    batch = len(features)
    donor_targets = targets.clone()
    primary_weight = torch.ones(batch, dtype=features.dtype, device=features.device)
    if probability == 0.0 or batch < 2:
        return features, targets, donor_targets, primary_weight

    mixed = features.clone()
    donor_shift = int(torch.randint(1, batch, (), device=features.device))
    donor_index = torch.roll(
        torch.arange(batch, device=features.device), shifts=donor_shift
    )
    selected = torch.rand(batch, device=features.device) < probability
    right_hand = torch.rand(batch, device=features.device) < 0.5
    for sample in torch.where(selected)[0].tolist():
        start = 21 if bool(right_hand[sample]) else 0
        mixed[sample, :, start : start + 21] = features[
            donor_index[sample], :, start : start + 21
        ]
    donor_targets[selected] = targets[donor_index[selected]]
    primary_weight[selected] = 0.5
    return mixed, targets, donor_targets, primary_weight


def partmix_cross_entropy(
    logits: torch.Tensor,
    primary_targets: torch.Tensor,
    donor_targets: torch.Tensor,
    primary_weight: torch.Tensor,
    *,
    label_smoothing: float,
) -> torch.Tensor:
    """Cross entropy for ordinary samples and the two labels of PartMix samples."""
    primary_loss = F.cross_entropy(
        logits, primary_targets, reduction="none", label_smoothing=label_smoothing
    )
    donor_loss = F.cross_entropy(
        logits, donor_targets, reduction="none", label_smoothing=label_smoothing
    )
    return (
        primary_weight * primary_loss + (1.0 - primary_weight) * donor_loss
    ).mean()


def supervised_contrastive_loss(
    embeddings: torch.Tensor, targets: torch.Tensor, temperature: float = 0.10
) -> torch.Tensor:
    """Pull same-class clips together without treating anchors as positives."""
    if embeddings.ndim != 2 or targets.ndim != 1 or len(embeddings) != len(targets):
        raise ValueError("embeddings and targets must be aligned rank-2/rank-1 tensors")
    if temperature <= 0.0:
        raise ValueError("contrastive temperature must be positive")
    normalized = F.normalize(embeddings, dim=1)
    similarities = normalized @ normalized.T / temperature
    identity = torch.eye(len(targets), dtype=torch.bool, device=targets.device)
    positives = targets[:, None].eq(targets[None, :]) & ~identity
    valid = positives.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0.0
    similarities = similarities - similarities.max(dim=1, keepdim=True).values.detach()
    exp_similarities = torch.exp(similarities) * ~identity
    log_probabilities = similarities - torch.log(
        exp_similarities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )
    mean_positive = (log_probabilities * positives).sum(dim=1) / positives.sum(
        dim=1
    ).clamp_min(1)
    return -mean_positive[valid].mean()


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.updates = 0
        self.shadow = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        # Early checkpoints must not remain dominated by random initialization.
        # The decay ramps toward the requested ceiling as optimizer steps accrue.
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].lerp_(value.detach(), 1.0 - decay)
            else:
                self.shadow[key].copy_(value)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def evaluate(
    model: SLTStage1V17,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    loss_sum = correct = top5_correct = total = 0.0
    confusion = np.zeros(
        (model.config.num_classes, model.config.num_classes), dtype=np.int64
    )
    for features, targets in loader:
        targets_cpu = targets.numpy().astype(np.int64, copy=True)
        asynchronous = device.type == "cuda"
        features = features.to(device, non_blocking=asynchronous)
        targets = targets.to(device, non_blocking=asynchronous)
        with amp_context(device, use_amp):
            logits = model(features)
            loss = F.cross_entropy(logits, targets)
        predictions = logits.argmax(dim=1)
        top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
        batch_size = targets.numel()
        if device.type == "mps":
            # PyTorch 2.8 may otherwise expose an incompletely synchronized MPS
            # integer transfer to CPU during metric accumulation.
            torch.mps.synchronize()
        loss_sum += float(loss.detach().cpu()) * batch_size
        total += batch_size
        predictions_cpu = predictions.detach().cpu().numpy().astype(np.int64, copy=False)
        top5_cpu = top5.detach().cpu().numpy().astype(np.int64, copy=False)
        correct += int((predictions_cpu == targets_cpu).sum())
        top5_correct += int((top5_cpu == targets_cpu[:, None]).any(axis=1).sum())
        np.add.at(confusion, (targets_cpu, predictions_cpu), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "loss": loss_sum / max(total, 1),
        "top1": 100.0 * correct / max(total, 1),
        "top5": 100.0 * top5_correct / max(total, 1),
        "macro_f1": 100.0 * float(f1.mean()),
        "samples": float(total),
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    if (
        args.contrastive_weight < 0.0
        or args.contrastive_decay_epochs < 0
        or args.phonology_weight < 0.0
        or args.part_auxiliary_weight < 0.0
    ):
        raise ValueError("objective weights and contrastive decay epochs must be non-negative")
    if (args.phonology_weight > 0.0) != (args.phonology_targets is not None):
        raise ValueError(
            "phonology-targets and a positive phonology-weight must be supplied together"
        )
    if not 0.0 <= args.partmix_probability <= 1.0:
        raise ValueError("partmix-probability must be in [0, 1]")
    if not 0.0 <= args.full_roll_probability <= 1.0:
        raise ValueError("full-roll-probability must be in [0, 1]")
    if not 0.0 <= args.mild_roll_degrees <= args.maximum_roll_degrees <= 180.0:
        raise ValueError(
            "roll degrees must satisfy 0 <= mild-roll <= maximum-roll <= 180"
        )
    if args.partmix_probability > 0.0 and (
        args.contrastive_weight > 0.0
        or args.phonology_weight > 0.0
        or args.part_auxiliary_weight > 0.0
    ):
        raise ValueError(
            "PartMix is an isolated controlled study and cannot be combined with "
            "contrastive or phonology objectives"
        )
    if args.part_auxiliary_weight > 0.0 and args.temporal_encoder != "partwise_global":
        raise ValueError("part auxiliary supervision requires partwise_global")
    if args.part_auxiliary_weight > 0.0 and (
        args.contrastive_weight > 0.0 or args.phonology_weight > 0.0
    ):
        raise ValueError("part auxiliary supervision must be tested as an isolated objective")
    if args.keypoint_temporal_gate and args.temporal_encoder != "partwise_global":
        raise ValueError("keypoint temporal gating requires partwise_global")
    if args.keypoint_temporal_gate and (
        args.bone_features
        or args.hand_angle_features
        or args.partmix_probability > 0.0
        or args.contrastive_weight > 0.0
        or args.phonology_weight > 0.0
        or args.part_auxiliary_weight > 0.0
    ):
        raise ValueError(
            "keypoint temporal gating must first be tested as an isolated part-wise component"
        )
    if args.articulated_pose_pretrained is not None and not args.articulated_pose_embedding:
        raise ValueError(
            "--articulated-pose-pretrained requires --articulated-pose-embedding"
        )
    if args.articulated_pose_embedding and args.temporal_encoder != "partwise_global":
        raise ValueError("articulated pose embedding requires partwise_global")
    if args.articulated_pose_embedding and (
        args.bone_features
        or args.hand_angle_features
        or args.keypoint_temporal_gate
        or args.partmix_probability > 0.0
        or args.contrastive_weight > 0.0
        or args.phonology_weight > 0.0
        or args.part_auxiliary_weight > 0.0
    ):
        raise ValueError(
            "articulated pose embedding must first be tested as an isolated part-wise component"
        )
    if args.static_hand_token != "none" and args.temporal_encoder != "partwise_global":
        raise ValueError("static hand token requires partwise_global")
    if args.static_hand_token != "none" and (
        args.bone_features
        or args.hand_angle_features
        or args.keypoint_temporal_gate
        or args.articulated_pose_embedding
        or args.partmix_probability > 0.0
        or args.contrastive_weight > 0.0
        or args.phonology_weight > 0.0
        or args.part_auxiliary_weight > 0.0
    ):
        raise ValueError(
            "static hand token must first be tested as an isolated part-wise component"
        )
    if args.masked_pose_pretrained is not None and args.temporal_encoder != "partwise_global":
        raise ValueError("masked pose pretraining requires partwise_global")
    if args.masked_pose_pretrained is not None and (
        args.bone_features
        or args.hand_angle_features
        or args.keypoint_temporal_gate
        or args.articulated_pose_embedding
        or args.static_hand_token != "none"
        or args.partmix_probability > 0.0
        or args.contrastive_weight > 0.0
        or args.phonology_weight > 0.0
        or args.part_auxiliary_weight > 0.0
    ):
        raise ValueError(
            "masked pose initialization must first be tested on part-wise-only"
        )
    if args.attention_score_mixing and args.temporal_encoder != "partwise_global":
        raise ValueError("attention score mixing requires partwise_global")
    if args.attention_score_mixing and (
        args.bone_features
        or args.hand_angle_features
        or args.keypoint_temporal_gate
        or args.articulated_pose_embedding
        or args.static_hand_token != "none"
        or args.masked_pose_pretrained is not None
        or args.partmix_probability > 0.0
        or args.contrastive_weight > 0.0
        or args.phonology_weight > 0.0
        or args.part_auxiliary_weight > 0.0
    ):
        raise ValueError(
            "attention score mixing must first be tested on part-wise-only"
        )
    if args.freeze_warm_start_epochs < 0:
        raise ValueError("freeze-warm-start-epochs must be non-negative")
    if args.freeze_warm_start_epochs and args.initialize_from is None:
        raise ValueError("freeze-warm-start-epochs requires --initialize-from")
    if args.fine_tune_from is not None and (
        args.initialize_from is not None
        or args.articulated_pose_pretrained is not None
        or args.masked_pose_pretrained is not None
    ):
        raise ValueError(
            "--fine-tune-from cannot be combined with partial initialization"
        )
    if args.fine_tune_from is not None and args.freeze_warm_start_epochs:
        raise ValueError("exact fine-tuning does not use graph warm-start freezing")
    if args.citizen_top1_floor_correct < 0:
        raise ValueError("citizen-top1-floor-correct must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    LOG.info("device=%s amp=%s", device, use_amp)

    expected_schema = extractor_schema_fingerprint(args.extractor)
    train_dataset = Citizen100V17Dataset(
        args.data_root, "train", args.manifest, args.rejections,
        cache=not args.no_cache, expected_schema=expected_schema,
    )
    validation_dataset = Citizen100V17Dataset(
        args.data_root, "val", args.manifest, args.rejections,
        cache=not args.no_cache, expected_schema=expected_schema,
    )
    if train_dataset.label_to_index != validation_dataset.label_to_index:
        raise ValueError("train/validation class maps differ")
    phonology = (
        load_phonology_supervision(
            args.phonology_targets, Path(args.manifest), train_dataset.label_to_index
        )
        if args.phonology_targets is not None
        else None
    )

    has_supplement = args.supplement_root is not None or args.supplement_manifest is not None
    if has_supplement and (
        args.supplement_root is None or args.supplement_manifest is None
    ):
        raise ValueError("supplement-root and supplement-manifest must be supplied together")
    if has_supplement and not args.approve_supplement:
        raise ValueError("SemLex supplement requires explicit --approve-supplement")
    if has_supplement and args.evaluate_test:
        raise ValueError("Citizen test is sealed during supplemented model development")
    supplement_dataset: SemLexSupplementV17Dataset | None = None
    if has_supplement:
        supplement_dataset = SemLexSupplementV17Dataset(
            args.supplement_root,
            args.supplement_manifest,
            train_dataset.label_to_index,
            cache=not args.no_cache,
            expected_schema=expected_schema,
        )

    has_local = args.local_root is not None or args.local_manifest is not None
    if has_local and (args.local_root is None or args.local_manifest is None):
        raise ValueError("local-root and local-manifest must be supplied together")
    if has_local and not args.approve_local_supplement:
        raise ValueError("local supplement requires explicit --approve-local-supplement")
    if has_local and args.evaluate_test:
        raise ValueError("Citizen test is sealed during local-supplement development")
    local_dataset: LocalReviewSupplementV17Dataset | None = None
    if has_local:
        local_tiers = tuple(
            tier.strip() for tier in args.local_tiers.split(",") if tier.strip()
        )
        local_dataset = LocalReviewSupplementV17Dataset(
            args.local_root,
            args.local_manifest,
            train_dataset.label_to_index,
            allowed_tiers=local_tiers,
            cache=not args.no_cache,
            expected_schema=expected_schema,
            mask_mouth_nodes=args.mask_local_mouth_nodes,
        )

    has_local_validation = (
        args.local_validation_root is not None
        or args.local_validation_manifest is not None
    )
    if has_local_validation and (
        args.local_validation_root is None
        or args.local_validation_manifest is None
    ):
        raise ValueError(
            "local-validation-root and local-validation-manifest must be supplied together"
        )
    if has_local_validation and args.evaluate_test:
        raise ValueError("Citizen test is sealed during local validation development")
    local_validation_dataset: LocalValidationV17Dataset | None = None
    if has_local_validation:
        local_validation_dataset = LocalValidationV17Dataset(
            args.local_validation_root,
            args.local_validation_manifest,
            train_dataset.label_to_index,
            cache=not args.no_cache,
            expected_schema=expected_schema,
            mask_mouth_nodes=args.mask_local_mouth_nodes,
        )

    training_data_provenance: dict[str, object] = {
        "citizen_manifest": str(Path(args.manifest)),
        "citizen_manifest_sha256": sha256_file(Path(args.manifest)),
        "citizen_train_samples": len(train_dataset),
        "citizen_validation_samples": len(validation_dataset),
        "citizen_test_accessed": False,
        "online_augmentation": {
            "name": "augment_v17_random_per_batch",
            "camera_roll": "continuous_isotropic_xy_rotation",
            "full_roll_probability": args.full_roll_probability,
            "maximum_roll_degrees": args.maximum_roll_degrees,
            "mild_roll_degrees": args.mild_roll_degrees,
            "aspect_ratio_policy": (
                "extractor_isotropic_coordinates; no anisotropic feature distortion"
            ),
        },
        "architecture": {
            "spatial_encoder": args.spatial_encoder,
            "temporal_encoder": args.temporal_encoder,
            "part_depth": args.part_depth,
            "bone_features": args.bone_features,
            "hand_angle_features": args.hand_angle_features,
            "keypoint_temporal_gate": args.keypoint_temporal_gate,
            "articulated_pose_embedding": args.articulated_pose_embedding,
            "articulated_pose_initialization": (
                "distance_pretrained"
                if args.articulated_pose_pretrained is not None
                else "random_capacity_control"
                if args.articulated_pose_embedding
                else None
            ),
            "static_hand_token": args.static_hand_token,
            "masked_pose_initialization": args.masked_pose_pretrained is not None,
            "attention_score_mixing": args.attention_score_mixing,
            "canonicalize_camera_roll": args.canonicalize_camera_roll,
            "input_modality": args.input_modality,
        },
        "objective": {
            "classification": "cross_entropy",
            "label_smoothing": args.label_smoothing,
            "supervised_contrastive_weight": args.contrastive_weight,
            "supervised_contrastive_temperature": args.contrastive_temperature,
            "supervised_contrastive_decay_epochs": args.contrastive_decay_epochs,
            "phonology_weight": args.phonology_weight,
            "part_auxiliary_weight": args.part_auxiliary_weight,
            "part_auxiliary_heads": (
                ["left_hand", "right_hand", "face", "body"]
                if args.part_auxiliary_weight > 0.0
                else []
            ),
            "part_auxiliary_missing_policy": (
                "skip_clip_part_when_no_node_is_observed"
                if args.part_auxiliary_weight > 0.0
                else None
            ),
            "partmix": {
                "probability": args.partmix_probability,
                "parts": ["left_hand_21", "right_hand_21"],
                "selection": "uniform_one_hand_non_self_cyclic_donor",
                "label_weights": [0.5, 0.5],
            },
        },
    }
    if phonology is not None:
        training_data_provenance["objective"]["phonology"] = {
            "path": phonology["path"],
            "sha256": phonology["sha256"],
            "attributes": phonology["attributes"],
            "missing_target_index": -100,
        }
    if supplement_dataset is not None:
        training_data_provenance.update(
            {
                "semlex_manifest": str(Path(args.supplement_manifest)),
                "semlex_manifest_sha256": sha256_file(Path(args.supplement_manifest)),
                "semlex_train_samples": len(supplement_dataset),
                "semlex_train_only_approved": True,
                "semlex_test_accessed": False,
            }
        )
    if local_dataset is not None:
        training_data_provenance.update(
            {
                "local_manifest": str(Path(args.local_manifest)),
                "local_manifest_sha256": sha256_file(Path(args.local_manifest)),
                "local_train_samples": len(local_dataset),
                "local_approved_tiers": list(local_dataset.allowed_tiers),
                "local_train_only_user_approved": True,
                "local_signer_identity_known": bool(local_dataset.known_signers),
                "local_signers": list(local_dataset.known_signers),
                "local_manifest_format": local_dataset.manifest_format,
                "local_mouth_node_policy": (
                    "zero_only_mouth_left_mouth_right_upper_lip_lower_lip"
                    if args.mask_local_mouth_nodes
                    else "unchanged"
                ),
            }
        )
    if local_validation_dataset is not None:
        training_data_provenance.update(
            {
                "local_validation_manifest": str(
                    Path(args.local_validation_manifest)
                ),
                "local_validation_manifest_sha256": sha256_file(
                    Path(args.local_validation_manifest)
                ),
                "local_validation_samples": len(local_validation_dataset),
                "local_validation_signer_disjoint": False,
                "local_validation_signer_overlap_user_approved": True,
                "checkpoint_selection_metric": "citizen_official_validation_top1",
                "local_validation_role": "secondary_familiar_signer_domain_diagnostic",
                "local_validation_mouth_node_policy": (
                    "zero_only_mouth_left_mouth_right_upper_lip_lower_lip"
                    if args.mask_local_mouth_nodes
                    else "unchanged"
                ),
            }
        )

    model_dim, model_depth, epochs = args.dim, args.depth, args.epochs
    if args.smoke:
        train_data = train_dataset.balanced_subset(2)
        if supplement_dataset is not None:
            train_data = ConcatDataset((train_data, supplement_dataset))
        if local_dataset is not None:
            train_data = ConcatDataset((train_data, local_dataset))
        validation_data = validation_dataset.balanced_subset(1)
        model_dim, model_depth, epochs = 64, 1, 1
        args.batch_size = min(args.batch_size, 32)
        args.max_train_batches = 2
    else:
        train_parts: list[Dataset] = [train_dataset]
        if supplement_dataset is not None:
            train_parts.append(supplement_dataset)
        if local_dataset is not None:
            train_parts.append(local_dataset)
        train_data = ConcatDataset(tuple(train_parts)) if len(train_parts) > 1 else train_dataset
        validation_data = validation_dataset

    sampler = None
    if args.sampling == "class_source_balanced":
        sampling_targets, sampling_sources = dataset_targets_and_sources(train_data)
        sample_weights, sampling_summary = class_source_balanced_weights(
            sampling_targets,
            sampling_sources,
            train_dataset.num_classes,
            parse_source_probabilities(args.source_probabilities),
        )
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_data),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        training_data_provenance["sampling"] = sampling_summary
    else:
        training_data_provenance["sampling"] = {"mode": "shuffle_without_replacement"}

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    local_validation_loader = (
        DataLoader(
            local_validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        if local_validation_dataset is not None
        else None
    )
    config = Stage1V17Config(
        num_classes=train_dataset.num_classes,
        dim=model_dim,
        depth=model_depth,
        heads=args.heads if model_dim % args.heads == 0 else 4,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        drop_path=args.drop_path,
        use_pairwise=not args.no_pairwise,
        input_modality=args.input_modality,
        spatial_encoder=args.spatial_encoder,
        graph_node_dim=args.graph_node_dim,
        graph_layers=args.graph_layers,
        graph_heads=args.graph_heads,
        temporal_encoder=args.temporal_encoder,
        part_depth=args.part_depth,
        use_bone_features=args.bone_features,
        use_hand_angle_features=args.hand_angle_features,
        use_keypoint_temporal_gate=args.keypoint_temporal_gate,
        use_articulated_pose_embedding=args.articulated_pose_embedding,
        static_hand_token=args.static_hand_token,
        use_attention_score_mixing=args.attention_score_mixing,
        canonicalize_camera_roll=args.canonicalize_camera_roll,
        use_part_auxiliary=args.part_auxiliary_weight > 0.0,
        phonology_head_sizes=(phonology["head_sizes"] if phonology is not None else ()),
    )
    model = SLTStage1V17(config).to(device)
    initialization_info: dict[str, object] | None = None
    initial_metrics: dict[str, float] | None = None
    initial_local_validation_metrics: dict[str, float] | None = None
    if args.articulated_pose_pretrained is not None:
        if args.supplement_manifest is None:
            raise ValueError("articulated pose pretraining requires the supplement manifest")
        geometry_initialization = initialize_articulated_pose_embedding(
            model,
            Path(args.articulated_pose_pretrained),
            Path(args.manifest),
            Path(args.supplement_manifest),
            expected_schema,
        )
        training_data_provenance["articulated_pose_initialization"] = (
            geometry_initialization
        )
        LOG.info(
            "loaded articulated pose embedding=%s sha256=%s",
            args.articulated_pose_pretrained,
            geometry_initialization["sha256"],
        )
    if args.masked_pose_pretrained is not None:
        if args.supplement_manifest is None:
            raise ValueError("masked pose pretraining requires the supplement manifest")
        masked_initialization = initialize_masked_pose_encoder(
            model,
            Path(args.masked_pose_pretrained),
            Path(args.manifest),
            Path(args.supplement_manifest),
            expected_schema,
        )
        training_data_provenance["masked_pose_initialization"] = masked_initialization
        LOG.info(
            "loaded masked pose encoder=%s sha256=%s",
            args.masked_pose_pretrained,
            masked_initialization["sha256"],
        )
    if args.initialize_from is not None:
        initialization_info = initialize_flat_graph_residual(
            model,
            Path(args.initialize_from),
            Path(args.manifest),
            expected_schema,
        )
        # The residual gate is exactly zero, so this is also an executable check that
        # the challenger begins at the frozen flat model's measured performance.
        initial_metrics = evaluate(model, validation_loader, device, use_amp=use_amp)
        initialization_info["initial_validation_metrics"] = initial_metrics
        training_data_provenance["initialization"] = initialization_info
        LOG.info(
            "warm_start=%s initial_top1=%.2f residual_scale=%.6f",
            args.initialize_from,
            initial_metrics["top1"],
            float(model.graph_residual_scale.detach()),
        )
    if args.fine_tune_from is not None:
        initialization_info = initialize_exact_stage1_finetune(
            model,
            Path(args.fine_tune_from),
            Path(args.manifest),
            expected_schema,
            train_dataset.label_to_index,
        )
        initial_metrics = evaluate(model, validation_loader, device, use_amp=use_amp)
        initial_local_validation_metrics = (
            evaluate(model, local_validation_loader, device, use_amp=use_amp)
            if local_validation_loader is not None
            else None
        )
        initialization_info["initial_validation_metrics"] = initial_metrics
        initialization_info["initial_local_validation_metrics"] = (
            initial_local_validation_metrics
        )
        training_data_provenance["initialization"] = initialization_info
        training_data_provenance["optimization"] = {
            "mode": "balanced_replay_domain_adaptation",
            "checkpoint_selection": (
                "citizen_official_validation_top1_primary; "
                "local_validation_top1_only_breaks_exact_citizen_ties"
            ),
            "local_validation_used_for_selection": "exact_ties_only",
        }
        LOG.info(
            "exact_fine_tune=%s initial_top1=%.2f",
            args.fine_tune_from,
            initial_metrics["top1"],
        )
    if args.freeze_warm_start_epochs:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name == "graph_residual_scale" or name.startswith("graph_encoder.")
            )
        training_data_provenance["optimization"] = {
            "freeze_warm_start_epochs": args.freeze_warm_start_epochs,
            "frozen_during_warmup": "all pretrained flat/classifier parameters",
            "trainable_during_warmup": "graph_encoder and graph_residual_scale",
            "joint_finetune_after_warmup": True,
        }
    phonology_target_maps = (
        {
            name: targets.to(device)
            for name, targets in phonology["target_maps"].items()
        }
        if phonology is not None
        else {}
    )
    LOG.info(
        "train=%d val=%d classes=%d parameters=%d sampling=%s",
        len(train_data), len(validation_data), config.num_classes, model.parameter_count,
        args.sampling,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    warmup = max(1, min(args.warmup_epochs, epochs))

    def learning_rate(epoch_index: int) -> float:
        if epoch_index < warmup:
            return float(epoch_index + 1) / warmup
        progress = (epoch_index - warmup) / max(epochs - warmup, 1)
        return args.minimum_lr_ratio + (1.0 - args.minimum_lr_ratio) * 0.5 * (
            1.0 + np.cos(np.pi * progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ExponentialMovingAverage(model, args.ema_decay)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_data_provenance.json").write_text(
        json.dumps(training_data_provenance, indent=2) + "\n", encoding="utf-8"
    )
    history: list[dict[str, float]] = []
    best_top1 = initial_metrics["top1"] if initial_metrics is not None else -1.0
    best_local_validation_metrics: dict[str, float] | None = (
        initial_local_validation_metrics
    )
    best_gate_local_top1 = -1.0
    best_gate_citizen_correct = -1
    best_gate_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = (
        {key: value.detach().cpu().clone() for key, value in ema.shadow.items()}
        if initial_metrics is not None
        else None
    )
    stale_epochs = 0
    if initial_metrics is not None and best_state is not None:
        (output / "initialization_metrics.json").write_text(
            json.dumps(initial_metrics, indent=2) + "\n", encoding="utf-8"
        )
        checkpoint = make_stage1_checkpoint(
            model,
            best_state,
            epoch=0,
            validation_metrics=initial_metrics,
            label_to_index=train_dataset.label_to_index,
            manifest_sha256=sha256_file(Path(args.manifest)),
            schema_fingerprint=train_dataset.expected_schema,
        )
        checkpoint["training_data_provenance"] = training_data_provenance
        if initial_local_validation_metrics is not None:
            checkpoint["local_validation_metrics"] = (
                initial_local_validation_metrics
            )
        torch.save(checkpoint, output / "best_model.pth")
        initial_correct = int(round(
            initial_metrics["top1"] * initial_metrics["samples"] / 100.0
        ))
        if (
            args.citizen_top1_floor_correct
            and initial_correct >= args.citizen_top1_floor_correct
            and initial_local_validation_metrics is not None
        ):
            checkpoint["promotion_gate_selection"] = {
                "citizen_floor_correct": args.citizen_top1_floor_correct,
                "selection": "maximum_local_top1_subject_to_citizen_floor",
                "semlex_and_orientation_evaluation_required": True,
            }
            torch.save(checkpoint, output / "best_promotion_gate_model.pth")
            best_gate_local_top1 = initial_local_validation_metrics["top1"]
            best_gate_citizen_correct = initial_correct
            best_gate_epoch = 0

    for epoch in range(1, epochs + 1):
        if args.freeze_warm_start_epochs and epoch == args.freeze_warm_start_epochs + 1:
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            LOG.info("unfroze pretrained flat/classifier parameters at epoch=%d", epoch)
        model.train()
        contrastive_weight = args.contrastive_weight
        if args.contrastive_decay_epochs > 0:
            contrastive_weight *= max(
                0.0,
                1.0 - (epoch - 1) / float(args.contrastive_decay_epochs),
            )
        optimizer.zero_grad(set_to_none=True)
        total_loss = total_phonology_loss = total_part_auxiliary_loss = 0.0
        partmix_total = seen = 0.0
        started = time.monotonic()
        for batch_index, (features, targets) in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            asynchronous = device.type == "cuda"
            features = augment_v17(
                features.to(device, non_blocking=asynchronous),
                full_roll_probability=args.full_roll_probability,
                maximum_roll_degrees=args.maximum_roll_degrees,
                mild_roll_degrees=args.mild_roll_degrees,
            )
            targets = targets.to(device, non_blocking=asynchronous)
            features, targets, donor_targets, primary_weight = partmix_hands_v17(
                features, targets, args.partmix_probability
            )
            with amp_context(device, use_amp):
                if phonology is not None:
                    logits, embeddings, auxiliary_logits = model.forward_multitask(features)
                    phonology_loss = phonology_auxiliary_loss(
                        auxiliary_logits, phonology_target_maps, targets
                    )
                    part_loss = embeddings.sum() * 0.0
                elif args.part_auxiliary_weight > 0.0:
                    (
                        logits,
                        embeddings,
                        auxiliary_logits,
                        part_valid,
                    ) = model.forward_part_auxiliary(features)
                    part_loss = part_auxiliary_loss(
                        auxiliary_logits, targets, part_valid
                    )
                    phonology_loss = embeddings.sum() * 0.0
                else:
                    logits, embeddings = model(features, return_embeddings=True)
                    phonology_loss = embeddings.sum() * 0.0
                    part_loss = embeddings.sum() * 0.0
                classification_loss = partmix_cross_entropy(
                    logits,
                    targets,
                    donor_targets,
                    primary_weight,
                    label_smoothing=args.label_smoothing,
                )
                contrastive_loss = supervised_contrastive_loss(
                    embeddings, targets, args.contrastive_temperature
                )
                loss = (
                    classification_loss
                    + contrastive_weight * contrastive_loss
                    + args.phonology_weight * phonology_loss
                    + args.part_auxiliary_weight * part_loss
                ) / args.accumulation_steps
            scaler.scale(loss).backward()
            should_step = (
                (batch_index + 1) % args.accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            batch_size = targets.numel()
            total_loss += float(loss.detach()) * args.accumulation_steps * batch_size
            total_phonology_loss += float(phonology_loss.detach()) * batch_size
            total_part_auxiliary_loss += float(part_loss.detach()) * batch_size
            partmix_total += float((primary_weight < 1.0).sum().detach())
            seen += batch_size

        live_state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        model.load_state_dict(ema.shadow)
        metrics = evaluate(model, validation_loader, device, use_amp=use_amp)
        local_validation_metrics = (
            evaluate(model, local_validation_loader, device, use_amp=use_amp)
            if local_validation_loader is not None
            else None
        )
        model.load_state_dict(live_state)
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "train_loss": total_loss / max(seen, 1),
            **metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.monotonic() - started,
            "contrastive_weight": contrastive_weight,
            "phonology_weight": args.phonology_weight,
            "train_phonology_loss": total_phonology_loss / max(seen, 1),
            "train_part_auxiliary_loss": total_part_auxiliary_loss / max(seen, 1),
            "partmix_fraction": partmix_total / max(seen, 1),
        }
        if local_validation_metrics is not None:
            row.update(
                {
                    f"local_val_{name}": value
                    for name, value in local_validation_metrics.items()
                }
            )
        history.append(row)
        LOG.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f top1=%.2f top5=%.2f macro_f1=%.2f seconds=%.1f",
            epoch, row["train_loss"], row["loss"], row["top1"], row["top5"],
            row["macro_f1"], row["seconds"],
        )
        if local_validation_metrics is not None:
            LOG.info(
                "epoch=%d local_val_loss=%.4f local_top1=%.2f local_top5=%.2f local_macro_f1=%.2f",
                epoch,
                local_validation_metrics["loss"],
                local_validation_metrics["top1"],
                local_validation_metrics["top5"],
                local_validation_metrics["macro_f1"],
            )
        citizen_improved = metrics["top1"] > best_top1
        citizen_tied_local_improved = (
            metrics["top1"] == best_top1
            and local_validation_metrics is not None
            and (
                best_local_validation_metrics is None
                or local_validation_metrics["top1"]
                > best_local_validation_metrics["top1"]
            )
        )
        if citizen_improved or citizen_tied_local_improved:
            best_top1 = metrics["top1"]
            best_local_validation_metrics = local_validation_metrics
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in ema.shadow.items()}
            checkpoint = make_stage1_checkpoint(
                model,
                best_state,
                epoch=epoch,
                validation_metrics=metrics,
                label_to_index=train_dataset.label_to_index,
                manifest_sha256=sha256_file(Path(args.manifest)),
                schema_fingerprint=train_dataset.expected_schema,
            )
            checkpoint["training_data_provenance"] = training_data_provenance
            if local_validation_metrics is not None:
                checkpoint["local_validation_metrics"] = local_validation_metrics
            temporary = output / "best_model.pth.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "best_model.pth")
        else:
            stale_epochs += 1
        citizen_correct = int(round(metrics["top1"] * metrics["samples"] / 100.0))
        gate_improved = (
            args.citizen_top1_floor_correct
            and citizen_correct >= args.citizen_top1_floor_correct
            and local_validation_metrics is not None
            and (
                local_validation_metrics["top1"] > best_gate_local_top1
                or (
                    local_validation_metrics["top1"] == best_gate_local_top1
                    and citizen_correct > best_gate_citizen_correct
                )
            )
        )
        if gate_improved:
            gate_state = {
                key: value.detach().cpu().clone()
                for key, value in ema.shadow.items()
            }
            gate_checkpoint = make_stage1_checkpoint(
                model,
                gate_state,
                epoch=epoch,
                validation_metrics=metrics,
                label_to_index=train_dataset.label_to_index,
                manifest_sha256=sha256_file(Path(args.manifest)),
                schema_fingerprint=train_dataset.expected_schema,
            )
            gate_checkpoint["training_data_provenance"] = training_data_provenance
            gate_checkpoint["local_validation_metrics"] = local_validation_metrics
            gate_checkpoint["promotion_gate_selection"] = {
                "citizen_floor_correct": args.citizen_top1_floor_correct,
                "selection": "maximum_local_top1_subject_to_citizen_floor",
                "semlex_and_orientation_evaluation_required": True,
            }
            temporary = output / "best_promotion_gate_model.pth.tmp"
            torch.save(gate_checkpoint, temporary)
            temporary.replace(output / "best_promotion_gate_model.pth")
            best_gate_local_top1 = local_validation_metrics["top1"]
            best_gate_citizen_correct = citizen_correct
            best_gate_epoch = epoch
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stale_epochs >= args.patience:
            LOG.info("early stopping after %d stale epochs", stale_epochs)
            break

    result: dict[str, object] = {
        "best_validation_top1": best_top1,
        "epochs_completed": len(history),
        "parameters": model.parameter_count,
        "device": str(device),
        "extractor": args.extractor,
        "schema_fingerprint": expected_schema,
        "training_data_provenance": training_data_provenance,
        "test_evaluated": False,
    }
    if best_local_validation_metrics is not None:
        result["local_validation_at_selected_checkpoint"] = (
            best_local_validation_metrics
        )
    if best_gate_epoch is not None:
        result["promotion_gate_candidate"] = {
            "epoch": best_gate_epoch,
            "citizen_correct": best_gate_citizen_correct,
            "citizen_floor_correct": args.citizen_top1_floor_correct,
            "local_validation_top1": best_gate_local_top1,
            "requires_semlex_and_orientation_evaluation": True,
        }
    if args.evaluate_test:
        if best_state is None:
            raise RuntimeError("no best checkpoint was produced")
        model.load_state_dict(best_state)
        test_dataset = Citizen100V17Dataset(
            args.data_root, "test", args.manifest, args.rejections,
            cache=not args.no_cache, expected_schema=expected_schema,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=device.type == "cuda",
        )
        result["test"] = evaluate(model, test_loader, device, use_amp=use_amp)
        result["test_evaluated"] = True
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/local/citizen100_v17/landmarks"))
    parser.add_argument("--extractor", choices=EXTRACTORS, default="apple")
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument("--rejections", type=Path, default=Path("data/local/citizen100_v17/rejections.csv"))
    parser.add_argument(
        "--supplement-root",
        type=Path,
        help="Train-only reviewed SemLex landmark root arranged as LABEL/*.v17.npz",
    )
    parser.add_argument(
        "--supplement-manifest",
        type=Path,
        help="Reviewed SemLex train-only selection manifest",
    )
    parser.add_argument(
        "--approve-supplement",
        action="store_true",
        help="Explicitly approve the reviewed train-only supplement for this run",
    )
    parser.add_argument(
        "--local-root", type=Path,
        help="Train-only local landmark root arranged as LABEL/*.v17.npz",
    )
    parser.add_argument(
        "--local-manifest", type=Path,
        help="Model-screened local review manifest",
    )
    parser.add_argument(
        "--local-tiers", default="tier_a_dual_top1",
        help="Comma-separated local consensus tiers explicitly approved for training",
    )
    parser.add_argument(
        "--approve-local-supplement", action="store_true",
        help="Explicitly approve the selected train-only local tiers for this experiment",
    )
    parser.add_argument(
        "--local-validation-root", type=Path,
        help="Secondary local validation landmark root arranged as LABEL/*.v17.npz",
    )
    parser.add_argument(
        "--local-validation-manifest", type=Path,
        help="Non-signer-disjoint local validation manifest; never used as Citizen test",
    )
    parser.add_argument(
        "--mask-local-mouth-nodes",
        action="store_true",
        help=(
            "Zero only the four mouth/lip nodes for local train/validation rows; "
            "eyes, brows, nose, jaw, chin, Citizen, and SemLex remain unchanged"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/stage1_v17"))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--sampling",
        choices=("shuffle", "class_source_balanced"),
        default="shuffle",
        help="Training sampler; balanced mode equalizes expected class/source exposure",
    )
    parser.add_argument(
        "--source-probabilities",
        help="Optional balanced-sampler source margins, e.g. citizen=.45,semlex=.45,local=.10",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.02)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--label-smoothing", type=float, default=0.10)
    parser.add_argument(
        "--full-roll-probability", type=float, default=0.35,
        help="Per-sample probability of arbitrary camera-roll augmentation",
    )
    parser.add_argument(
        "--maximum-roll-degrees", type=float, default=180.0,
        help="Maximum absolute arbitrary camera roll; 180 covers the full circle",
    )
    parser.add_argument(
        "--mild-roll-degrees", type=float, default=12.0,
        help="Maximum absolute roll for samples outside the full-roll branch",
    )
    parser.add_argument(
        "--canonicalize-camera-roll", action="store_true",
        help="Align every clip by its shoulder axis with an eye-line fallback",
    )
    parser.add_argument(
        "--partmix-probability", type=float, default=0.0,
        help="Probability of replacing one complete hand with a non-self batch donor",
    )
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.10)
    parser.add_argument(
        "--contrastive-decay-epochs", type=int, default=0,
        help="Linearly decay contrastive weight to zero over this many epochs",
    )
    parser.add_argument(
        "--phonology-targets", type=Path,
        help="Manifest-locked ASL-LEX phonological target JSON",
    )
    parser.add_argument(
        "--phonology-weight", type=float, default=0.0,
        help="Weight for the mean ASL-LEX phonological auxiliary loss",
    )
    parser.add_argument(
        "--part-auxiliary-weight", type=float, default=0.0,
        help="Weight for training-only per-part gloss classifiers",
    )
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--citizen-top1-floor-correct",
        type=int,
        default=0,
        help=(
            "Also retain the local-best replay checkpoint meeting this exact "
            "Citizen validation correct-count floor; zero disables"
        ),
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--head-dropout", type=float, default=0.25)
    parser.add_argument("--drop-path", type=float, default=0.08)
    parser.add_argument("--no-pairwise", action="store_true")
    parser.add_argument(
        "--spatial-encoder",
        choices=("flat", "graph_parts", "flat_graph_residual"),
        default="flat",
        help="Flat projection, graph replacement, or zero-gated flat+graph residual",
    )
    parser.add_argument(
        "--initialize-from", type=Path,
        help="Strict flat checkpoint warm start for flat_graph_residual",
    )
    parser.add_argument(
        "--fine-tune-from", type=Path,
        help=(
            "Strict exact-checkpoint initialization for balanced replay domain "
            "adaptation; architecture and label mapping must be identical"
        ),
    )
    parser.add_argument(
        "--freeze-warm-start-epochs", type=int, default=0,
        help="Train only the new graph/gate before joint warm-start fine-tuning",
    )
    parser.add_argument("--graph-node-dim", type=int, default=64)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-heads", type=int, default=4)
    parser.add_argument(
        "--temporal-encoder",
        choices=("global", "partwise_global"),
        default="global",
        help="Global Squeezeformer or isolated hand/face/body encoders before global fusion",
    )
    parser.add_argument(
        "--part-depth", type=int, default=1,
        help="Squeezeformer depth for each isolated anatomical stream",
    )
    parser.add_argument(
        "--bone-features", action="store_true",
        help="Derive directed hand/arm bone vectors and bone motion inside the model",
    )
    parser.add_argument(
        "--hand-angle-features", action="store_true",
        help="Derive missing-aware cosine flexion angles at internal finger joints",
    )
    parser.add_argument(
        "--keypoint-temporal-gate", action="store_true",
        help="Learn identity-initialized per-keypoint temporal reliability gates",
    )
    parser.add_argument(
        "--articulated-pose-embedding", action="store_true",
        help="Fuse a 64-D wrist-relative hand-geometry embedding per frame",
    )
    parser.add_argument(
        "--articulated-pose-pretrained", type=Path,
        help="Strict self-supervised initialization for the geometry embedding only",
    )
    parser.add_argument(
        "--static-hand-token",
        choices=("none", "quality", "low_motion"),
        default="none",
        help="Fuse three quality-only or reliable low-motion hand frames",
    )
    parser.add_argument(
        "--masked-pose-pretrained", type=Path,
        help="Strict multi-stream masked-pose initialization for the part-wise encoder",
    )
    parser.add_argument(
        "--attention-score-mixing", action="store_true",
        help="Apply one zero-initialized depthwise 3x3 residual to each attention score map",
    )
    parser.add_argument(
        "--input-modality", choices=("all", "hands", "face", "mouth"), default="all",
        help="Restrict model-visible nodes while preserving the fixed v17 input schema",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--max-train-batches", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--evaluate-test", action="store_true",
        help="Evaluate the untouched Citizen test split only for a final selected run",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    result = train(build_parser().parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
