"""Strict data contract for the sign-specialized MoViNet v17 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


FRAMES = 16
VIEWS = 3
VIEW_NAMES = ("left", "right", "union")


@dataclass(frozen=True)
class MoViNetRecord:
    crop_path: Path
    item_id: str
    target: int
    landmark_feature: np.ndarray
    base_logits: np.ndarray


def load_aligned_records(
    crop_root: Path,
    feature_cache: Path,
    split: str,
) -> list[MoViNetRecord]:
    """Join RGB crops to frozen Apple features without relying on directory order."""
    if split not in ("train", "val"):
        raise ValueError("the official Citizen test split is unavailable to MoViNet")
    with np.load(feature_cache, allow_pickle=False) as payload:
        features = payload["features"].astype(np.float32)
        logits = payload["logits"].astype(np.float32)
        targets = payload["targets"].astype(np.int64)
        item_ids = payload["item_ids"].astype(str)
        cached_split = str(payload["split"])
        mode = str(payload["mode"])
    if cached_split != split or mode != "landmark":
        raise ValueError("Apple feature cache split/mode mismatch")
    if features.shape != (len(item_ids), 256) or logits.shape != (len(item_ids), 100):
        raise ValueError("unexpected Apple feature-cache shape")
    if not np.isfinite(features).all() or not np.isfinite(logits).all():
        raise ValueError("Apple feature cache contains non-finite values")
    if len(set(item_ids.tolist())) != len(item_ids):
        raise ValueError("duplicate feature-cache item IDs")

    records: list[MoViNetRecord] = []
    for index, item_id in enumerate(item_ids):
        label, stem = item_id.split("/", 1)
        crop_path = crop_root / split / label / f"{stem}.hand_rgb_v17.npz"
        if not crop_path.is_file():
            raise FileNotFoundError(f"missing aligned RGB crop: {crop_path}")
        records.append(
            MoViNetRecord(
                crop_path=crop_path,
                item_id=item_id,
                target=int(targets[index]),
                landmark_feature=features[index],
                base_logits=logits[index],
            )
        )
    return records


def decode_crop_archive(path: Path, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode real JPEG crops; invalid views remain exact zero and explicitly masked."""
    with np.load(path, allow_pickle=False) as payload:
        blob = payload["jpeg_blob"]
        offsets = payload["jpeg_offsets"]
        valid = payload["valid"].astype(np.bool_)
        boxes = payload["boxes_normalized"].astype(np.float32)
    if offsets.shape != (FRAMES, VIEWS, 2) or valid.shape != (FRAMES, VIEWS):
        raise ValueError(f"{path}: invalid crop archive shape")
    if boxes.shape != (FRAMES, VIEWS, 4) or not np.isfinite(boxes).all():
        raise ValueError(f"{path}: invalid box trajectory")

    pixels = np.zeros((FRAMES, VIEWS, resolution, resolution, 3), dtype=np.uint8)
    for frame_index, view_index in np.argwhere(valid):
        start, length = offsets[frame_index, view_index]
        if start < 0 or length <= 0:
            raise ValueError(f"{path}: valid view has no JPEG payload")
        decoded = cv2.imdecode(blob[start:start + length], cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"{path}: JPEG decode failed")
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        interpolation = cv2.INTER_AREA if decoded.shape[0] > resolution else cv2.INTER_CUBIC
        pixels[frame_index, view_index] = cv2.resize(
            decoded, (resolution, resolution), interpolation=interpolation
        )
    if np.any(pixels[~valid] != 0):
        raise ValueError(f"{path}: invalid crop contains pixels")
    boxes = boxes.copy()
    boxes[~valid] = 0
    return pixels, valid, boxes


def mirror_sign_views(
    pixels: np.ndarray,
    valid: np.ndarray,
    boxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror pixels and exchange anatomical left/right views and box trajectories."""
    pixels = np.flip(pixels, axis=3)[:, [1, 0, 2]].copy()
    valid = valid[:, [1, 0, 2]].copy()
    boxes = boxes[:, [1, 0, 2]].copy()
    old_x0 = boxes[..., 0].copy()
    boxes[..., 0] = 1.0 - boxes[..., 2]
    boxes[..., 2] = 1.0 - old_x0
    boxes[~valid] = 0
    pixels[~valid] = 0
    return pixels, valid, boxes


def augment_sign_views(
    pixels: np.ndarray,
    valid: np.ndarray,
    boxes: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply sign-safe temporal, mirror, photometric, and missing-view augmentation."""
    if rng.random() < 0.5:
        pixels, valid, boxes = mirror_sign_views(pixels, valid, boxes)

    if rng.random() < 0.65:
        base = np.linspace(0.0, 1.0, FRAMES)
        rate = rng.uniform(0.84, 1.16)
        offset = rng.uniform(-0.04, 0.04)
        indices = np.rint(np.clip((base - 0.5) * rate + 0.5 + offset, 0, 1) * (FRAMES - 1))
        indices = indices.astype(np.int64)
        pixels, valid, boxes = pixels[indices], valid[indices], boxes[indices]

    gain = rng.uniform(0.88, 1.12, size=(1, 1, 1, 1, 3))
    bias = rng.uniform(-8.0, 8.0, size=(1, 1, 1, 1, 3))
    pixels = np.clip(pixels.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

    drop = (rng.random(valid.shape) < 0.04) & valid
    valid = valid & ~drop
    pixels[~valid] = 0
    boxes[~valid] = 0
    return pixels, valid, boxes
