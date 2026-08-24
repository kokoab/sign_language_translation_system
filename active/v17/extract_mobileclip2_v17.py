#!/usr/bin/env python3
"""Extract frozen MobileCLIP2-S0 RGB sequences for Citizen train/validation."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.schema_mobileclip2_v17 import (
        CHECKPOINT_SHA256,
        EMBEDDING_DIM,
        MODEL_NAME,
        PRETRAINED_TAG,
        MobileCLIP2V17Config,
        schema_fingerprint,
        schema_payload,
    )
else:
    from .schema_mobileclip2_v17 import (
        CHECKPOINT_SHA256,
        EMBEDDING_DIM,
        MODEL_NAME,
        PRETRAINED_TAG,
        MobileCLIP2V17Config,
        schema_fingerprint,
        schema_payload,
    )


LOG = logging.getLogger("mobileclip2_v17")
ALLOWED_SPLITS = ("train", "val")


class ImageEncoderOnly(nn.Module):
    """Retain only OpenCLIP's visual tower for image-feature extraction."""

    def __init__(self, visual: nn.Module):
        super().__init__()
        self.visual = visual

    def encode_image(self, image: torch.Tensor, normalize: bool = False):
        features = self.visual(image)
        return (
            torch.nn.functional.normalize(features, dim=-1)
            if normalize
            else features
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_rejected(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["split"], row["canonical_label"], row["video"])
            for row in csv.DictReader(handle)
        }


def reference_indices(frame_count: int, maximum_frames: int = 96) -> np.ndarray:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if frame_count <= maximum_frames:
        return np.arange(frame_count, dtype=np.int64)
    return np.rint(np.linspace(0, frame_count - 1, maximum_frames)).astype(np.int64)


def temporal_sample_indices(
    frame_count: int,
    trim_start: int,
    trim_end_exclusive: int,
    sequence_length: int,
    maximum_reference_frames: int = 96,
) -> np.ndarray:
    """Map Apple-v17 sampled-frame trim positions back to raw video indices."""
    reference = reference_indices(frame_count, maximum_reference_frames)
    start = max(0, min(int(trim_start), len(reference) - 1))
    end = max(start + 1, min(int(trim_end_exclusive), len(reference)))
    active = reference[start:end]
    positions = np.rint(np.linspace(0, len(active) - 1, sequence_length)).astype(int)
    return active[positions]


def landmark_reference_indices(
    metadata: dict[str, object], maximum_reference_frames: int = 96
) -> np.ndarray:
    """Reconstruct the exact raw-frame sample used by ``read_video_frames``.

    Some legacy MP4/WebM containers over-report their frame count.  The landmark
    extractor records both the reported and actually decoded counts, so RGB
    extractors must reproduce its sampling decision instead of assuming the
    container's reported count was accurate.
    """
    reported = int(metadata["reported_frame_count"])
    decoded = int(metadata["decoded_frame_count"])
    sampled = int(metadata["source_frames_before_hand_trim"])
    if decoded < 1:
        raise ValueError("landmark metadata has no decoded frames")
    if reported <= 0 or reported <= maximum_reference_frames:
        rng = np.random.default_rng(17)
        selected: list[int] = []
        for raw_index in range(decoded):
            if len(selected) < maximum_reference_frames:
                selected.append(raw_index)
            else:
                candidate = int(rng.integers(0, raw_index + 1))
                if candidate < maximum_reference_frames:
                    selected[candidate] = raw_index
        reference = np.asarray(sorted(selected), dtype=np.int64)
    else:
        requested = set(
            np.rint(
                np.linspace(0, reported - 1, maximum_reference_frames)
            ).astype(int).tolist()
        )
        reference = np.asarray(
            sorted(index for index in requested if index < decoded), dtype=np.int64
        )
    if len(reference) != sampled:
        raise ValueError(
            "landmark sampling provenance mismatch "
            f"({len(reference)} reconstructed != {sampled} recorded)"
        )
    return reference


def temporal_sample_from_reference(
    reference: np.ndarray,
    trim_start: int,
    trim_end_exclusive: int,
    sequence_length: int,
) -> np.ndarray:
    if reference.ndim != 1 or len(reference) < 1:
        raise ValueError("reference frame indices must be a non-empty vector")
    start = max(0, min(int(trim_start), len(reference) - 1))
    end = max(start + 1, min(int(trim_end_exclusive), len(reference)))
    active = reference[start:end]
    positions = np.rint(np.linspace(0, len(active) - 1, sequence_length)).astype(int)
    return active[positions]


def letterbox_rgb(frame_bgr: np.ndarray, size: int = 256) -> np.ndarray:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"expected BGR image, got {frame_bgr.shape}")
    height, width = frame_bgr.shape[:2]
    if min(height, width) < 1:
        raise ValueError("empty image")
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x0 = (size - resized_width) // 2
    y0 = (size - resized_height) // 2
    canvas[y0:y0 + resized_height, x0:x0 + resized_width] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def decode_selected_frames(
    video_path: Path, selected_indices: np.ndarray
) -> tuple[list[np.ndarray], dict[str, object]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    reported_rotation = int(round(capture.get(cv2.CAP_PROP_ORIENTATION_META)))
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    reported_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    wanted = set(int(value) for value in selected_indices)
    decoded: dict[int, np.ndarray] = {}
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in wanted:
            decoded[frame_index] = frame
            wanted.remove(frame_index)
        frame_index += 1
    capture.release()
    if wanted:
        raise RuntimeError(f"{video_path}: missing selected frames {sorted(wanted)}")
    frames = [decoded[int(index)] for index in selected_indices]
    height, width = frames[0].shape[:2]
    return frames, {
        "reported_frame_count": reported_count,
        "decoded_frame_count": frame_index,
        "reported_rotation_degrees": reported_rotation,
        "orientation_mode": "opencv_metadata_auto",
        "oriented_width": width,
        "oriented_height": height,
        "orientation": "square" if width == height else "portrait" if height > width else "landscape",
        "selected_raw_frame_indices": selected_indices.tolist(),
    }


def load_landmark_trim(path: Path) -> tuple[int, int, int]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
    return (
        int(metadata["hand_trim_start_frame"]),
        int(metadata["hand_trim_end_frame_exclusive"]),
        int(metadata["source_frames_before_hand_trim"]),
    )


def load_landmark_sampling_contract(
    path: Path, maximum_reference_frames: int = 96
) -> tuple[int, int, np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
    reference = landmark_reference_indices(metadata, maximum_reference_frames)
    return (
        int(metadata["hand_trim_start_frame"]),
        int(metadata["hand_trim_end_frame_exclusive"]),
        reference,
        metadata,
    )


def validate_decoded_video_contract(
    video_path: Path,
    video_metadata: dict[str, object],
    landmark_metadata: dict[str, object],
) -> None:
    for key in ("reported_frame_count", "decoded_frame_count"):
        if int(video_metadata[key]) != int(landmark_metadata[key]):
            raise ValueError(
                f"{video_path}: {key} changed since landmark extraction "
                f"({video_metadata[key]} != {landmark_metadata[key]})"
            )
    if (
        int(video_metadata["reported_rotation_degrees"])
        != int(landmark_metadata["reported_rotation_degrees"])
    ):
        raise ValueError(f"{video_path}: orientation metadata changed")


def build_encoder(device: torch.device, precision: str = "fp32"):
    try:
        import open_clip
        from open_clip.model import CLIPVisionCfg, _build_vision_tower
        from open_clip.pretrained import download_pretrained
        from open_clip.transform import PreprocessCfg, image_transform_v2
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError(
            "OpenCLIP is required; use artifacts/generated/mobileclip2_env/bin/python"
        ) from error
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"unsupported encoder precision: {precision}")

    model_config = open_clip.get_model_config(MODEL_NAME)
    pretrained_config = open_clip.get_pretrained_cfg(MODEL_NAME, PRETRAINED_TAG)
    checkpoint_path = Path(download_pretrained(pretrained_config))
    if file_sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError(f"MobileCLIP2 checkpoint hash mismatch: {checkpoint_path}")

    # This extractor never uses text.  Construct the 11.4M-parameter visual tower
    # directly on the meta device and materialize only ``visual.*`` tensors from the
    # official checkpoint.  The former OpenCLIP factory path briefly allocated both
    # towers plus random initial weights and caused long MPS runs to be terminated
    # under memory pressure even after the unused text tower was deleted.
    with torch.device("meta"):
        visual = _build_vision_tower(
            int(model_config["embed_dim"]),
            CLIPVisionCfg(**model_config["vision_cfg"]),
        )
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        visual_state = {
            key.removeprefix("visual."): checkpoint.get_tensor(key)
            for key in checkpoint.keys()
            if key.startswith("visual.")
        }
    incompatible = visual.load_state_dict(visual_state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "MobileCLIP2 visual checkpoint mismatch: "
            f"missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    target_dtype = torch.float16 if precision == "fp16" else torch.float32
    image_encoder = ImageEncoderOnly(
        visual.to(device=device, dtype=target_dtype)
    ).eval()

    preprocess_config = PreprocessCfg(
        size=int(model_config["vision_cfg"]["image_size"]),
        mode="RGB",
        mean=tuple(pretrained_config["mean"]),
        std=tuple(pretrained_config["std"]),
        interpolation=str(pretrained_config["interpolation"]),
        resize_mode=str(pretrained_config["resize_mode"]),
        fill_color=0,
    )
    preprocess = image_transform_v2(preprocess_config, is_train=False)
    del visual_state
    gc.collect()
    return image_encoder, preprocess


def extract_clip(
    video_path: Path,
    landmark_path: Path,
    model,
    preprocess,
    device: torch.device,
    config: MobileCLIP2V17Config,
) -> tuple[np.ndarray, dict[str, object]]:
    trim_start, trim_end, reference, landmark_metadata = load_landmark_sampling_contract(
        landmark_path, config.maximum_reference_frames
    )
    selected = temporal_sample_from_reference(
        reference, trim_start, trim_end, config.sequence_length,
    )
    frames, video_metadata = decode_selected_frames(video_path, selected)
    validate_decoded_video_contract(video_path, video_metadata, landmark_metadata)
    tensors = torch.stack(
        [preprocess(Image.fromarray(letterbox_rgb(frame, config.input_size))) for frame in frames]
    ).to(device)
    with torch.inference_mode():
        embeddings = model.encode_image(tensors, normalize=config.normalize_embeddings)
    if device.type == "mps":
        torch.mps.synchronize()
    value = embeddings.float().cpu().numpy()
    if value.shape != (config.sequence_length, EMBEDDING_DIM):
        raise RuntimeError(f"unexpected embedding shape {value.shape}")
    if not np.isfinite(value).all():
        raise RuntimeError("non-finite MobileCLIP2 embedding")
    metadata = {
        **video_metadata,
        "schema_fingerprint": schema_fingerprint(config),
        "model_name": MODEL_NAME,
        "pretrained_tag": PRETRAINED_TAG,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "landmark_trim_source": str(landmark_path),
        "hand_trim_start_frame": trim_start,
        "hand_trim_end_frame_exclusive": trim_end,
        "embedding_norm_mean": float(np.linalg.norm(value, axis=1).mean()),
    }
    return value.astype(np.float16), metadata


def save_archive(
    path: Path,
    embeddings: np.ndarray,
    metadata: dict[str, object],
    config: MobileCLIP2V17Config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        embeddings=embeddings,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        schema_json=np.array(json.dumps(schema_payload(config), sort_keys=True)),
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.split not in ALLOWED_SPLITS:
        raise ValueError("the Citizen test split is sealed")
    config = MobileCLIP2V17Config(sequence_length=args.frames)
    config.validate()
    device = select_device(args.device)
    rejected = load_rejected(args.rejections)
    videos = []
    for path in sorted((args.raw_root / args.split).glob("*/*.mp4")):
        label = path.parent.name
        if (args.split, label, path.name) not in rejected:
            videos.append(path)
    if args.limit:
        videos = videos[:args.limit]
    model, preprocess = build_encoder(device)
    expected_schema = schema_fingerprint(config)
    started = time.monotonic()
    written = skipped = 0
    for index, video_path in enumerate(videos, start=1):
        relative = video_path.relative_to(args.raw_root / args.split)
        landmark_path = (
            args.landmark_root / args.split / relative.parent /
            f"{video_path.stem}.v17.npz"
        )
        if not landmark_path.exists():
            raise FileNotFoundError(landmark_path)
        output_path = (
            args.output_root / args.split / relative.parent /
            f"{video_path.stem}.mobileclip2_v17.npz"
        )
        if output_path.exists() and not args.overwrite:
            with np.load(output_path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
                shape = tuple(payload["embeddings"].shape)
            if metadata.get("schema_fingerprint") != expected_schema:
                raise ValueError(f"{output_path}: existing schema mismatch")
            if shape != (config.sequence_length, config.embedding_dim):
                raise ValueError(f"{output_path}: existing shape mismatch")
            skipped += 1
            continue
        embeddings, metadata = extract_clip(
            video_path, landmark_path, model, preprocess, device, config
        )
        metadata["video_path"] = str(video_path)
        save_archive(output_path, embeddings, metadata, config)
        written += 1
        if index == 1 or index % 25 == 0 or index == len(videos):
            LOG.info(
                "%s %d/%d written=%d skipped=%d elapsed=%.1fs",
                args.split, index, len(videos), written, skipped,
                time.monotonic() - started,
            )
    return {
        "split": args.split,
        "clips": len(videos),
        "written": written,
        "skipped": skipped,
        "device": str(device),
        "schema_fingerprint": expected_schema,
        "seconds": time.monotonic() - started,
        "test_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=ALLOWED_SPLITS)
    parser.add_argument("--raw-root", type=Path, default=Path("data/local/citizen100_v17/raw"))
    parser.add_argument("--landmark-root", type=Path, default=Path("data/local/citizen100_v17/landmarks"))
    parser.add_argument("--output-root", type=Path, default=Path("data/local/citizen100_v17/mobileclip2_s0"))
    parser.add_argument("--rejections", type=Path, default=Path("data/local/citizen100_v17/rejections.csv"))
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
