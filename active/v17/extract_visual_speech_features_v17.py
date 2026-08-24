#!/usr/bin/env python3
"""Cache frozen Auto-AVSR frontend features for train/validation visual speech."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_visual_speech_v17 import (
    AutoAVSRVisualFrontend,
    load_auto_avsr_frontend,
)
from active.v17.train_stage_1_v17 import select_device
from active.v17.train_stage_1_visual_speech_v17 import (
    AUTO_AVSR_MODEL_NAME,
    AUTO_AVSR_REPORTED_LRS3_WER,
    AUTO_AVSR_SOURCE,
    AUTO_AVSR_TRAINING_HOURS,
    CitizenVisualSpeechV17Dataset,
    prepare_pixels,
    sha256_file,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.split not in ("train", "val"):
        raise ValueError("frozen visual-speech cache accepts only train/val")
    dataset = CitizenVisualSpeechV17Dataset(
        args.data_root, args.split, args.manifest, args.rejections, args.view
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    device = select_device(args.device)
    frontend = AutoAVSRVisualFrontend()
    load_result = load_auto_avsr_frontend(frontend, str(args.pretrained_checkpoint))
    frontend = frontend.to(device).eval()
    output: list[np.ndarray] = []
    validity: list[np.ndarray] = []
    with torch.inference_mode():
        for pixels, valid, _ in loader:
            pixels, valid_device = prepare_pixels(
                pixels.to(device), valid.to(device), False
            )
            features = frontend(pixels)
            output.append(features.cpu().numpy().astype(np.float16))
            validity.append(valid_device.cpu().numpy().astype(np.bool_))
    features = np.concatenate(output)
    valid = np.concatenate(validity)
    metadata = {
        "format": "slt_auto_avsr_visual_features_v17",
        "split": args.split,
        "view": args.view,
        "samples": len(dataset),
        "shape": list(features.shape),
        "crop_schema_fingerprint": dataset.expected_schema,
        "manifest_sha256": sha256_file(args.manifest),
        "pretrained_checkpoint": str(args.pretrained_checkpoint),
        "pretrained_checkpoint_sha256": sha256_file(args.pretrained_checkpoint),
        "pretraining_source": AUTO_AVSR_SOURCE,
        "pretraining_model": AUTO_AVSR_MODEL_NAME,
        "reported_training_hours": AUTO_AVSR_TRAINING_HOURS,
        "reported_lrs3_visual_wer": AUTO_AVSR_REPORTED_LRS3_WER,
        "frontend_load_result": load_result,
        "pixels_augmented": False,
        "crop_mode": "center_88_from_aligned_112",
        "visual_only": True,
        "audio_accessed": False,
        "test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        features=features,
        valid=valid,
        targets=dataset.targets.numpy(),
        item_ids=np.asarray([str(path.relative_to(args.data_root)) for path in dataset.files]),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(args.output)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/local/citizen100_v17/visual_speech_rgb"))
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument("--rejections", type=Path, default=Path("data/local/citizen100_v17/rejections.csv"))
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--view", choices=("mouth", "lower_face", "full_face"), default="mouth")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
