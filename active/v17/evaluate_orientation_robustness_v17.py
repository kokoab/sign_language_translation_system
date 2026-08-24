#!/usr/bin/env python3
"""Evaluate a frozen v17 landmark checkpoint under arbitrary camera roll.

Only the official Citizen validation split is accepted. The evaluator never writes
back to data, selects a checkpoint, or accesses the consumed test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.train_stage_1_v17 import (
    Citizen100V17Dataset,
    extractor_schema_fingerprint,
    rotate_camera_roll_v17,
    select_device,
)


DEFAULT_ANGLES = (0.0, 17.0, 37.0, 73.0, 90.0, 123.0, 180.0, 270.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_angles(value: str) -> tuple[float, ...]:
    try:
        angles = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("angles must be comma-separated numbers") from exc
    if not angles or not any(abs(angle) < 1e-12 for angle in angles):
        raise argparse.ArgumentTypeError("angles must include zero as the reference")
    if len(set(angles)) != len(angles) or any(not np.isfinite(angle) for angle in angles):
        raise argparse.ArgumentTypeError("angles must be finite and unique")
    return angles


def classification_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    predictions = logits.argmax(axis=1)
    top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
    return {
        "top1": 100.0 * float((predictions == targets).mean()),
        "top1_correct": int((predictions == targets).sum()),
        "top5": 100.0 * float((top5 == targets[:, None]).any(axis=1).mean()),
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("not a v17 Stage 1 checkpoint")
    if checkpoint.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("checkpoint manifest mismatch")
    expected_schema = extractor_schema_fingerprint("apple")
    if checkpoint.get("schema_fingerprint") != expected_schema:
        raise ValueError("checkpoint extractor schema mismatch")
    if checkpoint.get("training_data_provenance", {}).get("citizen_test_accessed") is not False:
        raise ValueError("checkpoint does not prove sealed Citizen test provenance")

    dataset = Citizen100V17Dataset(
        args.data_root,
        "val",
        args.manifest,
        args.rejections,
        cache=True,
        expected_schema=expected_schema,
    )
    if checkpoint.get("label_to_index") != dataset.label_to_index:
        raise ValueError("checkpoint label mapping mismatch")
    device = select_device(args.device)
    model_config = dict(checkpoint["model_config"])
    if args.force_canonicalize_camera_roll:
        model_config["canonicalize_camera_roll"] = True
    model = SLTStage1V17(Stage1V17Config(**model_config))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    angle_logits: list[np.ndarray] = []
    with torch.inference_mode():
        for angle_degrees in args.angles:
            batches = []
            angle_radians = angle_degrees * np.pi / 180.0
            for features, _ in loader:
                features = rotate_camera_roll_v17(
                    features.to(device), angle_radians
                )
                logits = model(features)
                if device.type == "mps":
                    torch.mps.synchronize()
                batches.append(logits.float().cpu().numpy())
            angle_logits.append(np.concatenate(batches))

    values = np.stack(angle_logits)
    targets = dataset.targets.numpy()
    zero_index = args.angles.index(0.0)
    reference_predictions = values[zero_index].argmax(axis=1)
    rows = []
    for angle, logits in zip(args.angles, values):
        predictions = logits.argmax(axis=1)
        rows.append(
            {
                "angle_degrees": angle,
                **classification_metrics(logits, targets),
                "prediction_agreement_with_zero": 100.0
                * float((predictions == reference_predictions).mean()),
            }
        )
    nonzero = [row for row in rows if row["angle_degrees"] != 0.0]
    result: dict[str, object] = {
        "format": "slt_v17_orientation_robustness_evaluation",
        "split": "citizen_official_val",
        "samples": len(dataset),
        "angles_degrees": list(args.angles),
        "per_angle": rows,
        "worst_nonzero_top1": min(float(row["top1"]) for row in nonzero),
        "mean_nonzero_top1": float(np.mean([row["top1"] for row in nonzero])),
        "worst_nonzero_prediction_agreement": min(
            float(row["prediction_agreement_with_zero"]) for row in nonzero
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "camera_roll_canonicalization": bool(model.config.canonicalize_camera_roll),
        "model_inference_accessed": True,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "logits.npz",
        logits=values.astype(np.float32),
        targets=targets,
        angles_degrees=np.asarray(args.angles, dtype=np.float32),
        item_ids=np.asarray(
            [str(path.relative_to(args.data_root)) for path in dataset.files]
        ),
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("data/local/citizen100_v17/landmarks"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("active/v17/citizen100_manifest.json"),
    )
    parser.add_argument(
        "--rejections", type=Path,
        default=Path("data/local/citizen100_v17/rejections.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--angles", type=parse_angles,
        default=DEFAULT_ANGLES,
        help="Comma-separated degrees; must include zero",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--force-canonicalize-camera-roll", action="store_true",
        help="Diagnostic only: enable parameter-free roll canonicalization on an old checkpoint",
    )
    return parser


def main() -> int:
    result = evaluate(build_parser().parse_args())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
