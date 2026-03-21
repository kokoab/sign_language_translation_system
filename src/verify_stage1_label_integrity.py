"""
Stage-1 model-based label integrity verification (fast).

For each extracted landmark clip in `ASL_landmarks_float16/*.npy`:
  - expected_label = <label> prefix from filename: {label}_{stem}_{hash}.npy
  - predicted_label = Stage-1 classifier argmax label from `weights/best_model.pth`
  - compare expected vs predicted and write mismatches to CSV/JSON

Stage-1 is a single-sign classifier, so this is a practical way to flag
clips that are likely stored under the wrong ASL class folder.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from train_stage_1 import SLTStage1


def parse_label_and_stem(npy_name: str) -> Tuple[str, str]:
    # Expected filename format: {label}_{stem}_{hash}.npy
    base = npy_name[:-4]  # strip ".npy"
    label, stem, _hash = base.rsplit("_", 2)
    return label, stem


def find_mp4_for_stem(
    mp4_root: Path,
    label: str,
    stem: str,
    exts: Sequence[str] = (".mp4", ".MP4", ".mov", ".MOV"),
) -> Optional[str]:
    label_dir = mp4_root / label
    if not label_dir.exists() or not label_dir.is_dir():
        return None
    for ext in exts:
        p = label_dir / f"{stem}{ext}"
        if p.exists():
            return str(p)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks-dir", default="ASL_landmarks_float16")
    ap.add_argument("--mp4-root", default=str(Path("data/raw_videos/ASL VIDEOS")))
    ap.add_argument("--stage1-ckpt", default="weights/best_model.pth")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    ap.add_argument("--output-csv", default="verification_outputs/stage1_label_integrity_mismatches_all.csv")
    ap.add_argument("--output-json", default="verification_outputs/stage1_label_integrity_summary_all.json")
    args = ap.parse_args()

    landmarks_dir = Path(args.landmarks_dir)
    mp4_root = Path(args.mp4_root)
    stage1_ckpt_path = Path(args.stage1_ckpt)
    assert landmarks_dir.exists(), f"Missing landmarks dir: {landmarks_dir}"
    assert stage1_ckpt_path.exists(), f"Missing Stage-1 checkpoint: {stage1_ckpt_path}"

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(str(stage1_ckpt_path), map_location="cpu", weights_only=False)
    num_classes = int(ckpt["num_classes"])
    in_channels = int(ckpt.get("in_channels", 10))
    d_model = int(ckpt["d_model"])
    nhead = int(ckpt["nhead"])
    num_transformer_layers = int(ckpt["num_transformer_layers"])

    idx_to_label: Dict[str, str] = ckpt["idx_to_label"]

    # Build model and restore weights.
    model = SLTStage1(
        num_classes=num_classes,
        in_channels=in_channels,
        d_model=d_model,
        nhead=nhead,
        num_transformer_layers=num_transformer_layers,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Collect npy samples
    npy_paths = sorted([p for p in landmarks_dir.iterdir() if p.is_file() and p.suffix == ".npy"])
    if args.max_samples and args.max_samples > 0:
        npy_paths = npy_paths[: args.max_samples]

    out_csv_path = Path(args.output_csv)
    out_json_path = Path(args.output_json)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    class_stats: Dict[str, Dict[str, int]] = {}

    def ensure_stats(label: str) -> None:
        if label not in class_stats:
            class_stats[label] = {"n": 0, "match": 0, "mismatch": 0}

    with torch.no_grad():
        for start in range(0, len(npy_paths), args.batch_size):
            batch_paths = npy_paths[start : start + args.batch_size]
            B = len(batch_paths)

            xs = np.empty((B, 32, 47, 10), dtype=np.float32)
            expected_labels: List[str] = []
            stems: List[str] = []
            npy_names: List[str] = []

            for i, p in enumerate(batch_paths):
                npy_names.append(p.name)
                expected, stem = parse_label_and_stem(p.name)
                expected_labels.append(expected)
                stems.append(stem)

                arr = np.load(p).astype(np.float32, copy=False)
                if arr.shape != (32, 47, 10):
                    raise ValueError(f"Bad shape for {p}: got {arr.shape}, expected (32,47,10)")
                xs[i] = arr

            x = torch.from_numpy(xs).to(device)
            logits = model(x)  # [B, num_classes]
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).cpu().numpy().tolist()  # python ints

            pred_confs = torch.max(probs, dim=-1).values.cpu().numpy().astype(float).tolist()

            for b in range(B):
                expected = expected_labels[b]
                ensure_stats(expected)

                pred_id = int(pred_ids[b])
                pred_label = idx_to_label.get(str(pred_id), idx_to_label.get(pred_id, f"UNK_{pred_id}"))

                ok = pred_label == expected
                class_stats[expected]["n"] += 1
                class_stats[expected]["match"] += int(ok)
                class_stats[expected]["mismatch"] += int(not ok)

                mp4_path = find_mp4_for_stem(mp4_root=mp4_root, label=expected, stem=stems[b])

                if not ok:
                    rows.append(
                        {
                            "npy_name": npy_names[b],
                            "expected_label": expected,
                            "predicted_label": pred_label,
                            "confidence": float(pred_confs[b]),
                            "mp4_path": mp4_path if mp4_path is not None else "",
                        }
                    )

            if (start // args.batch_size) % 200 == 0:
                done = min(start + B, len(npy_paths))
                print(f"[INFO] Processed {done}/{len(npy_paths)} samples...")

    # Write mismatches CSV
    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["npy_name", "expected_label", "predicted_label", "confidence", "mp4_path"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    total_n = int(sum(s["n"] for s in class_stats.values()))
    total_mismatch = int(sum(s["mismatch"] for s in class_stats.values()))
    strict_rate = 1.0 - (total_mismatch / max(total_n, 1))

    summary = {
        "total_samples": total_n,
        "mismatch_count": total_mismatch,
        "match_rate": strict_rate,
        "device": str(device),
        "per_class": class_stats,
    }
    out_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print top mismatches by confidence
    rows.sort(key=lambda r: float(r["confidence"]), reverse=True)
    print(f"[INFO] Done. Match rate: {strict_rate:.4f} | Mismatches: {total_mismatch}")
    print("[INFO] Top 30 mismatches by confidence:")
    for r in rows[:30]:
        print(
            f"  - expected={r['expected_label']} predicted={r['predicted_label']} conf={r['confidence']:.4f} npy={r['npy_name']} mp4={r['mp4_path']}"
        )


if __name__ == "__main__":
    main()

