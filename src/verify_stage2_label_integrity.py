"""
Stage-2 model-based label integrity verification.

Compares the Stage-2 CTC-decoded gloss prediction (from extracted landmarks)
to the expected label derived from:
  - the parent folder name under `data/raw_videos/ASL VIDEOS` / class label
  - the label prefix in `ASL_landmarks_float16/<label>_<stem>_<hash>.npy`

Outputs:
  - CSV with per-sample prediction + strict/contains match flags
  - JSON summary with per-class mismatch counts
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from train_stage_2 import SLTStage2CTC


BLANK_TOKEN_ID = 0


def parse_label_and_stem(npy_name: str) -> Tuple[str, str]:
    """
    Expected filename format:
      {label}_{stem}_{hash}.npy
    where label can include underscores.
    """
    base = npy_name[:-4]  # strip ".npy"
    label, stem, _hash = base.rsplit("_", 2)
    return label, stem


def find_mp4_for_stem(
    mp4_root: Path, label: str, stem: str, exts: Sequence[str] = (".mp4", ".MP4", ".mov", ".MOV")
) -> Optional[str]:
    label_dir = mp4_root / label
    if not label_dir.exists() or not label_dir.is_dir():
        return None
    for ext in exts:
        p = label_dir / f"{stem}{ext}"
        if p.exists():
            return str(p)
    return None


def ctc_collapse(token_ids: Sequence[int], blank: int = BLANK_TOKEN_ID) -> List[int]:
    """
    Collapse CTC output:
      - remove blanks
      - remove consecutive repeats (after blank removal resets the repeat guard)
    Matches the logic used in `decode_ctc` in `src/train_stage_2.py`.
    """
    decoded: List[int] = []
    last_tok = blank
    for tok in token_ids:
        if tok != blank and tok != last_tok:
            decoded.append(int(tok))
        last_tok = int(tok)
    return decoded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks-dir", default="ASL_landmarks_float16")
    ap.add_argument("--mp4-root", default=str(Path("data/raw_videos/ASL VIDEOS")))
    ap.add_argument("--stage2-ckpt", default="weights/stage2_best_model.pth")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    ap.add_argument("--output-csv", default="verification_outputs/stage2_label_integrity_mismatches.csv")
    ap.add_argument("--output-json", default="verification_outputs/stage2_label_integrity_summary.json")
    args = ap.parse_args()

    landmarks_dir = Path(args.landmarks_dir)
    mp4_root = Path(args.mp4_root)
    stage2_ckpt_path = Path(args.stage2_ckpt)
    assert landmarks_dir.exists(), f"Missing landmarks dir: {landmarks_dir}"
    assert stage2_ckpt_path.exists(), f"Missing Stage-2 checkpoint: {stage2_ckpt_path}"

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(str(stage2_ckpt_path), map_location="cpu", weights_only=False)
    vocab_size = int(ckpt["vocab_size"])
    idx_to_gloss: Dict[int, str] = ckpt["idx_to_gloss"]

    # Load model
    model = SLTStage2CTC(vocab_size=vocab_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load sample list (based on existing extracted npy outputs)
    npy_paths = sorted([p for p in landmarks_dir.iterdir() if p.is_file() and p.suffix == ".npy"])
    if args.max_samples and args.max_samples > 0:
        npy_paths = npy_paths[: args.max_samples]

    os_total = len(npy_paths)
    print(f"[INFO] Loaded model on {device}. Samples={os_total}")

    out_csv_path = Path(args.output_csv)
    out_json_path = Path(args.output_json)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    class_stats: Dict[str, Dict[str, int]] = {}

    def ensure_stats(label: str) -> None:
        if label not in class_stats:
            class_stats[label] = {
                "n": 0,
                "match_strict": 0,
                "mismatch_strict": 0,
                # More practical: treat as match if the model only outputs the expected label
                # (e.g. CTC may output [ABOUT, ABOUT]).
                "match_expected_all_tokens": 0,
                "mismatch_expected_all_tokens": 0,
                # Softer signal: expected label appears somewhere in decoded sequence.
                "match_contains": 0,
                "mismatch_contains": 0,
            }

    # Main loop
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
                label, stem = parse_label_and_stem(p.name)
                expected_labels.append(label)
                stems.append(stem)
                arr = np.load(p).astype(np.float32, copy=False)
                if arr.shape != (32, 47, 10):
                    raise ValueError(f"Bad shape for {p}: got {arr.shape}, expected (32,47,10)")
                xs[i] = arr

            x_pad = torch.from_numpy(xs).to(device)
            x_lens = torch.full((B,), 32, dtype=torch.long, device=device)

            logits, out_lens = model(x_pad, x_lens)
            log_probs = torch.log_softmax(logits, dim=-1)  # [B, T', V]
            pred_ids = torch.argmax(log_probs, dim=-1)  # [B, T']

            for b in range(B):
                expected = expected_labels[b]
                ensure_stats(expected)
                class_stats[expected]["n"] += 1

                out_len_b = int(out_lens[b].item())
                seq_ids = pred_ids[b, :out_len_b].tolist()
                decoded_ids = ctc_collapse(seq_ids, blank=BLANK_TOKEN_ID)
                decoded_labels = [idx_to_gloss.get(int(t), f"UNK_{t}") for t in decoded_ids]

                # If model predicts nothing but blanks, treat as empty list.
                # Confidence proxy: mean probability of the greedy argmax token per timestep.
                greedy_prob = torch.exp(log_probs[b, torch.arange(out_len_b, device=log_probs.device), pred_ids[b, :out_len_b]])
                confidence = float(greedy_prob.mean().item()) if out_len_b > 0 else 0.0

                pred_str = " ".join(decoded_labels) if decoded_labels else "(empty)"
                strict_ok = decoded_labels == [expected]
                contains_ok = expected in decoded_labels
                all_expected_ok = bool(decoded_labels) and all(t == expected for t in decoded_labels)

                mp4_path = find_mp4_for_stem(mp4_root=mp4_root, label=expected, stem=stems[b])

                class_stats[expected]["match_strict"] += int(strict_ok)
                if not strict_ok:
                    class_stats[expected]["mismatch_strict"] += 1
                class_stats[expected]["match_expected_all_tokens"] += int(all_expected_ok)
                if not all_expected_ok:
                    class_stats[expected]["mismatch_expected_all_tokens"] += 1

                # Softer mismatch signal: expected label never appears in decoded list
                class_stats[expected]["match_contains"] += int(contains_ok)
                if not contains_ok:
                    class_stats[expected]["mismatch_contains"] += 1

                rows.append(
                    {
                        "npy_name": npy_names[b],
                        "expected_label": expected,
                        "predicted_glosses": pred_str,
                        "decoded_count": len(decoded_labels),
                        "match_strict": strict_ok,
                        "match_expected_all_tokens": all_expected_ok,
                        "match_contains": contains_ok,
                        "confidence_mean_argmax": confidence,
                        "mp4_path": mp4_path if mp4_path is not None else "",
                    }
                )

            if (start // args.batch_size) % 50 == 0:
                done = min(start + B, len(npy_paths))
                print(f"[INFO] Processed {done}/{len(npy_paths)} samples...")

    # Write outputs
    fieldnames = [
        "npy_name",
        "expected_label",
        "predicted_glosses",
        "decoded_count",
        "match_strict",
        "match_expected_all_tokens",
        "match_contains",
        "confidence_mean_argmax",
        "mp4_path",
    ]

    with out_csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # Ensure booleans are JSON/csv-friendly
            r2 = dict(r)
            w.writerow(r2)

    # Summary JSON
    total_n = len(rows)
    strict_matches = sum(1 for r in rows if r["match_strict"])
    all_tokens_matches = sum(1 for r in rows if r["match_expected_all_tokens"])
    contains_matches = sum(1 for r in rows if r["match_contains"])

    summary = {
        "total_samples": total_n,
        "strict_match_count": int(strict_matches),
        "strict_match_rate": strict_matches / max(total_n, 1),
        "match_expected_all_tokens_count": int(all_tokens_matches),
        "match_expected_all_tokens_rate": all_tokens_matches / max(total_n, 1),
        "contains_match_count": int(contains_matches),
        "contains_match_rate": contains_matches / max(total_n, 1),
        "device": str(device),
        "idx_to_gloss_blank_id": BLANK_TOKEN_ID,
        "per_class": class_stats,
    }
    out_json_path.write_text(json.dumps(summary, indent=2))

    # Print top mismatches
    # (Highest confidence strict mismatches where expected doesn't match)
    mismatches = [r for r in rows if not r["match_expected_all_tokens"]]
    mismatches.sort(key=lambda x: float(x["confidence_mean_argmax"]), reverse=True)
    print(f"[INFO] Done. Strict match rate: {summary['strict_match_rate']:.4f}")
    print(f"[INFO] Expected-only-token mismatches: {len(mismatches)}")
    print("[INFO] Top 20 expected-only-token mismatches by confidence:")
    for r in mismatches[:20]:
        print(
            f"  - {r['expected_label']} | pred={r['predicted_glosses']} | conf={r['confidence_mean_argmax']:.4f} | npy={r['npy_name']} | mp4={r['mp4_path']}"
        )


if __name__ == "__main__":
    main()

