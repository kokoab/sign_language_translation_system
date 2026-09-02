#!/usr/bin/env python3
"""Train a landmark-only Stage-2 CTC preview for conditional live inference.

The accepted multimodal selector remains the fallback teacher. The preview consumes
only the landmark-token slice already present in approved Stage-2 caches, derives
per-frame landmark logits with the frozen Stage-1 classifier, and never sees hand RGB.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import copy
import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_stage2_v17 import (
    Stage2TemporalHeadV17,
    load_frozen_unified_stage1,
    load_stage2_general_ctc_selector,
)
from active.v17.train_stage_2_balanced_multivoice_v17 import (
    domain_edits,
    source_weights,
)
from active.v17.train_stage_2_general_selector_distill_v17 import sha256
from active.v17.train_stage_2_v17 import (
    CombinedDataset,
    RealPhraseDataset,
    SyntheticCompositionDataset,
    collate,
    evaluate,
)


LOG = logging.getLogger("train_stage_2_landmark_cascade_v17")
LANDMARK_DIM = 256
HAND_DIM = 256


class LandmarkStage2Preview(nn.Module):
    """Rebuild the 612-D Stage-2 contract without hand-image evidence."""

    def __init__(self, model: Stage2TemporalHeadV17, landmark_classifier: nn.Module):
        super().__init__()
        self.model = model
        self.landmark_classifier = landmark_classifier.eval()
        for parameter in self.landmark_classifier.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.landmark_classifier.eval()
        return self

    def landmark_view(self, frozen_features: torch.Tensor) -> torch.Tensor:
        tokens = frozen_features[..., :LANDMARK_DIM]
        with torch.no_grad():
            logits = self.landmark_classifier(tokens)
        return torch.cat((tokens, torch.zeros_like(tokens), logits), dim=-1)

    def forward(self, frozen_features: torch.Tensor, window_mask: torch.Tensor):
        return self.model(self.landmark_view(frozen_features), window_mask)


def validation_edits(phrase: dict[str, object], context: dict[str, object]) -> list[int]:
    return [
        domain_edits(phrase, "asllrp_contiguous"),
        domain_edits(phrase, "local_phrases"),
        domain_edits(context, "asllrp_segmented_validation"),
    ]


def selection_key(edits: list[int]) -> tuple[int, ...]:
    return (-sum(edits), -edits[1], -edits[2], -edits[0])


def train_seed(
    seed: int,
    teacher,
    landmark_classifier: nn.Module,
    train_dataset,
    weights: torch.Tensor,
    phrase_loader: DataLoader,
    context_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = copy.deepcopy(teacher.primary.base).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = True
    preview = LandmarkStage2Preview(
        model, copy.deepcopy(landmark_classifier).to(device)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=args.samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, collate_fn=collate,
    )
    initial_phrase = evaluate(preview, phrase_loader, device)
    initial_context = evaluate(preview, context_loader, device)
    initial_edits = validation_edits(initial_phrase, initial_context)
    best = {
        "epoch": 0,
        "edits": initial_edits,
        "phrase": initial_phrase,
        "context": initial_context,
        "state": copy.deepcopy(model.state_dict()),
    }
    history = [{
        "epoch": 0,
        "edits": initial_edits,
        "phrase_validation": initial_phrase,
        "context_validation": initial_context,
    }]
    patience = 0
    for epoch in range(1, args.epochs + 1):
        preview.train()
        teacher.eval()
        totals = {
            "loss": 0.0, "ctc": 0.0, "distill": 0.0,
            "samples": 0, "discarded_nonfinite_batches": 0,
        }
        for batch in loader:
            full = batch["features"].to(device)
            mask = batch["window_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, lengths = preview(full, mask)
            with torch.inference_mode():
                teacher_logits, teacher_lengths, _ = teacher.forward_with_selection(
                    full, mask
                )
            if not torch.equal(lengths, teacher_lengths):
                raise RuntimeError("teacher and preview CTC lengths differ")
            ctc = criterion(
                logits.log_softmax(-1).transpose(0, 1),
                batch["targets"].to(device), lengths,
                batch["target_lengths"].to(device),
            )
            token_mask = mask.repeat_interleave(model.config.tokens_per_window, 1)
            temperature = args.temperature
            per_token = F.kl_div(
                F.log_softmax(logits / temperature, -1),
                F.softmax(teacher_logits / temperature, -1),
                reduction="none",
            ).sum(-1)
            distill = (per_token * token_mask).sum() / token_mask.sum().clamp_min(1)
            distill = distill * temperature * temperature
            loss = args.ctc_weight * ctc + args.distill_weight * distill
            if not torch.isfinite(loss):
                totals["discarded_nonfinite_batches"] += 1
                if totals["discarded_nonfinite_batches"] > args.max_nonfinite_batches:
                    raise RuntimeError(
                        f"repeated non-finite loss at seed {seed}, epoch {epoch}"
                    )
                continue
            loss.backward()
            if any(
                parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ):
                optimizer.zero_grad(set_to_none=True)
                totals["discarded_nonfinite_batches"] += 1
                if totals["discarded_nonfinite_batches"] > args.max_nonfinite_batches:
                    raise RuntimeError(
                        f"repeated non-finite gradient at seed {seed}, epoch {epoch}"
                    )
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            count = len(mask)
            totals["loss"] += float(loss.detach()) * count
            totals["ctc"] += float(ctc.detach()) * count
            totals["distill"] += float(distill.detach()) * count
            totals["samples"] += count
        scheduler.step()
        phrase = evaluate(preview, phrase_loader, device)
        context = evaluate(preview, context_loader, device)
        edits = validation_edits(phrase, context)
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / totals["samples"],
            "ctc_loss": totals["ctc"] / totals["samples"],
            "distill_loss": totals["distill"] / totals["samples"],
            "discarded_nonfinite_batches": totals["discarded_nonfinite_batches"],
            "edits": edits,
            "phrase_validation": phrase,
            "context_validation": context,
        }
        history.append(row)
        if selection_key(edits) > selection_key(best["edits"]):
            best = {
                "epoch": epoch, "edits": edits, "phrase": phrase,
                "context": context, "state": copy.deepcopy(model.state_dict()),
            }
            patience = 0
        else:
            patience += 1
        LOG.info(
            "seed=%d epoch=%d loss=%.4f edits=%s best=%d patience=%d",
            seed, epoch, row["loss"], edits, best["epoch"], patience,
        )
        if patience >= args.patience:
            break
    seed_dir = args.output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "slt_stage2_landmark_preview_v17",
        "format_version": 1,
        "model_config": model.config.to_dict(),
        "model_state_dict": best["state"],
        "seed": seed,
        "epoch": best["epoch"],
        "validation_edits": best["edits"],
        "phrase_validation": best["phrase"],
        "context_validation": best["context"],
        "feature_contract": "landmark_tokens_256 + zero_hand_256 + landmark_logits_100",
        "teacher": args.teacher.as_posix(),
        "teacher_sha256": sha256(args.teacher),
        "stage1_checkpoint": args.stage1_checkpoint.as_posix(),
        "stage1_checkpoint_sha256": sha256(args.stage1_checkpoint),
        "test_evaluated": False,
    }
    torch.save(checkpoint, seed_dir / "best_model.pth")
    (seed_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {
        "seed": seed,
        "epoch": best["epoch"],
        "edits": best["edits"],
        "phrase_validation": best["phrase"],
        "context_validation": best["context"],
        "checkpoint": (seed_dir / "best_model.pth").as_posix(),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else args.device
    )
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    teacher, _ = load_stage2_general_ctc_selector(args.teacher)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    landmark, _, _, _ = load_frozen_unified_stage1(args.stage1_checkpoint)
    phrase_train = RealPhraseDataset(args.phrase_root, "train")
    context_train = RealPhraseDataset(args.context_train_root, "train")
    synthetic = SyntheticCompositionDataset(args.synthetic_pool, args.synthetic_plan)
    train_dataset = CombinedDataset([phrase_train, context_train, synthetic])
    desired = {
        "local_phrases": 0.30,
        "asllrp_contiguous": 0.20,
        "asllrp_segmented_train": 0.30,
        "synthetic_citizen_train": 0.05,
        "synthetic_multivoice_train": 0.075,
        "synthetic_balanced_multivoice_train": 0.075,
    }
    weights, counts = source_weights(
        phrase_train, context_train, synthetic, desired
    )
    phrase_loader = DataLoader(
        RealPhraseDataset(args.phrase_root, "validation"),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate,
    )
    context_loader = DataLoader(
        RealPhraseDataset(args.context_validation_root, "validation"),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    candidates = [
        train_seed(
            seed, teacher, landmark.classifier, train_dataset, weights,
            phrase_loader, context_loader, args, device,
        )
        for seed in args.seeds
    ]
    winner = max(candidates, key=lambda row: selection_key(row["edits"]))
    selected_path = Path(winner["checkpoint"])
    payload = torch.load(selected_path, map_location="cpu", weights_only=False)
    output_path = args.output / "best_model.pth"
    torch.save(payload, output_path)
    result = {
        "selected_seed": winner["seed"],
        "selected_epoch": winner["epoch"],
        "edits": winner["edits"],
        "phrase_validation": winner["phrase_validation"],
        "context_validation": winner["context_validation"],
        "checkpoint": output_path.as_posix(),
        "checkpoint_sha256": sha256(output_path),
        "training_source_counts": counts,
        "training_source_sampling_mass": desired,
        "candidate_results": candidates,
        "seconds": time.monotonic() - started,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--teacher", type=Path, default=Path(
        "artifacts/models/stage2_v17_general_ctc_selector_v1/model.pth"
    ))
    value.add_argument("--stage1-checkpoint", type=Path, default=Path(
        "artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"
    ))
    value.add_argument("--phrase-root", type=Path, default=Path(
        "data/local/stage2_v17_frozen_features"
    ))
    value.add_argument("--context-train-root", type=Path, default=Path(
        "data/local/stage2_v17_asllrp_segmented_train_frozen_features"
    ))
    value.add_argument("--context-validation-root", type=Path, default=Path(
        "data/local/stage2_v17_asllrp_segmented_validation_frozen_features"
    ))
    value.add_argument("--synthetic-pool", type=Path, default=Path(
        "data/local/stage2_v17_synthetic/train_only_multivoice_pool_v3.npz"
    ))
    value.add_argument("--synthetic-plan", type=Path, default=Path(
        "active/v17/stage2_balanced_multivoice_plan_v17.json"
    ))
    value.add_argument("--output", type=Path, default=Path(
        "artifacts/models/stage2_v17_landmark_cascade_preview_v1"
    ))
    value.add_argument("--seeds", type=int, nargs="+", default=(9811,))
    value.add_argument("--device", default="auto")
    value.add_argument("--mps-memory-fraction", type=float, default=0.12)
    value.add_argument("--epochs", type=int, default=12)
    value.add_argument("--patience", type=int, default=4)
    value.add_argument("--batch-size", type=int, default=12)
    value.add_argument("--samples-per-epoch", type=int, default=3000)
    value.add_argument("--lr", type=float, default=5e-6)
    value.add_argument("--weight-decay", type=float, default=0.01)
    value.add_argument("--ctc-weight", type=float, default=0.5)
    value.add_argument("--distill-weight", type=float, default=1.0)
    value.add_argument("--temperature", type=float, default=2.0)
    value.add_argument("--gradient-clip", type=float, default=0.5)
    value.add_argument("--max-nonfinite-batches", type=int, default=2)
    return value


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
