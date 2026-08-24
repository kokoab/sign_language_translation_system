#!/usr/bin/env python3
"""Evaluate a phrase-agnostic CTC cross-score selector for two Stage-2 heads.

The selector never inspects gloss identities.  It may replace the calibrated
primary greedy hypothesis with the direct-transition specialist hypothesis only
when both contain the same number of non-blank tokens and the alternate has a
large enough sequence-probability advantage.  Score weights and the threshold are
selected from training rows, then frozen before any validation row is evaluated.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_stage2_v17 import (
    Stage2TemporalHeadV17,
    Stage2V17Config,
    load_stage2_context_adapted,
)
from active.v17.train_stage_2_v17 import (
    CombinedDataset,
    RealPhraseDataset,
    collate,
    collapse_ctc,
    edit_distance,
)


@dataclass(frozen=True)
class Row:
    source: str
    item_id: str
    reference: tuple[int, ...]
    primary_logits: torch.Tensor
    specialist_logits: torch.Tensor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_specialist(path: Path) -> Stage2TemporalHeadV17:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "slt_stage2_ctc_v17":
        raise ValueError(f"{path}: expected a plain v17 CTC checkpoint")
    model = Stage2TemporalHeadV17(Stage2V17Config(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def make_loader(datasets: list[RealPhraseDataset], batch_size: int) -> DataLoader:
    return DataLoader(
        CombinedDataset(datasets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )


def collect_rows(
    primary: torch.nn.Module,
    specialist: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[Row]:
    primary.eval()
    specialist.eval()
    rows: list[Row] = []
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["window_mask"].to(device)
            primary_logits, lengths = primary(features, mask)
            specialist_logits, specialist_lengths = specialist(features, mask)
            if not torch.equal(lengths, specialist_lengths):
                raise RuntimeError("primary and specialist lengths differ")
            lengths_list = lengths.cpu().tolist()
            target_lengths = batch["target_lengths"].tolist()
            targets = batch["targets"].tolist()
            offset = 0
            for index, target_length in enumerate(target_lengths):
                reference = tuple(
                    int(token) - 1 for token in targets[offset:offset + target_length]
                )
                offset += target_length
                length = int(lengths_list[index])
                rows.append(Row(
                    source=str(batch["sources"][index]),
                    item_id=str(batch["item_ids"][index]),
                    reference=reference,
                    primary_logits=primary_logits[index, :length].detach().cpu().float(),
                    specialist_logits=(
                        specialist_logits[index, :length].detach().cpu().float()
                    ),
                ))
    return rows


def greedy(logits: torch.Tensor) -> tuple[int, ...]:
    return tuple(collapse_ctc(logits.argmax(dim=-1).numpy()))


def calibrated_logits(row: Row, blend_weight: float, blank_bias: float) -> torch.Tensor:
    logits = row.primary_logits * (1.0 - blend_weight) + row.specialist_logits * blend_weight
    logits = logits.clone()
    logits[:, 0] += blank_bias
    return logits


def ctc_log_probability(logits: torch.Tensor, hypothesis: tuple[int, ...]) -> float:
    if not hypothesis:
        return float(logits.log_softmax(dim=-1)[:, 0].sum().item())
    targets = torch.tensor([token + 1 for token in hypothesis], dtype=torch.long)
    loss = F.ctc_loss(
        logits.log_softmax(dim=-1).unsqueeze(1),
        targets,
        input_lengths=torch.tensor([len(logits)], dtype=torch.long),
        target_lengths=torch.tensor([len(targets)], dtype=torch.long),
        reduction="sum",
        zero_infinity=False,
    )
    value = -float(loss.item())
    return value if np.isfinite(value) else -1.0e30


def metrics(rows: list[Row], hypotheses: list[tuple[int, ...]]) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"edits": 0, "tokens": 0, "exact": 0, "samples": 0}
    )
    for row, hypothesis in zip(rows, hypotheses):
        stats = totals[row.source]
        stats["edits"] += edit_distance(list(row.reference), list(hypothesis))
        stats["tokens"] += len(row.reference)
        stats["exact"] += int(row.reference == hypothesis)
        stats["samples"] += 1
    domains = {
        source: {
            **values,
            "wer": values["edits"] / max(1, values["tokens"]),
            "sequence_accuracy": values["exact"] / max(1, values["samples"]),
        }
        for source, values in sorted(totals.items())
    }
    return {
        "domains": domains,
        "equal_domain_mean_wer": float(
            np.mean([domain["wer"] for domain in domains.values()])
        ),
        "worst_domain_wer": float(max(domain["wer"] for domain in domains.values())),
        "total_edits": int(sum(domain["edits"] for domain in domains.values())),
        "total_tokens": int(sum(domain["tokens"] for domain in domains.values())),
    }


def prepare(
    rows: list[Row], blend_weight: float, blank_bias: float, normalization: str,
    candidate_mode: str,
) -> dict[str, Any]:
    base_hypotheses: list[tuple[int, ...]] = []
    alternate_hypotheses: list[tuple[int, ...]] = []
    primary_deltas = np.full(len(rows), -np.inf, dtype=np.float64)
    specialist_deltas = np.full(len(rows), -np.inf, dtype=np.float64)
    eligible = np.zeros(len(rows), dtype=np.bool_)
    for index, row in enumerate(rows):
        base_logits = calibrated_logits(row, blend_weight, blank_bias)
        base = greedy(base_logits)
        alternate = greedy(row.specialist_logits)
        base_hypotheses.append(base)
        alternate_hypotheses.append(alternate)
        if base == alternate or len(alternate) < 2:
            continue
        if candidate_mode == "equal_length":
            candidate_allowed = len(base) >= 2 and len(base) == len(alternate)
        elif candidate_mode == "nonshortening_one":
            candidate_allowed = 0 <= len(alternate) - len(base) <= 1
        elif candidate_mode == "equal_or_prefix_extension":
            candidate_allowed = (
                (len(base) >= 2 and len(base) == len(alternate))
                or (
                    len(alternate) == len(base) + 1
                    and alternate[:len(base)] == base
                )
            )
        else:
            raise ValueError(f"unknown candidate mode: {candidate_mode}")
        if not candidate_allowed:
            continue
        divisor = {
            "none": 1.0,
            "tokens": float(len(base)),
            "frames": float(len(base_logits)),
        }[normalization]
        primary_deltas[index] = (
            ctc_log_probability(row.primary_logits, alternate)
            - ctc_log_probability(row.primary_logits, base)
        ) / divisor
        specialist_deltas[index] = (
            ctc_log_probability(row.specialist_logits, alternate)
            - ctc_log_probability(row.specialist_logits, base)
        ) / divisor
        eligible[index] = True
    return {
        "base": base_hypotheses,
        "alternate": alternate_hypotheses,
        "primary_deltas": primary_deltas,
        "specialist_deltas": specialist_deltas,
        "eligible": eligible,
    }


def apply_selector(
    prepared: dict[str, Any], primary_score_weight: float, margin: float
) -> tuple[list[tuple[int, ...]], np.ndarray, np.ndarray]:
    deltas = combine_deltas(prepared, primary_score_weight)
    selected = prepared["eligible"] & (deltas >= margin)
    hypotheses = [
        prepared["alternate"][index] if selected[index] else prepared["base"][index]
        for index in range(len(selected))
    ]
    return hypotheses, selected, deltas


def combine_deltas(prepared: dict[str, Any], primary_score_weight: float) -> np.ndarray:
    """Mix finite eligible score deltas without creating ``0 * -inf`` NaNs."""
    deltas = np.full(len(prepared["eligible"]), -np.inf, dtype=np.float64)
    eligible = prepared["eligible"]
    deltas[eligible] = (
        prepared["primary_deltas"][eligible] * primary_score_weight
        + prepared["specialist_deltas"][eligible] * (1.0 - primary_score_weight)
    )
    return deltas


def select_on_train(
    rows: list[Row], blend_weight: float, blank_bias: float, candidate_mode: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    best: tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]] | None = None
    audit = {"configurations": 0, "feasible_configurations": 0}
    for normalization in ("none", "tokens", "frames"):
        prepared = prepare(
            rows, blend_weight, blank_bias, normalization, candidate_mode
        )
        base_metrics = metrics(rows, prepared["base"])
        base_domain_edits = {
            source: int(values["edits"])
            for source, values in base_metrics["domains"].items()
        }
        for primary_weight in np.linspace(0.0, 1.0, 21):
            deltas = combine_deltas(prepared, float(primary_weight))
            finite = np.sort(np.unique(deltas[np.isfinite(deltas)]))
            if len(finite):
                margins = np.concatenate((
                    [finite[-1] + 1.0e-6],
                    finite,
                    [finite[0] - 1.0e-6],
                ))
            else:
                margins = np.asarray([0.0])
            for margin in margins.tolist():
                audit["configurations"] += 1
                hypotheses, selected, _ = apply_selector(
                    prepared, float(primary_weight), float(margin)
                )
                result = metrics(rows, hypotheses)
                no_domain_worse = all(
                    int(values["edits"]) <= base_domain_edits[source]
                    for source, values in result["domains"].items()
                )
                if not no_domain_worse:
                    continue
                audit["feasible_configurations"] += 1
                key = (
                    result["equal_domain_mean_wer"],
                    result["worst_domain_wer"],
                    result["total_edits"],
                    int(selected.sum()),
                    -float(margin),
                    normalization,
                    float(primary_weight),
                )
                config = {
                    "normalization": normalization,
                    "primary_score_weight": float(primary_weight),
                    "specialist_score_weight": 1.0 - float(primary_weight),
                    "margin": float(margin),
                    "eligible_rows": int(prepared["eligible"].sum()),
                    "selected_rows": int(selected.sum()),
                }
                if best is None or key < best[0]:
                    best = (key, config, result)
    if best is None:
        raise RuntimeError("no train-feasible CTC selector configuration")
    return {"config": best[1], "metrics": best[2]}, audit


def changed_rows(
    rows: list[Row], prepared: dict[str, Any], selected: np.ndarray, deltas: np.ndarray
) -> list[dict[str, Any]]:
    output = []
    for index in np.flatnonzero(selected).tolist():
        row = rows[index]
        base = prepared["base"][index]
        alternate = prepared["alternate"][index]
        output.append({
            "source": row.source,
            "item_id": row.item_id,
            "reference": list(row.reference),
            "base": list(base),
            "alternate": list(alternate),
            "score_delta": float(deltas[index]),
            "edit_change": (
                edit_distance(list(row.reference), list(alternate))
                - edit_distance(list(row.reference), list(base))
            ),
        })
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    primary, _ = load_stage2_context_adapted(args.primary)
    specialist = load_specialist(args.specialist)
    if primary.base.config.to_dict() != specialist.config.to_dict():
        raise ValueError("primary and specialist model configurations differ")
    primary.to(device)
    specialist.to(device)

    train_loader = make_loader([
        RealPhraseDataset(args.phrase_root, "train"),
        RealPhraseDataset(args.context_train_root, "train"),
    ], args.batch_size)
    validation_loader = make_loader([
        RealPhraseDataset(args.phrase_root, "validation"),
        RealPhraseDataset(args.context_validation_root, "validation"),
    ], args.batch_size)
    train_rows = collect_rows(primary, specialist, train_loader, device)
    validation_rows = collect_rows(primary, specialist, validation_loader, device)

    if args.fixed_margin is None:
        selected_train, search_audit = select_on_train(
            train_rows, args.blend_weight, args.blank_bias, args.candidate_mode
        )
        config = selected_train["config"]
        selection_data = "train roles only"
    else:
        provisional = prepare(
            train_rows, args.blend_weight, args.blank_bias, "none",
            args.candidate_mode,
        )
        _, provisional_selected, _ = apply_selector(
            provisional, 0.0, args.fixed_margin
        )
        config = {
            "normalization": "none",
            "primary_score_weight": 0.0,
            "specialist_score_weight": 1.0,
            "margin": float(args.fixed_margin),
            "eligible_rows": int(provisional["eligible"].sum()),
            "selected_rows": int(provisional_selected.sum()),
        }
        search_audit = {
            "configurations": 1,
            "feasible_configurations": None,
            "selection": "predeclared specialist likelihood-ratio rule",
        }
        selection_data = "none; fixed before scoring train or validation labels"
    train_prepared = prepare(
        train_rows, args.blend_weight, args.blank_bias, config["normalization"],
        args.candidate_mode,
    )
    train_hypotheses, train_selected, train_deltas = apply_selector(
        train_prepared, config["primary_score_weight"], config["margin"]
    )
    validation_prepared = prepare(
        validation_rows, args.blend_weight, args.blank_bias, config["normalization"],
        args.candidate_mode,
    )
    validation_hypotheses, validation_selected, validation_deltas = apply_selector(
        validation_prepared, config["primary_score_weight"], config["margin"]
    )

    report = {
        "format": "stage2_general_ctc_selector_experiment_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design": {
            "phrase_identity_used": False,
            "rule": (
                "replace calibrated-primary greedy sequence with specialist greedy "
                "sequence only for a different candidate allowed by candidate_mode "
                "whose weighted full-CTC log-probability advantage clears a frozen margin"
            ),
            "selection_data": selection_data,
            "validation_scored_after_selection": True,
            "blend_weight": args.blend_weight,
            "blank_bias": args.blank_bias,
            "candidate_mode": args.candidate_mode,
            **config,
        },
        "search_audit": search_audit,
        "primary": args.primary.as_posix(),
        "primary_sha256": sha256(args.primary),
        "specialist": args.specialist.as_posix(),
        "specialist_sha256": sha256(args.specialist),
        "train": {
            "rows": len(train_rows),
            "base_metrics": metrics(train_rows, train_prepared["base"]),
            "selected_metrics": metrics(train_rows, train_hypotheses),
            "changes": changed_rows(
                train_rows, train_prepared, train_selected, train_deltas
            ),
        },
        "validation": {
            "rows": len(validation_rows),
            "base_metrics": metrics(validation_rows, validation_prepared["base"]),
            "selected_metrics": metrics(validation_rows, validation_hypotheses),
            "changes": changed_rows(
                validation_rows, validation_prepared, validation_selected,
                validation_deltas,
            ),
        },
        "limitations": [
            "The base 90/10 logit blend and blank bias came from prior development exploration.",
            "These are development validations, not an independent signer/capture set.",
            "Recognition accuracy does not establish visually natural coarticulation.",
        ],
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "consumed_rit_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary", type=Path,
        default=Path(
            "artifacts/models/stage2_v17_multivoice_transfer_context_adapted_v3/model.pth"
        ),
    )
    parser.add_argument(
        "--specialist", type=Path,
        default=Path("artifacts/models/stage2_v17_signer_voice_ctc_pilot_v1/best_model.pth"),
    )
    parser.add_argument(
        "--phrase-root", type=Path,
        default=Path("data/local/stage2_v17_frozen_features"),
    )
    parser.add_argument(
        "--context-train-root", type=Path,
        default=Path("data/local/stage2_v17_asllrp_segmented_train_frozen_features"),
    )
    parser.add_argument(
        "--context-validation-root", type=Path,
        default=Path("data/local/stage2_v17_asllrp_segmented_validation_frozen_features"),
    )
    parser.add_argument("--blend-weight", type=float, default=0.10)
    parser.add_argument("--blank-bias", type=float, default=0.30)
    parser.add_argument("--fixed-margin", type=float)
    parser.add_argument(
        "--candidate-mode",
        choices=(
            "equal_length", "nonshortening_one", "equal_or_prefix_extension",
        ),
        default="equal_or_prefix_extension",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--report", type=Path,
        default=Path(
            "artifacts/reports/stage2_v17_general_ctc_selector_experiment.json"
        ),
    )
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps({
        "design": report["design"],
        "train": report["train"]["selected_metrics"],
        "validation": report["validation"]["selected_metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
