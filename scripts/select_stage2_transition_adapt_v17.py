#!/usr/bin/env python3
"""Make the fixed Stage-2 transition-adaptation selection decision offline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


INPUT_SHA256 = "0468e3f491f1872f7be8a915eeb994581801f99006b6bee6e630e2d6f134cda7"
FORMAT = "slt_stage2_transition_adapt_training_v17"
ARMS = ("no_stem", "with_stem")
SEEDS = (1701, 1702)
BASELINE = {"target_edits": 603, "target_tokens": 284, "local_edits": 6,
            "local_tokens": 259, "exact_edits": 9, "exact_tokens": 24,
            "contextual_edits": 43, "contextual_tokens": 254,
            "citizen_correct": 331, "citizen_samples": 378}
GATE_LIMITS = {"target": 542, "local": 6, "exact": 9, "contextual": 43,
               "citizen": 328}
FIXED_DESIGN = {"epochs": 20, "patience": 5, "samples_per_epoch": 1800,
                "batch_size": 16, "projection_epochs": 2, "projection_lr": 0.0001,
                "backbone_lr": 0.000005, "weight_decay": 0.02,
                "gradient_clip": 1.0, "distill_weight": 1.0, "temperature": 2.0}
HISTORY_SHA256 = {("no_stem", 1701): "19a87c0ecc4559b5c45981d164d5bb97e6716611d583a028ecba08978e27c8b8",
                  ("no_stem", 1702): "92465c0efdcb785c6eef70966225358b7a793db9563f7e152a08793fd2407ea7",
                  ("with_stem", 1701): "2400ad491a33c38ac10ddcd8865e93966c0b871059b3a1ab6cc026412b8040e7",
                  ("with_stem", 1702): "06acb5e35b19e72503388ca148531adc1463e956d0b662c88494611138054ab3"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or sha256(path) != expected_sha256:
        raise ValueError(f"pinned input mismatch: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError(f"unexpected report format: {data.get('format')!r}")
    return data


def verify_history_hash(path: Path, arm: str, seed: int) -> None:
    if not path.is_file() or sha256(path) != HISTORY_SHA256[(arm, seed)]:
        raise ValueError(f"history mismatch: {path}")


def assert_matching_initializations(initializations: list[dict[str, Any]]) -> dict[str, Any]:
    if not initializations or any(item != initializations[0] for item in initializations[1:]):
        raise ValueError("epoch-0 initialized candidates differ")
    return initializations[0]


def canonical_initialized(data: dict[str, Any], root: Path) -> dict[str, Any]:
    initializations = []
    for row in data["results"]:
        history_path = root / row["history"]
        verify_history_hash(history_path, row["arm"], row["seed"])
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(history, list) or not history or history[0].get("epoch") != 0 or "initialized_candidate" not in history[0]:
            raise ValueError(f"missing epoch-0 initialized candidate: {history_path}")
        initializations.append(history[0]["initialized_candidate"])
    return assert_matching_initializations(initializations)


def gates(summary: dict[str, Any]) -> dict[str, bool]:
    return {"target": summary["target_edits"] <= GATE_LIMITS["target"],
            "local": summary["local_edits"] <= GATE_LIMITS["local"],
            "exact": summary["exact_edits"] <= GATE_LIMITS["exact"],
            "contextual": summary["contextual_edits"] <= GATE_LIMITS["contextual"],
            "citizen": summary["citizen_correct"] >= GATE_LIMITS["citizen"]}


def arm_summary(records: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    rows = [row for row in records if row["arm"] == arm]
    seed_set = {row["seed"] for row in rows}
    qualified = seed_set == set(SEEDS) and len(rows) == len(SEEDS) and all(
        all(gates(row["validation"]["summary"]).values()) for row in rows)
    return {"arm": arm, "seeds_present": sorted(seed_set), "qualified": qualified,
            "mean_target_wer": (sum(row["validation"]["summary"]["target_edits"] /
                                     row["validation"]["summary"]["target_tokens"] for row in rows) /
                                len(rows) if rows else None)}


def select(records: list[dict[str, Any]]) -> dict[str, Any]:
    arms = [arm_summary(records, arm) for arm in ARMS]
    passing = [arm for arm in arms if arm["qualified"]]
    chosen = min(passing, key=lambda arm: (arm["mean_target_wer"], ARMS.index(arm["arm"]))) if passing else None
    return {"arms": arms, "selected_arm": chosen["arm"] if chosen else None,
            "integration_allowed": bool(chosen),
            "runtime_action": "integrate_candidate" if chosen else "retain_current_selector"}


def edit_operations(reference: list[int], prediction: list[int]) -> dict[str, int]:
    """Levenshtein totals plus the trainer's adjacent-repeat discrepancy."""
    matrix = [[0] * (len(prediction) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1): matrix[i][0] = i
    for j in range(len(prediction) + 1): matrix[0][j] = j
    for i, actual in enumerate(reference, 1):
        for j, predicted in enumerate(prediction, 1):
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1,
                               matrix[i - 1][j - 1] + (actual != predicted))
    i, j = len(reference), len(prediction)
    result = Counter()
    while i or j:
        if i and j and matrix[i][j] == matrix[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1]):
            if reference[i - 1] != prediction[j - 1]: result["substitutions"] += 1
            i, j = i - 1, j - 1
        elif j and matrix[i][j] == matrix[i][j - 1] + 1:
            result["insertions"] += 1
            j -= 1
        else:
            result["deletions"] += 1
            i -= 1
    result["repeated_sign_errors"] = abs(sum(a == b for a, b in zip(reference, reference[1:])) -
                                           sum(a == b for a, b in zip(prediction, prediction[1:])))
    return {key: result[key] for key in ("substitutions", "deletions", "insertions", "repeated_sign_errors")}


def _verify(data: dict[str, Any], root: Path) -> None:
    if data.get("expected_baseline") != BASELINE or data.get("fixed_design") != FIXED_DESIGN:
        raise ValueError("baseline or fixed design changed")
    if any(data.get(flag) for flag in ("citizen_test_accessed", "semlex_test_accessed", "local_test_accessed",
                                       "rit_external_evaluation_accessed", "test_evaluated")):
        raise ValueError("protected test access recorded")
    for path_key, sha_key in (("experiment_manifest", "experiment_manifest_sha256"),
                              ("frozen_inputs", "frozen_inputs_sha256"), ("warm_start", "warm_start_sha256"),
                              ("teacher", "teacher_sha256")):
        path = root / data[path_key]
        if not path.is_file() or sha256(path) != data[sha_key]:
            raise ValueError(f"provenance mismatch: {path_key}")
    if {(row["arm"], row["seed"]) for row in data["results"]} != {(a, s) for a in ARMS for s in SEEDS}:
        raise ValueError("results must contain exactly two arms and fixed seeds")
    for row in data["results"]:
        for key, expected in (("checkpoint", row["checkpoint_sha256"]),):
            path = root / row[key]
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"artifact mismatch: {path}")
        verify_history_hash(root / row["history"], row["arm"], row["seed"])


def _example_row(domain: str, example: dict[str, Any], model: str, checkpoint: str | None = None,
                 arm: str | None = None, seed: int | None = None, epoch: int | None = None) -> dict[str, Any]:
    reference, prediction = example["reference"], example["prediction"]
    ops = edit_operations(reference, prediction)
    positions = (["first"] if len(reference) == 1 else
                 ["first" if index == 0 else "last" if index == len(reference) - 1 else "middle"
                  for index in range(len(reference))])
    return {"model": model, "snapshot": model, "arm": arm, "seed": seed, "epoch": epoch,
            "checkpoint": checkpoint, "domain": domain, "item_id": example["item_id"],
            "source": example.get("source"), "reference": reference, "prediction": prediction,
            **ops, "exact_sequence": reference == prediction, "reference_positions": positions,
            "duration_windows": example.get("windows"), "duration_bucket": f"{example.get('windows', 0)}_windows"}


def _write_outputs(data: dict[str, Any], input_path: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    decision = select(data["results"])
    rows = []
    for result in data["results"]:
        summary = result["validation"]["summary"]
        for gate, actual_key, threshold in (("target", "target_edits", 542), ("local", "local_edits", 6),
                                            ("exact", "exact_edits", 9), ("contextual", "contextual_edits", 43),
                                            ("citizen", "citizen_correct", 328)):
            rows.append({"arm": result["arm"], "seed": result["seed"], "gate": gate,
                         "actual": summary[actual_key], "threshold": threshold,
                         "pass": gates(summary)[gate], "checkpoint": result["checkpoint"],
                         "checkpoint_sha256": result["checkpoint_sha256"]})
    best = min(data["results"], key=lambda row: (row["validation"]["summary"]["target_edits"],
                                                   ARMS.index(row["arm"]), row["seed"]))
    initialized = canonical_initialized(data, Path.cwd())
    decision.update({"format": "slt_stage2_transition_selection_v17", "input": str(input_path),
                     "input_sha256": sha256(input_path), "baseline": BASELINE,
                     "per_seed": [{"arm": row["arm"], "seed": row["seed"],
                                   "metrics": row["validation"]["summary"], "gates": gates(row["validation"]["summary"]),
                                   "checkpoint": row["checkpoint"], "checkpoint_sha256": row["checkpoint_sha256"],
                                   "history": row["history"], "history_sha256": sha256(Path.cwd() / row["history"])}
                                  for row in data["results"]],
                     "best_candidate_for_comparison": {"arm": best["arm"], "seed": best["seed"],
                                                       "checkpoint": best["checkpoint"]},
                     "initialized_candidate": {"origin_arm": "no_stem", "origin_seed": 1701,
                                               "epoch": 0, "history_sha256": HISTORY_SHA256[("no_stem", 1701)]}})
    (output / "selection.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "failure.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output / "per_example_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for domain, examples in data["baseline"]["examples"].items():
            for example in examples: handle.write(json.dumps(_example_row(domain, example, "baseline_selector"), sort_keys=True) + "\n")
        for domain, examples in initialized["examples"].items():
            for example in examples: handle.write(json.dumps(_example_row(domain, example, "initialized_candidate", data["warm_start"], "no_stem", 1701, 0), sort_keys=True) + "\n")
        for result in data["results"]:
            snapshot = f"best_{result['arm']}_{result['seed']}"
            for domain, examples in result["validation"]["examples"].items():
                for example in examples: handle.write(json.dumps(_example_row(domain, example, snapshot, result["checkpoint"], result["arm"], result["seed"], result["best_epoch"]), sort_keys=True) + "\n")
    _write_readme(data, best, decision, output)
    return decision


def _write_readme(data: dict[str, Any], best: dict[str, Any], decision: dict[str, Any], output: Path) -> None:
    initialized = canonical_initialized(data, Path.cwd())
    lines = ["# Stage-2 transition adaptation: matched failure result", "",
             f"Input SHA-256: `{decision['input_sha256']}`. Selection is **null**; runtime action is `{decision['runtime_action']}`.", "",
             "No arm qualified: every seed improved target-only ASLLRP OTHER WER, but each violated one or more frozen retention gates.", "",
             "| Arm / seed | target WER | OTHER-inclusive WER | local familiar-domain edits | exact edits | contextual edits | Citizen correct | STEM held participants |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    baseline_domains, baseline_summary = data["baseline"]["domains"], data["baseline"]["summary"]
    lines.append(f"| baseline selector | {baseline_domains['target']['wer']:.4f} | {baseline_domains['target_full']['wer']:.4f} | {baseline_summary['local_edits']}/259 | {baseline_summary['exact_edits']}/24 | {baseline_summary['contextual_edits']}/254 | {baseline_summary['citizen_correct']}/378 | {baseline_summary['stem_correct']}/{baseline_summary['stem_samples']} |")
    initialized_summary, initialized_domains = initialized["summary"], initialized["domains"]
    lines.append(f"| initialized candidate / epoch 0 | {initialized_domains['target']['wer']:.4f} | {initialized_domains['target_full']['wer']:.4f} | {initialized_summary['local_edits']}/259 | {initialized_summary['exact_edits']}/24 | {initialized_summary['contextual_edits']}/254 | {initialized_summary['citizen_correct']}/378 | {initialized_summary['stem_correct']}/{initialized_summary['stem_samples']} |")
    for row in data["results"]:
        domains, summary = row["validation"]["domains"], row["validation"]["summary"]
        lines.append(f"| {row['arm']} / {row['seed']} | {domains['target']['wer']:.4f} | {domains['target_full']['wer']:.4f} | {summary['local_edits']}/259 | {summary['exact_edits']}/24 | {summary['contextual_edits']}/254 | {summary['citizen_correct']}/378 | {summary['stem_correct']}/{summary['stem_samples']} |")
    def full_rows(domain: str) -> list[str]:
        rows = [("baseline selector", data["baseline"]["domains"][domain])]
        rows += [("initialized candidate / epoch 0", initialized["domains"][domain])]
        rows += [(f"{row['arm']} / {row['seed']}", row["validation"]["domains"][domain]) for row in data["results"]]
        return [f"| {label} | {metric['edits']}/{metric['tokens']} | {metric['wer']:.4f} | {metric['exact']}/{metric['samples']} | {metric['sequence_accuracy']:.4f} | {metric['substitutions']} | {metric['deletions']} | {metric['insertions']} | {metric['repeated_sign_errors']} |" for label, metric in rows]
    d, baseline_target = best["validation"]["domains"], data["baseline"]["domains"]["target"]
    lines += ["", "## OTHER-inclusive validation details", "",
              "The accepted selector has no OTHER output; its row is descriptive only and is not a comparative baseline or selection gate.", "",
              "| Run | edits/tokens | WER | exact/samples | sequence accuracy | substitutions | deletions | insertions | repeated-sign errors |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", *full_rows("target_full"),
              "", "## STEM held-participant details", "",
              "| Run | edits/tokens | WER | exact/samples | sequence accuracy | substitutions | deletions | insertions | repeated-sign errors |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", *full_rows("stem"),
              "", f"Best unqualified checkpoint for comparison: `{best['checkpoint']}` ({best['checkpoint_sha256']}).",
              f"Target-only by-position aggregates, baseline: `{json.dumps(baseline_target['position_error_buckets'], sort_keys=True)}`; best candidate: `{json.dumps(d['target']['position_error_buckets'], sort_keys=True)}`.",
              f"Target-only by-duration aggregates, baseline: `{json.dumps(baseline_target['duration_error_buckets'], sort_keys=True)}`; best candidate: `{json.dumps(d['target']['duration_error_buckets'], sort_keys=True)}`.",
              "", "Frozen gates: target <=542 edits (10% relative from 603), local <=6, exact <=9, contextual <=43, Citizen >=328/378. Local phrase is familiar-domain retention, not a generalization claim.",
              "", f"Provenance: manifest `{data['experiment_manifest_sha256']}`, frozen inputs `{data['frozen_inputs_sha256']}`, warm start `{data['warm_start_sha256']}`, teacher `{data['teacher_sha256']}`.",
              "", "Reproducible rollback launch (accepted selector, unchanged):",
              "```sh", "venv/bin/python scripts/live_reel_continuous_v17.py --camera 0", "```",
              "", "Protected-test flags are all false. This result does not establish iPhone performance or general continuous-ASL accuracy; no further sweep was launched."]
    lines.insert(4, "The per-example evidence bundle covers six snapshots over all seven domains: baseline selector, the epoch-0 initialized candidate (identical across all four histories), and all four best seed checkpoints (7,272 rows).")
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/reports/stage2_v17_transition_adapt_v2/training_result.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/stage2_v17_transition_adapt_v2"))
    args = parser.parse_args()
    data = load_pinned(args.input, INPUT_SHA256)
    _verify(data, Path.cwd())
    decision = _write_outputs(data, args.input, args.output)
    print(json.dumps({"selection": decision["selected_arm"], "runtime_action": decision["runtime_action"]}, sort_keys=True))


if __name__ == "__main__":
    main()
