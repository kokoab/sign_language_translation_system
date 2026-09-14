#!/usr/bin/env python3
"""Reduce the all-window audit to duration, exposure, and signer-coverage evidence."""
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path

REPORT = Path(__file__).resolve().parent


def duration_bin(value):
    seconds = float(value)
    if seconds < .20:
        return "under_0.20"
    if seconds < .27:
        return "0.20_to_0.27"
    if seconds < .40:
        return "0.27_to_0.40"
    if seconds < .53:
        return "0.40_to_0.53"
    return "at_least_0.53"


def metrics(rows):
    return {
        "samples": len(rows),
        "epoch12_full_accuracy": sum(row["prediction_o5s5_epoch_12"] == row["target"] for row in rows) / len(rows),
        "correct_in_any_candidate_epoch_fraction": sum(row["correct_any_candidate_epoch"] == "True" for row in rows) / len(rows),
        "mean_foreground_fraction": sum(float(row["foreground_fraction"]) for row in rows) / len(rows),
    }


def main():
    with (REPORT / "all_window_predictions.csv").open(encoding="utf-8") as stream:
        windows = list(csv.DictReader(stream))
    with (REPORT / "event_durations.csv").open(encoding="utf-8") as stream:
        events = list(csv.DictReader(stream))

    by_duration = {}
    for key in sorted({(row["role"], row["source"], duration_bin(row["target_seconds"]))
                       for row in windows if row["category"] == "context"}):
        selected = [row for row in windows if row["category"] == "context"
                    and (row["role"], row["source"], duration_bin(row["target_seconds"])) == key]
        by_duration["/".join(key)] = metrics(selected)

    by_window = {}
    for key in sorted({(row["role"], row["source"], f'{float(row["window_seconds"]):.2f}')
                       for row in windows if row["category"] == "context"}):
        selected = [row for row in windows if row["category"] == "context"
                    and (row["role"], row["source"], f'{float(row["window_seconds"]):.2f}') == key]
        by_window["/".join(key)] = metrics(selected)

    by_exposure = {}
    for source in sorted({row["source"] for row in windows if row["role"] == "train"}):
        for seen in (False, True):
            selected = [row for row in windows if row["role"] == "train"
                        and row["source"] == source and row["category"] == "context"
                        and (int(row["training_draws"]) > 0) == seen]
            if selected:
                by_exposure[f'{source}/{"seen" if seen else "never_drawn"}'] = metrics(selected)

    train_signers = defaultdict(set)
    for event in events:
        if event["role"] == "train" and event["source"] == "o5s5":
            train_signers[event["label"]].add(event["signer"])
    lg_labels = {row["target_label"] for row in windows
                 if row["role"] == "validation" and row["source"] == "o5s5"}
    signer_coverage = {
        "train_classes": len(train_signers), "lg_classes": len(lg_labels),
        "lg_classes_seen_in_o5s5_train": sum(label in train_signers for label in lg_labels),
        "lg_classes_missing_from_o5s5_train": sorted(label for label in lg_labels if label not in train_signers),
        "training_classes_by_distinct_o5s5_signers": dict(sorted(Counter(len(value) for value in train_signers.values()).items())),
        "lg_classes_by_distinct_o5s5_training_signers": dict(sorted(Counter(len(train_signers[label]) for label in lg_labels).items())),
    }
    assert sum(signer_coverage["training_classes_by_distinct_o5s5_signers"].values()) == 49
    assert signer_coverage["lg_classes"] == 27
    output = {
        "duration_bins": by_duration, "window_durations": by_window,
        "training_exposure": by_exposure, "o5s5_signer_coverage": signer_coverage,
        "protected_test_accessed": False,
    }
    (REPORT / "diagnostic_breakdown.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(signer_coverage, indent=2))


if __name__ == "__main__":
    main()
