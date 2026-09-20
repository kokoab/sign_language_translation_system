#!/usr/bin/env python3
"""Train and evaluate one frozen-encoder, utterance-complete CTC head."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from active.v17.finish_ctc_v17 import FinishCTCHead, ctc_loss, decode, error_rate
from active.v17.train_stage_2_other_ctc_v17 import _edit_operations

PRIOR = ROOT / "artifacts/reports/joint_ctc_v17_20260914"
ALIGNED = ROOT / "artifacts/reports/joint_ctc_aligned_v17_20260915"
SOURCE_CACHE = ROOT / "artifacts/generated/joint_ctc_v17_20260914/data.pt"
EVIDENCE_CACHE = ROOT / "artifacts/generated/finish_ctc_v17_20260920/evidence.pt"
MODEL = ROOT / "artifacts/models/finish_ctc_v17_20260920/final.pth"
DEVICE = torch.device("mps")
SEED = 17201
EPOCHS = 20
BATCH_SIZE = 16


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(name: str, value) -> None:
    path = HERE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def emit(value) -> None:
    print(json.dumps(value), flush=True)


def prior_module():
    path = PRIOR / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("joint_ctc_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def encode_windows(base, features: np.ndarray) -> torch.Tensor:
    output = []
    for begin in range(0, len(features), 64):
        values = torch.as_tensor(features[begin:begin + 64], device=DEVICE)
        encoded, _ = base.encode(values)
        output.append(torch.cat((encoded, base.classifier(encoded)), -1).cpu())
    return torch.cat(output).contiguous()


def encode_chunk_rows(base, rows) -> list[torch.Tensor]:
    all_chunks = np.concatenate([row["value"].features for row in rows])
    encoded = encode_windows(base, all_chunks)
    result, cursor = [], 0
    for row in rows:
        parts = []
        for keep in row["value"].keep:
            parts.append(encoded[cursor, torch.as_tensor(keep)])
            cursor += 1
        result.append(torch.cat(parts))
    assert cursor == len(encoded)
    return result


def make_samples(evidence, targets, group, identities):
    return [dict(evidence=value, target=tuple(target), group=group, identity=identity)
            for value, target, identity in zip(evidence, targets, identities)]


def prepare() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for this experiment")
    old = prior_module()
    audit = json.loads((PRIOR / "data_audit.json").read_text())
    if sha(SOURCE_CACHE) != audit["cache_sha256"]:
        raise RuntimeError("frozen source cache changed")
    data = torch.load(SOURCE_CACHE, map_location="cpu", weights_only=False)
    base = old.fresh().base.to(DEVICE).eval()

    train_sequences = [row for row in data["sequences"] if row["role"] == "train"]
    validation_sequences = data["evaluation"]
    sequence_evidence = encode_chunk_rows(base, train_sequences)
    evaluation_evidence = encode_chunk_rows(base, validation_sequences)

    replay_evidence = {role: encode_windows(base, value["features"])
                       for role, value in data["replay"].items()}
    core_evidence = {}
    for role in ("train", "validation"):
        rows = [row for row in data["cores"] if row["role"] == role]
        core_evidence[role] = encode_windows(base, np.stack([row["features"] for row in rows]))
    background_evidence = {role: encode_windows(base, value["features"])
                           for role, value in data["background"].items()}

    train = make_samples(sequence_evidence, [row["targets"] for row in train_sequences],
                         "continuous", [row["identity"] for row in train_sequences])
    replay = data["replay"]["train"]
    train += make_samples(replay_evidence["train"], [(int(t) + 1,) for t in replay["targets"]],
                          "isolated", replay["identities"])
    cores = [row for row in data["cores"] if row["role"] == "train"]
    train += make_samples(core_evidence["train"], [(row["target"] + 1,) for row in cores],
                          "positive_core", [row["identity"] for row in cores])
    background = data["background"]["train"]
    train += make_samples(background_evidence["train"], [()] * len(background["identities"]),
                          "verified_blank", background["identities"])

    prepared = dict(
        train=train,
        evaluation=[dict(evidence=value, row=row["row"])
                    for value, row in zip(evaluation_evidence, validation_sequences)],
        validation_replay=dict(evidence=replay_evidence["validation"],
                               targets=data["replay"]["validation"]["targets"],
                               sources=data["replay"]["validation"]["sources"]),
        validation_cores=dict(evidence=core_evidence["validation"],
                              targets=[row["target"] for row in data["cores"]
                                       if row["role"] == "validation"]),
        validation_background=background_evidence["validation"],
        labels=data["labels"],
    )
    EVIDENCE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = EVIDENCE_CACHE.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        torch.save(prepared, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(EVIDENCE_CACHE)
    save("data_audit.json", dict(
        source_cache_sha256=sha(SOURCE_CACHE), evidence_cache_sha256=sha(EVIDENCE_CACHE),
        stage1_checkpoint_sha256=audit["base_sha256"],
        code_sha256={str(path.relative_to(ROOT)):sha(path) for path in (
            Path(__file__), ROOT / "active/v17/finish_ctc_v17.py",
            ROOT / "test/test_finish_ctc_v17.py")},
        counts=dict(Counter(sample["group"] for sample in train)),
        evaluation_recordings=len(prepared["evaluation"]),
        train_validation_signer_disjoint=True, protected_test_accessed=False,
    ))
    emit({"prepared": True, "counts": Counter(sample["group"] for sample in train)})


def padded(samples, device=DEVICE):
    lengths = [len(sample["evidence"]) for sample in samples]
    values = nn.utils.rnn.pad_sequence([sample["evidence"] for sample in samples], batch_first=True)
    return values.to(device), lengths, [sample["target"] for sample in samples]


def batches(samples, epoch):
    order = np.random.default_rng(SEED + epoch).permutation(len(samples)).tolist()
    order.sort(key=lambda index: len(samples[index]["evidence"]))
    grouped = [order[begin:begin + BATCH_SIZE] for begin in range(0, len(order), BATCH_SIZE)]
    np.random.default_rng(SEED + epoch + 1000).shuffle(grouped)
    assert sorted(index for batch in grouped for index in batch) == list(range(len(samples)))
    return grouped


@torch.inference_mode()
def score_samples(model, samples):
    model.eval()
    edits = Counter()
    exact = 0
    for begin in range(0, len(samples), BATCH_SIZE):
        batch = samples[begin:begin + BATCH_SIZE]
        values, lengths, _ = padded(batch)
        paths = model(values, lengths).argmax(-1).cpu().tolist()
        for sample, path, length in zip(batch, paths, lengths):
            predicted = tuple(decode(path[:length]))
            reference = tuple(value for value in sample["target"] if value <= 100)
            exact += predicted == reference
            edits.update(item["operation"] for item in _edit_operations(reference, predicted))
    total = sum(sum(value <= 100 for value in sample["target"]) for sample in samples)
    return dict(exact=exact, samples=len(samples), known_tokens=total,
                wer_percent=error_rate(edits, total), operations=dict(edits))


def train() -> None:
    audit = json.loads((HERE / "data_audit.json").read_text())
    if sha(EVIDENCE_CACHE) != audit["evidence_cache_sha256"]:
        raise RuntimeError("prepared evidence changed")
    data = torch.load(EVIDENCE_CACHE, map_location="cpu", weights_only=False)
    torch.manual_seed(SEED)
    model = FinishCTCHead().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        started = time.perf_counter()
        schedule = batches(data["train"], epoch)
        for indices in schedule:
            batch = [data["train"][index] for index in indices]
            values, lengths, targets = padded(batch)
            optimizer.zero_grad(set_to_none=True)
            loss = ctc_loss(model(values, lengths), targets, lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        metrics = score_samples(model, [sample for sample in data["train"]
                                        if sample["group"] == "continuous"])
        record = dict(epoch=epoch, loss=total_loss / len(schedule), seconds=time.perf_counter()-started,
                      continuous_train=metrics,
                      coverage=dict(Counter(data["train"][i]["group"] for batch in schedule for i in batch)))
        history.append(record)
        save("training_history.json", history)
        emit(record)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    package = dict(format="slt_finish_ctc_v17", seed=SEED, epoch=EPOCHS,
                   state_dict={key:value.cpu() for key,value in model.state_dict().items()},
                   label_to_index=data["labels"], source_cache_sha256=audit["source_cache_sha256"],
                   evidence_cache_sha256=audit["evidence_cache_sha256"],
                   public_gloss_count=100, blank_index=0, unknown_index=101)
    temporary = MODEL.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        torch.save(package, handle); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(MODEL)


def source_metrics(rows):
    output = {}
    for source in sorted({row["source"] for row in rows}):
        selected = [row for row in rows if row["source"] == source]
        operations = Counter()
        for row in selected:
            operations.update(row["operations"])
        tokens = sum(len(row["reference"]) for row in selected)
        output[source] = dict(samples=len(selected), reference_tokens=tokens,
                              wer_percent=error_rate(operations, tokens),
                              **{name:operations[name] for name in ("substitution", "deletion", "insertion")})
    return output


@torch.inference_mode()
def evaluate() -> None:
    data = torch.load(EVIDENCE_CACHE, map_location="cpu", weights_only=False)
    package = torch.load(MODEL, map_location="cpu", weights_only=False)
    model = FinishCTCHead().to(DEVICE).eval()
    model.load_state_dict(package["state_dict"])
    labels = {index + 1: label for label, index in data["labels"].items()}
    rows = []
    for begin in range(0, len(data["evaluation"]), BATCH_SIZE):
        batch = data["evaluation"][begin:begin + BATCH_SIZE]
        samples = [dict(evidence=item["evidence"], target=()) for item in batch]
        values, lengths, _ = padded(samples)
        paths = model(values, lengths).argmax(-1).cpu().tolist()
        for item, path, length in zip(batch, paths, lengths):
            row = item["row"]
            final = [labels[value] for value in decode(path[:length])]
            operations = Counter(value["operation"] for value in _edit_operations(row["reference"], final))
            rows.append(dict(item_id=row["source_item_id"], source=row["source"],
                             reference=row["reference"], final=final, operations=dict(operations)))

    replay = data["validation_replay"]
    replay_samples = make_samples(replay["evidence"], [(int(t)+1,) for t in replay["targets"]],
                                  "validation_isolated", [str(i) for i in range(len(replay["targets"]))])
    replay_rows = []
    for source in sorted(set(replay["sources"])):
        chosen = [sample for sample, value in zip(replay_samples, replay["sources"]) if value == source]
        replay_rows.append((source, score_samples(model, chosen)))
    cores = data["validation_cores"]
    core_samples = make_samples(cores["evidence"], [(int(t)+1,) for t in cores["targets"]],
                                "validation_core", [str(i) for i in range(len(cores["targets"]))])
    background_samples = make_samples(data["validation_background"],
                                      [()] * len(data["validation_background"]),
                                      "validation_blank", [str(i) for i in range(len(data["validation_background"]))])
    background = score_samples(model, background_samples)
    metrics = source_metrics(rows)
    adjacent_predictions = sum(sum(a == b for a, b in zip(row["final"], row["final"][1:])) for row in rows)
    adjacent_references = sum(sum(a == b for a, b in zip(row["reference"], row["reference"][1:])) for row in rows)
    prior = json.loads((ALIGNED / "evaluation.json").read_text())
    result = dict(
        checkpoint=str(MODEL.relative_to(ROOT)), checkpoint_sha256=sha(MODEL),
        architecture="frozen Stage-1 frame evidence -> one bidirectional GRU -> CTC",
        parameter_count=sum(value.numel() for value in model.parameters()),
        public_gloss_count=100, hidden_states=["blank", "UNKNOWN"], metrics=metrics, rows=rows,
        isolated_ctc=dict(replay_rows), core_ctc=score_samples(model, core_samples),
        verified_blank=background, adjacent_duplicates=dict(reference=adjacent_references,
                                                              prediction=adjacent_predictions),
        repaired_causal_comparator=prior["metrics"], protected_test_accessed=False,
        citizen_test_accessed=False, runtime_promoted=False,
    )
    save("evaluation.json", result)
    connected = metrics.get("asllrp_other_ctc", {})
    familiar = metrics.get("local_phrases", {})
    lines = [
        "# Finish-time bounded CTC experiment", "",
        "This experiment uses the frozen Stage-1 encoder and one bidirectional GRU CTC head.",
        "The visible alphabet is exactly 100 glosses; blank and UNKNOWN are internal only.", "",
        "## Results", "",
        "| Measure | Finish CTC | Repaired causal CTC |", "| --- | ---: | ---: |",
        f"| Connected WER | {connected.get('wer_percent', float('nan')):.2f}% | {prior['metrics']['asllrp_other_ctc']['wer_percent']:.2f}% |",
        f"| Familiar WER | {familiar.get('wer_percent', float('nan')):.2f}% | {prior['metrics']['local_phrases']['wer_percent']:.2f}% |",
        f"| Predicted adjacent duplicates | {adjacent_predictions} | not recomputed |",
        f"| Verified blank false emissions | {background['samples']-background['exact']} / {background['samples']} | {prior['transition_false_emissions']} / 16 |",
        "", "Full machine-readable predictions and edit counts are in `evaluation.json`.",
        "No protected test data was accessed and no runtime was promoted.", "",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines))
    emit({"evaluation_complete": True, "connected": connected, "familiar": familiar})


def notify(message: str) -> None:
    subprocess.run(["osascript", "-e",
                    'on run argv\ndisplay notification (item 1 of argv) with title "SLT experiment"\nend run',
                    message], capture_output=True, timeout=15, check=False)


def worker() -> None:
    started = datetime.now(timezone.utc).isoformat()
    status = "failed"
    try:
        prepare(); train(); evaluate()
        status = "completed"
    except BaseException:
        (HERE / "FAILURE.md").write_text("# Finish CTC experiment failed\n\n```text\n" + traceback.format_exc() + "```\n")
        traceback.print_exc()
    finally:
        save("completion.json", dict(status=status, started_at=started,
                                     finished_at=datetime.now(timezone.utc).isoformat()))
        notify("Finish-time CTC experiment " + status + ". See " + str(HERE))
    if status != "completed":
        raise SystemExit(1)


def launch() -> None:
    if (HERE / "launch.json").exists():
        raise RuntimeError("experiment already launched")
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / "process.log").open("ab") as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    save("launch.json", dict(pid=child.pid, launched_at=datetime.now(timezone.utc).isoformat(),
                             notification_on_exit=True, polling=False))
    print(f"Launched {child.pid} with one exit notification; no polling.")


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker()
    elif args.launch:
        launch()
    else:
        parser.error("choose --launch")
