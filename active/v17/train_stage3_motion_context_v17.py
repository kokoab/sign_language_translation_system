"""Fit contextual motion scoring and measure corrections on real development clips."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from active.v17.continuous_decode_v17 import CTCPrefixDecoder
from active.v17.continuous_evidence_v17 import ContinuousConfig, ContinuousEvidenceModel
from active.v17.stage3_motion_context_v17 import MotionContextScorer, mismatched_motion_indices
from active.v17.train_continuous_evidence_v17 import sha256, source_group
from active.v17.train_streaming_tcn_ctc_v17 import edit_distance, refuse_protected


def collate_context(rows, eos_index):
    lengths = torch.tensor([len(r["evidence"]) for r in rows])
    width = max(len(r["targets"]) for r in rows) + 1
    evidence = torch.zeros(len(rows), int(lengths.max()), rows[0]["evidence"].shape[-1])
    tokens = torch.zeros(len(rows), width, dtype=torch.long)
    targets = torch.full_like(tokens, -100)
    for i, row in enumerate(rows):
        evidence[i, :len(row["evidence"])] = row["evidence"].float()
        candidate = tuple(row["targets"])
        tokens[i, 1:len(candidate) + 1] = torch.tensor(candidate)
        targets[i, :len(candidate) + 1] = torch.tensor((*candidate, eos_index))
    return evidence, lengths, tokens, targets


@torch.inference_mode()
def loss_on(model, rows, batch_size):
    model.eval(); loss, count = 0., 0
    for start in range(0, len(rows), batch_size):
        x, lengths, tokens, targets = collate_context(rows[start:start + batch_size], model.eos_index)
        logits = model(x, lengths, tokens)
        loss += float(F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100, reduction="sum"))
        count += int((targets != -100).sum())
    return loss / count


@torch.inference_mode()
def recognition_candidates(recognizer, rows):
    output = []
    for i, row in enumerate(rows):
        logits = recognizer(row["evidence"][None].float())[0].log_softmax(-1).numpy()
        beam = CTCPrefixDecoder(beam_width=8, token_topk=12)
        for step in logits:
            beam.step(step)
        output.append(beam.alternatives())
        if i % 200 == 0:
            print(json.dumps({"candidate_rows": i, "total": len(rows)}), flush=True)
    return output


@torch.inference_mode()
def correction_report(model, rows, candidates, weight, *, shuffled=False):
    totals = {}
    controls = mismatched_motion_indices(rows) if shuffled else list(range(len(rows)))
    for index, (row, beam) in enumerate(zip(rows, candidates)):
        evidence = rows[controls[index]]["evidence"]
        sequences = [prefix for prefix, _ in beam]
        context = model.score_candidates(evidence.float(), sequences).numpy()
        scores = np.asarray([score for _, score in beam]) + weight * context
        before, after, expected = sequences[0], sequences[int(scores.argmax())], tuple(row["targets"])
        old_edits, new_edits = edit_distance(expected, before), edit_distance(expected, after)
        bucket = totals.setdefault(row["source"], dict(samples=0, tokens=0, before_edits=0, after_edits=0,
            before_exact=0, after_exact=0, improved=0, worsened=0, correct_to_wrong=0, oracle_exact=0))
        bucket["samples"] += 1; bucket["tokens"] += len(expected)
        bucket["before_edits"] += old_edits; bucket["after_edits"] += new_edits
        bucket["before_exact"] += int(before == expected); bucket["after_exact"] += int(after == expected)
        bucket["improved"] += int(new_edits < old_edits); bucket["worsened"] += int(new_edits > old_edits)
        bucket["correct_to_wrong"] += int(before == expected and after != expected)
        bucket["oracle_exact"] += int(expected in sequences)
    for bucket in totals.values():
        bucket["before_wer"] = bucket["before_edits"] / max(1, bucket["tokens"])
        bucket["after_wer"] = bucket["after_edits"] / max(1, bucket["tokens"])
    return totals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recognizer", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("artifacts/models/stage3_motion_context_v17_v1"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=17101)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    refuse_protected((args.recognizer,))
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4); torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    checkpoint = torch.load(args.recognizer, map_location="cpu", weights_only=False)
    cache_path = Path(checkpoint["cache"])
    if sha256(cache_path) != checkpoint["cache_sha256"]:
        raise ValueError("recognition evidence cache changed")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    config = ContinuousConfig(**checkpoint["config"])
    recognizer = ContinuousEvidenceModel(config).eval()
    recognizer.load_state_dict(checkpoint["model_state_dict"], strict=True)
    train, validation = cache["train"], cache["validation"]
    model = MotionContextScorer(evidence_dim=config.input_dim, num_glosses=config.num_glosses)
    counts = Counter(source_group(r) for r in train)
    # Balance sources without a phrase table at inference.
    weights = torch.tensor([1 / counts[source_group(r)] for r in train])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    best, history = None, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.; steps = 0
        indices = torch.multinomial(weights, len(train), replacement=True).tolist()
        for start in range(0, len(indices), args.batch_size):
            rows = [train[i] for i in indices[start:start + args.batch_size]]
            x, lengths, tokens, targets = collate_context(rows, model.eos_index)
            logits = model(x, lengths, tokens)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100, label_smoothing=.05)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
            total += float(loss.detach()); steps += 1
        valid_loss = loss_on(model, validation, args.batch_size)
        row = dict(epoch=epoch, train_loss=total / steps, validation_loss=valid_loss)
        history.append(row); print(json.dumps(row), flush=True)
        if best is None or valid_loss < best["loss"]:
            best = dict(epoch=epoch, loss=valid_loss, state=copy.deepcopy(model.state_dict()))
    model.load_state_dict(best["state"]); model.eval()
    candidates = recognition_candidates(recognizer, validation)
    reports = {}
    for weight in (0., .1, .3, .5, 1.):
        reports[str(weight)] = correction_report(model, validation, candidates, weight)
    def selection(report):
        # Source-balanced continuous WER plus isolated regression; full OTHER tokens count.
        continuous = np.mean([report[s]["after_wer"] for s in ("local_phrases", "asllrp_contiguous", "ncslgr_strict")])
        isolated = np.mean([report[s]["after_wer"] for s in ("isolated:citizen", "isolated:semlex")])
        harmful = sum(r["correct_to_wrong"] for r in report.values()) / len(validation)
        return float(.7 * continuous + .3 * isolated + harmful)
    selected_weight = min(reports, key=lambda key: (selection(reports[key]), float(key)))
    shuffled = correction_report(model, validation, candidates, float(selected_weight), shuffled=True)
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(format="slt_stage3_motion_context_v17", selected_epoch=best["epoch"],
        selected_weight=float(selected_weight), promoted=False,
        validation_by_weight=reports, shuffled_motion_control=shuffled,
        history=history, elapsed_seconds=time.perf_counter() - started,
        parameter_count=sum(p.numel() for p in model.parameters()),
        test_accessed=False, limitations=["development-selected; no independent final evaluation",
        "can only select supplied recognition alternatives", "English renderer is unchanged"])
    torch.save(dict(format="slt_stage3_motion_context_v17", version=1,
        model_state_dict=model.state_dict(), evidence_dim=config.input_dim, num_glosses=config.num_glosses,
        recognizer=str(args.recognizer), recognizer_sha256=sha256(args.recognizer),
        label_to_index=cache["labels"], selected_weight=float(selected_weight)), args.output / "model.pth")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"history", "validation_by_weight", "shuffled_motion_control"}}, indent=2))


if __name__ == "__main__":
    main()
