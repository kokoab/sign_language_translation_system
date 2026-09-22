#!/usr/bin/env python3
"""Fixed offline decoder comparison for the two familiar-signer CTC heads."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from active.v17.model_unified_streaming_ctc_v17 import load_unified_streaming_head
from scripts.evaluate_previous_ctc_approved_v17 import collapse
from scripts.train_local_familiar_ctc_v17 import (
    RECIPE_PATH, digest, load_recipe, load_rows, subsets,
)
from scripts.train_youtube_motion_pilot_v17 import align_tokens

REPORT = ROOT / "artifacts/reports/familiar_decoder_v17_20260922"
SEEDS = (17521, 17522)
BEAM_WIDTH = 8
LM_WEIGHT = 0.1
NONBLANK = tuple(range(1, 102))  # The 100 locked glosses plus explicit OTHER.
NEG_INF = float("-inf")


def logadd(*values: float) -> float:
    maximum = max(values)
    if maximum == NEG_INF:
        return maximum
    return maximum + float(np.log(sum(np.exp(value - maximum) for value in values)))


def prefix_beam(log_probs: np.ndarray, *, width: int, bigram: np.ndarray | None = None,
                lm_weight: float = 0.0) -> list[int]:
    """Exact-vocabulary CTC prefix beam; the optional prior only scores extensions."""
    if log_probs.ndim != 2 or log_probs.shape[1] != 102:
        raise ValueError("expected [time, 102] CTC log probabilities")
    if width < 1 or lm_weight < 0 or (bigram is None) != (lm_weight == 0):
        raise ValueError("inconsistent beam/prior configuration")
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, NEG_INF)}
    for row in log_probs:
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        def add(prefix: tuple[int, ...], blank=NEG_INF, nonblank=NEG_INF) -> None:
            previous = next_beams.get(prefix, (NEG_INF, NEG_INF))
            next_beams[prefix] = (logadd(previous[0], blank), logadd(previous[1], nonblank))

        for prefix, (pb, pnb) in beams.items():
            total = logadd(pb, pnb)
            add(prefix, blank=total + float(row[0]))
            for token in NONBLANK:
                score = float(row[token])
                language = 0.0 if not prefix or bigram is None else lm_weight * float(bigram[prefix[-1], token])
                if prefix and token == prefix[-1]:
                    add(prefix, nonblank=pnb + score)
                    add(prefix + (token,), nonblank=pb + score + language)
                else:
                    add(prefix + (token,), nonblank=total + score + language)
        ranked = sorted(next_beams.items(), key=lambda item: logadd(*item[1]), reverse=True)
        beams = dict(ranked[:width])
    return list(max(beams.items(), key=lambda item: logadd(*item[1]))[0])


def build_bigram(records: list[dict]) -> tuple[np.ndarray, dict]:
    """One vote per distinct phrase transcript; no EOS term can force completion."""
    sequences = {tuple(record["targets"]) for record in records if not record["single"]}
    counts = np.ones((102, 102), dtype=np.float64)  # add-one smoothing for every nonblank label
    counts[:, 0] = 0.0
    for sequence in sequences:
        for left, right in zip(sequence, sequence[1:]):
            counts[left, right] += 1.0
    probabilities = counts / counts[:, 1:].sum(axis=1, keepdims=True)
    probabilities[:, 0] = 0.0
    log_probabilities = np.full((102, 102), NEG_INF, dtype=np.float64)
    log_probabilities[:, 1:] = np.log(probabilities[:, 1:])
    return log_probabilities, {
        "distinct_phrase_transcripts": len(sequences),
        "observed_bigrams": sum(max(0, len(sequence) - 1) for sequence in sequences),
        "smoothing": "add-one over labels 1..101 per predecessor; no BOS/EOS",
    }


def uniform_prior() -> np.ndarray:
    prior = np.full((102, 102), NEG_INF, dtype=np.float64)
    prior[:, 1:] = -np.log(len(NONBLANK))
    return prior


def known(tokens: list[int] | tuple[int, ...]) -> list[int]:
    return [int(token) for token in tokens if int(token) != 101]


def empty_bucket() -> dict:
    return dict(samples=0, exact=0, substitutions=0, deletions=0, insertions=0,
                target_tokens=0, predicted_tokens=0, changed_samples=0,
                changed_improved=0, changed_worsened=0, changed_same=0)


def score(rows: list[dict], hypotheses: dict[str, list[int]], greedy: dict[str, list[int]]) -> dict:
    buckets: dict[str, dict] = defaultdict(empty_bucket)
    for record in rows:
        prediction, baseline, target = hypotheses[record["id"]], greedy[record["id"]], list(record["targets"])
        edits, _ = align_tokens(known(target), known(prediction))
        base_edits, _ = align_tokens(known(target), known(baseline))
        for bucket in (buckets[record["source"]], buckets["all"]):
            bucket["samples"] += 1
            bucket["exact"] += int(prediction == target)
            bucket["target_tokens"] += len(known(target)); bucket["predicted_tokens"] += len(known(prediction))
            for key, value in edits.items(): bucket[key] += value
            changed = prediction != baseline
            bucket["changed_samples"] += int(changed)
            if changed:
                before, after = sum(base_edits.values()), sum(edits.values())
                bucket["changed_improved"] += int(after < before)
                bucket["changed_worsened"] += int(after > before)
                bucket["changed_same"] += int(after == before)
    for bucket in buckets.values():
        bucket["wer"] = (bucket["substitutions"] + bucket["deletions"] + bucket["insertions"]) / max(1, bucket["target_tokens"])
        bucket["exact_accuracy"] = bucket["exact"] / max(1, bucket["samples"])
    return dict(buckets)


def novelty(rows: list[dict], train: list[dict]) -> dict:
    sequences = {tuple(row["targets"]) for row in train if not row["single"]}
    bigrams = {(a, b) for sequence in sequences for a, b in zip(sequence, sequence[1:])}
    output = {}
    for source in ("local_phrases", "asllrp_contiguous"):
        target = [row for row in rows if row["source"] == source]
        references = [tuple(row["targets"]) for row in target]
        target_bigrams = {(a, b) for sequence in references for a, b in zip(sequence, sequence[1:])}
        output[source] = dict(samples=len(target), distinct_target_sequences=len(set(references)),
                              sequence_seen=sum(sequence in sequences for sequence in references),
                              sequence_novel=sum(sequence not in sequences for sequence in references),
                              distinct_target_bigrams=len(target_bigrams),
                              bigram_seen=len(target_bigrams & bigrams),
                              bigram_novel=len(target_bigrams - bigrams))
    return output


def load_head(recipe: dict, seed: int) -> tuple[torch.nn.Module, dict]:
    path = ROOT / recipe["model_dir"] / f"seed_{seed}_familiar.pth"
    expected = json.loads((ROOT / recipe["report_dir"] / "results.json").read_text())["results"][f"{seed}:familiar"]
    if digest(path) != expected["checkpoint_sha256"]:
        raise ValueError(f"candidate hash mismatch: {path}")
    candidate = torch.load(path, map_location="cpu", weights_only=False)
    if candidate.get("format") != "local_familiar_ctc_v17" or candidate.get("seed") != seed or candidate.get("arm") != "familiar":
        raise ValueError("candidate metadata mismatch")
    original = torch.load(ROOT / recipe["checkpoint"], map_location="cpu", weights_only=False)
    head = load_unified_streaming_head(original, device="cpu")
    head.load_state_dict(candidate["head_state_dict"], strict=True)
    return head.eval(), dict(path=str(path.relative_to(ROOT)), sha256=digest(path), selected_epoch=candidate["selected_epoch"])


def logits_for(head: torch.nn.Module, rows: list[dict], evidence: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    values = {}
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            padded = pad_sequence([evidence[row["id"]].float() for row in batch], batch_first=True)
            output = head(padded).log_softmax(-1).cpu().numpy()
            for row, value in zip(batch, output): values[row["id"]] = value[:len(evidence[row["id"]])]
    return values


def decode(rows: list[dict], logits: dict[str, np.ndarray], bigram: np.ndarray | None,
           weight: float) -> tuple[dict[str, list[int]], dict]:
    output = {}; started = time.perf_counter(); steps = 0
    for row in rows:
        value = logits[row["id"]]; steps += len(value)
        output[row["id"]] = prefix_beam(value, width=BEAM_WIDTH, bigram=bigram, lm_weight=weight) if bigram is not None else prefix_beam(value, width=BEAM_WIDTH)
    elapsed = time.perf_counter() - started
    return output, dict(seconds=elapsed, milliseconds_per_sample=1000 * elapsed / len(rows), milliseconds_per_step=1000 * elapsed / steps, steps=steps)


def run() -> dict:
    torch.set_num_threads(2)
    recipe = load_recipe()
    packed, _, _ = load_rows(recipe)
    rows = packed["all"]
    _, evaluation = subsets(rows, "familiar")
    familiar_train, _ = subsets(rows, "familiar")
    evidence_path = ROOT / recipe["report_dir"] / "evidence.pt"
    preflight = json.loads((ROOT / recipe["report_dir"] / "preflight.json").read_text())
    if digest(evidence_path) != preflight["evidence_sha256"]:
        raise ValueError("evidence cache hash mismatch")
    evidence = torch.load(evidence_path, map_location="cpu", weights_only=False)
    if len(evidence) != 6421:
        raise ValueError("unexpected evidence count")
    bigram, prior_info = build_bigram(familiar_train)
    result = dict(format="familiar_decoder_v17_20260922", recipe=str(RECIPE_PATH.relative_to(ROOT)), recipe_sha256=digest(RECIPE_PATH), evidence=str(evidence_path.relative_to(ROOT)), evidence_sha256=digest(evidence_path), fixed=dict(beam_width=BEAM_WIDTH, lm_weight=LM_WEIGHT, decoder="CTC prefix beam, 102-token vocabulary"), prior=prior_info, uniform_prior="log(1/101) for every nonblank extension; length-penalty control", evaluation_counts=dict(total=len(evaluation), local_phrases=sum(r["source"] == "local_phrases" for r in evaluation), asllrp_contiguous=sum(r["source"] == "asllrp_contiguous" for r in evaluation), single=sum(r["single"] for r in evaluation)), novelty=novelty(evaluation, familiar_train), candidates={})
    if result["evaluation_counts"] != {"total": 1735, "local_phrases": 60, "asllrp_contiguous": 12, "single": 1663}:
        raise ValueError("fixed evaluation mismatch")
    for seed in SEEDS:
        head, pin = load_head(recipe, seed)
        logits = logits_for(head, evaluation, evidence)
        greedy = {row["id"]: collapse(value.argmax(-1)) for row, value in ((row, logits[row["id"]]) for row in evaluation)}
        beam, beam_timing = decode(evaluation, logits, None, 0.0)
        uniform_beam, uniform_timing = decode(evaluation, logits, uniform_prior(), LM_WEIGHT)
        lm_beam, lm_timing = decode(evaluation, logits, bigram, LM_WEIGHT)
        result["candidates"][str(seed)] = dict(checkpoint=pin, greedy=score(evaluation, greedy, greedy), beam=score(evaluation, beam, greedy), uniform_prior_beam=score(evaluation, uniform_beam, greedy), weak_bigram_beam=score(evaluation, lm_beam, greedy), timing_cpu=dict(beam=beam_timing, uniform_prior_beam=uniform_timing, weak_bigram_beam=lm_timing))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true", required=True); parser.add_argument("--rerun", action="store_true"); args = parser.parse_args()
    if REPORT.exists() and not args.rerun: raise FileExistsError(f"refusing to overwrite {REPORT}")
    REPORT.mkdir(parents=True, exist_ok=args.rerun)
    result = run()
    (REPORT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Familiar decoder comparison", "", "Fixed, offline comparison: greedy CTC, prefix beam width 8, the same beam with a fixed 0.1 uniform prior control, and a fixed 0.1 add-one-smoothed bigram extension score. The bigram prior is trained only on distinct familiar-training phrase transcripts; it has no EOS term or explicit completion rule, though unsupported insertions can still occur.", "", "Known-WER collapses CTC then removes OTHER (101); exactness preserves OTHER. This is a reused development evaluation and does not establish arbitrary live performance.", ""]
    for seed, candidate in result["candidates"].items():
        lines += [f"## Familiar {seed}", "", "| Decoder | Local60 WER | ASLLRP12 WER | All WER | Local exact | Changed / improved / worsened | CPU ms/sample |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, label in (("greedy", "Greedy"), ("beam", "Beam 8"), ("uniform_prior_beam", "Beam 8 + uniform control"), ("weak_bigram_beam", "Beam 8 + weak bigram")):
            metrics = candidate[name]; all_metrics = metrics["all"]; timing = candidate.get("timing_cpu", {}).get(name.replace("weak_bigram_beam", "weak_bigram_beam"), {})
            if name == "greedy": timing = {"milliseconds_per_sample": 0.0}
            lines.append(f"| {label} | {metrics['local_phrases']['wer']:.2%} | {metrics['asllrp_contiguous']['wer']:.2%} | {all_metrics['wer']:.2%} | {metrics['local_phrases']['exact']}/60 | {all_metrics['changed_samples']} / {all_metrics['changed_improved']} / {all_metrics['changed_worsened']} | {timing['milliseconds_per_sample']:.2f} |")
        lines.append("")
    lines += ["## Coverage", "", "```json", json.dumps(result["novelty"], indent=2), "```", "", "The word-pair prior is deliberately weak and was not weight-tuned. The uniform control separates its constant per-token log penalty from learned pair preference. Both priors only score candidates generated from visual CTC extensions; neither adds an EOS completion rule."]
    (REPORT / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__": main()
