"""Train continuous observation with matched isolated replay and genuine gaps."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from active.v17.continuous_evidence_v17 import (
    ContinuousConfig, ContinuousEvidenceModel, interval_targets, load_continuous_samples,
    load_isolated_streams, observation_ends, observation_windows,
)
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.train_unified_streaming_ctc_v17 import EvidenceSequence, collate, evaluate
from active.v17.train_streaming_tcn_ctc_v17 import refuse_protected
from active.v17.geometry_v17 import resample_features


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw(args, role, labels):
    samples = load_continuous_samples(args.phrases, role, labels)
    causal_root = getattr(args, "causal_local_root", None)
    if causal_root is not None:
        for index, sample in enumerate(samples):
            if sample.source != "local_phrases":
                continue
            path = causal_root / Path(sample.path).relative_to(args.phrases)
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"]))
                frames = archive["features"].astype(np.float32)
            if (metadata.get("format") != "slt_causal_local_features_v17" or metadata["role"] != role
                    or metadata["identity"] != sample.identity or metadata["signer"] != sample.signer
                    or metadata["source_archive_sha256"] != sha256(sample.path)
                    or tuple(labels[g] + 1 for g in metadata["targets"]) != sample.targets):
                raise ValueError("causal feature provenance mismatch")
            samples[index] = replace(sample, frames=frames, path=str(path))
    if role == "train":
        for sample in list(samples):
            if sample.source != "local_phrases":
                continue
            for factor in getattr(args, "local_duration_factors", ()):
                if not np.isfinite(factor) or factor <= 0 or factor == 1:
                    raise ValueError("duration factors must be positive, finite and different from one")
                samples.append(replace(sample, frames=resample_features(sample.frames, max(4, round(len(sample.frames) * factor))),
                                       identity=f"{sample.identity}:duration:{factor}"))
    roots = (
        (Path("data/local/citizen100_v17/landmarks") / ("train" if role == "train" else "val"), "citizen"),
        (Path("data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17") if role == "train"
         else Path("data/local/semlex_citizen100_val_audit/landmarks_v17"), "semlex"),
    )
    for root, source in roots:
        samples.extend(load_isolated_streams(root, labels, source))
    return samples


def data_policy(args):
    root = getattr(args, "causal_local_root", None)
    return {"causal_local_root": str(root) if root else None,
            "local_duration_factors": list(getattr(args, "local_duration_factors", ()))}


@torch.inference_mode()
def encode_samples(model, raw, config, device, batch_size):
    rows = []
    model.eval().to(device)
    for offset in range(0, len(raw), 16):
        group = raw[offset:offset + 16]
        ends = [observation_ends(len(s.frames), config.stride) for s in group]
        windows = np.concatenate([
            observation_windows(s.frames, end, config.windows)
            for s, endpoints in zip(group, ends) for end in endpoints
        ])
        vectors = []
        for start in range(0, len(windows), batch_size):
            x = torch.from_numpy(windows[start:start + batch_size]).to(device)
            logits, embedding = model(x, return_embeddings=True)
            vectors.append(torch.cat((embedding, logits), -1).half().cpu())
        evidence = torch.cat(vectors).reshape(-1, config.input_dim)
        cursor = 0
        for sample, endpoints in zip(group, ends):
            count = len(endpoints)
            rows.append({
                "evidence": evidence[cursor:cursor + count].clone(),
                "targets": sample.targets, "source": sample.source,
                "identity": sample.identity, "signer": sample.signer,
                "path": sample.path, "frame_count": len(sample.frames),
                "ends": endpoints,
                "alignment": interval_targets(endpoints, sample.intervals, min(config.windows)),
            })
            cursor += count
        if offset % 160 == 0:
            print(json.dumps({"encoded": min(offset + 16, len(raw)), "total": len(raw)}), flush=True)
    return rows


def build_cache(args, config):
    checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    # A head-only signer split cannot undo earlier phrase exposure by the encoder.
    if checkpoint.get("phrase_adaptation") or checkpoint.get("asllrp_core_adaptation"):
        raise ValueError("encoder contains historical randomly split local phrase adaptation")
    labels = {str(k): int(v) for k, v in checkpoint["label_to_index"].items()}
    stage1 = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    stage1.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if stage1.config.dim != config.stage1_dim:
        raise ValueError("encoder dimension differs from evidence configuration")
    raw = {role: load_raw(args, role, labels) for role in ("train", "validation")}
    ids = [{s.identity for s in raw[role]} for role in raw]
    signers = [{(s.source.removeprefix("blank:"), s.signer) for s in raw[role] if s.signer} for role in raw]
    if ids[0] & ids[1] or signers[0] & signers[1]:
        raise ValueError("source identity or known signer leaks across development roles")
    paths = sorted({s.path for values in raw.values() for s in values})
    if getattr(args, "causal_local_root", None) is not None:
        originals = [str(args.phrases / Path(p).relative_to(args.causal_local_root))
                     for p in paths if Path(p).is_relative_to(args.causal_local_root)]
        paths = sorted(set(paths + originals))
    result = {
        "format": "slt_continuous_evidence_cache_v17", "version": 1,
        "base": str(args.base), "base_sha256": sha256(args.base),
        "config": config.to_dict(), "labels": labels,
        "data_policy": data_policy(args),
        "source_hashes": {p: sha256(p) for p in paths},
        "observation_policy": "rolling for phrases AND duration-restored isolated clips",
        "blank_policy": "complement of manual/published lexical intervals including OTHER",
        "test_accessed": False,
    }
    for role, samples in raw.items():
        print(json.dumps({"role": role, "sources": dict(Counter(s.source for s in samples))}), flush=True)
        result[role] = encode_samples(stage1, samples, config, args.device, args.embedding_batch_size)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.cache)
    return result


def evidence_sequences(rows):
    return [EvidenceSequence(r["evidence"].float().numpy(), tuple(r["targets"]),
                             r["source"], r["identity"]) for r in rows]


def source_group(row):
    if row["source"].startswith("blank:"):
        return "blank"
    return row["source"]


def metrics(model, samples, args, other_index):
    result = evaluate(model, samples, torch.device(args.device), args.batch_size, other_index)
    # Blank false emission includes OTHER too: every lexical emission is wrong on a true gap.
    with torch.inference_mode():
        blank = [s for s in samples if not s.targets]
        false = 0
        for start in range(0, len(blank), args.batch_size):
            batch = collate(blank[start:start + args.batch_size])
            paths = model(batch["evidence"].to(args.device)).argmax(-1).cpu()
            false += sum(bool((p[:int(n)] != 0).any()) for p, n in zip(paths, batch["lengths"]))
    result["blank_boundaries"]["any_token_false_emission_rate"] = false / max(1, len(blank))
    return result


def run(args):
    refuse_protected((args.phrases, args.base, args.cache))
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == "auto":
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    config = ContinuousConfig(windows=tuple(args.windows), stride=args.stride)
    started = time.perf_counter()
    if args.cache.exists():
        cache = torch.load(args.cache, map_location="cpu", weights_only=False)
        if cache["base_sha256"] != sha256(args.base) or cache["config"] != config.to_dict():
            raise ValueError("cache encoder or observation configuration changed")
        if cache.get("data_policy", {"causal_local_root": None, "local_duration_factors": []}) != data_policy(args):
            raise ValueError("cached camera normalization or duration policy changed")
        if any(sha256(p) != digest for p, digest in cache["source_hashes"].items()):
            raise ValueError("cached source data changed")
    else:
        cache = build_cache(args, config)
    print(json.dumps({"cache_seconds": time.perf_counter() - started}), flush=True)
    if args.cache_only:
        return
    train, validation = evidence_sequences(cache["train"]), evidence_sequences(cache["validation"])
    model = ContinuousEvidenceModel(config).to(args.device)
    counts = Counter(source_group(row) for row in cache["train"])
    # Each real task gets explicit mass; isolated retention cannot be drowned by long OTHER utterances.
    shares = {"isolated:citizen": 0.25, "isolated:semlex": 0.20,
              "local_phrases": 0.25, "asllrp_contiguous": 0.15,
              "ncslgr_strict": 0.10, "blank": 0.05}
    weights = torch.tensor([shares[source_group(r)] / counts[source_group(r)] for r in cache["train"]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    ctc = nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)
    best, history = None, []
    args.output.mkdir(parents=True, exist_ok=False)
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0; batches = 0
        indices = torch.multinomial(weights, args.samples_per_epoch, replacement=True).tolist()
        for start in range(0, len(indices), args.batch_size):
            selected = indices[start:start + args.batch_size]
            batch = collate([train[i] for i in selected])
            logits = model(batch["evidence"].to(args.device))
            # CPU CTC has stable backward on this Apple environment; gradients cross the copy.
            losses = ctc(logits.float().log_softmax(-1).transpose(0, 1).cpu(),
                         batch["targets"], batch["lengths"], batch["target_lengths"])
            valid = torch.isfinite(losses)
            if not valid.all():
                bad = [train[i].identity for i, keep in zip(selected, valid.tolist()) if not keep]
                raise ValueError(f"infeasible/nonfinite CTC sequences: {bad}")
            loss = (losses / batch["target_lengths"].clamp_min(1)).mean().to(args.device)
            aligned = np.full(tuple(logits.shape[:2]), -100, np.int64)
            for row, index in enumerate(selected):
                a = cache["train"][index]["alignment"]
                aligned[row, :len(a)] = a
            if (aligned != -100).any() and args.alignment_weight:
                loss = loss + args.alignment_weight * F.cross_entropy(
                    logits.flatten(0, 1), torch.from_numpy(aligned).flatten().to(args.device), ignore_index=-100)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
            total += float(loss.detach().cpu()); batches += 1
        scheduler.step()
        result = metrics(model, validation, args, config.other_index)
        sources = result["by_source"]
        score = (0.35 * sources["local_phrases"]["known_wer"]
                 + 0.20 * sources["asllrp_contiguous"]["known_wer"]
                 + 0.10 * sources["ncslgr_strict"]["known_wer"]
                 + 0.30 * result["isolated"]["known_wer"]
                 + 0.05 * result["blank_boundaries"]["any_token_false_emission_rate"])
        row = {"epoch": epoch, "loss": total / batches, "score": score,
               "local_wer": sources["local_phrases"]["known_wer"],
               "asllrp_wer": sources["asllrp_contiguous"]["known_wer"],
               "isolated_exact": result["isolated"]["exact_accuracy"],
               "blank_false": result["blank_boundaries"]["any_token_false_emission_rate"]}
        history.append(row); print(json.dumps(row), flush=True)
        if best is None or score < best["score"]:
            best = {"score": score, "epoch": epoch, "validation": result}
            torch.save({
                "format": "slt_continuous_evidence_v17", "version": 1,
                "config": config.to_dict(), "model_state_dict": copy.deepcopy(model.cpu().state_dict()),
                "base": str(args.base), "base_sha256": cache["base_sha256"],
                "label_to_index": cache["labels"], "cache": str(args.cache),
                "cache_sha256": sha256(args.cache), "selected": best,
                "test_accessed": False,
            }, args.output / "best_model.pth")
            model.to(args.device)
        (args.output / "result.json").write_text(json.dumps({
            "selected": best, "history": history, "config": config.to_dict(),
            "train_sources": dict(counts), "elapsed_seconds": time.perf_counter() - started,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "test_accessed": False,
        }, indent=2) + "\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("artifacts/models/stage1_v17_grounded_clean_lineage_v2/best_model.pth"))
    p.add_argument("--phrases", type=Path, default=Path("data/local/stage2_v17_grounded_signer_split"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/generated/continuous_evidence_v17_v1/cache.pth"))
    p.add_argument("--output", type=Path, default=Path("artifacts/models/continuous_evidence_v17_v1"))
    p.add_argument("--windows", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--samples-per-epoch", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--embedding-batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--alignment-weight", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=17091)
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--causal-local-root", type=Path)
    p.add_argument("--local-duration-factors", type=float, nargs="*", default=[])
    return p


if __name__ == "__main__":
    run(parser().parse_args())
