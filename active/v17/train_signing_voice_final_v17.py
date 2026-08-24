#!/usr/bin/env python3
"""Train the frozen signing-voice design on every eligible train-only voice."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")

import argparse
import gc
import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_signing_voice_v17 import SigningVoiceGeneratorV17, SigningVoiceV17Config
from active.v17.train_signing_voice_v17 import (
    VoicePairDataset,
    build_prototypes,
    cross_gloss_style_loss,
    emitted_style_loss,
    load_content_model,
    sha256,
    trajectory_terms,
    voice_centroids,
)


LOG = logging.getLogger("train_signing_voice_final_v17")


def run(args: argparse.Namespace) -> dict[str, object]:
    device_name = (
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with np.load(args.pool, allow_pickle=False) as payload:
        landmarks = payload["landmarks"].astype(np.float16)
        targets = payload["target_indices"].astype(np.int64)
        signers = payload["signer_ids"].astype(str)
        observed_frames = payload["observed_frames"].astype(np.int16)
    voices = sorted({
        signer for signer in set(signers.tolist())
        if len(set(targets[signers == signer].tolist())) >= 2
    })
    indices = np.arange(len(landmarks))[np.isin(signers, voices)]
    prototypes, medoids = build_prototypes(landmarks, targets, indices)
    voice_to_index = {voice: index for index, voice in enumerate(voices)}
    dataset = VoicePairDataset(
        landmarks, targets, signers, indices, prototypes, voice_to_index,
        seed=args.seed, fixed=False,
    )
    signer_counts = {
        signer: int((signers[dataset.indices] == signer).sum()) for signer in voices
    }
    weights = np.asarray([
        0.5 / len(dataset) + 0.5 / (len(voices) * signer_counts[str(signers[index])])
        for index in dataset.indices
    ])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            torch.from_numpy(weights).double(), len(dataset), replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        ),
        num_workers=0,
    )
    content_model, label_to_index = load_content_model(args.content_checkpoint, device)
    manifest_labels = {
        str(key): int(value)
        for key, value in json.loads(args.manifest.read_text())["label_to_index"].items()
    }
    if label_to_index != manifest_labels:
        raise ValueError("Stage-1 and signing-voice class maps differ")
    model = SigningVoiceGeneratorV17(SigningVoiceV17Config(
        dim=args.dim, style_dim=args.style_dim,
        encoder_depth=args.encoder_depth, decoder_depth=args.decoder_depth,
        heads=args.heads, dropout=args.dropout,
    ))
    model.install_style_classifier(len(voices))
    model.to(device)
    if model.style_classifier is None:
        raise RuntimeError("style classifier is unavailable")
    history = []
    started = time.monotonic()
    pretrain_parameters = list(model.style_encoder.parameters()) + list(
        model.style_classifier.parameters()
    )
    pretrain_optimizer = torch.optim.AdamW(
        pretrain_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    for epoch in range(1, args.style_pretrain_epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch in loader:
            target = batch["target_landmarks"].to(device)
            reference = batch["reference_landmarks"].to(device)
            voice_labels = batch["voice"].to(device)
            pretrain_optimizer.zero_grad(set_to_none=True)
            style = model.encode_style(reference)
            target_style = model.encode_style(target)
            voice_loss = 0.5 * (
                F.cross_entropy(model.style_classifier(style), voice_labels)
                + F.cross_entropy(model.style_classifier(target_style), voice_labels)
            )
            loss = (
                args.voice_weight * voice_loss
                + args.cross_gloss_style_weight * cross_gloss_style_loss(
                    style, target_style, voice_labels
                )
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                pretrain_parameters, args.gradient_clip, error_if_nonfinite=True
            )
            pretrain_optimizer.step()
            total += float(loss.detach()) * len(target)
            seen += len(target)
        value = total / seen
        history.append({"phase": "style_pretrain", "epoch": epoch, "train_loss": value})
        LOG.info("style_pretrain=%d/%d loss=%.5f", epoch, args.style_pretrain_epochs, value)
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    model.style_encoder.requires_grad_(False).eval()
    model.style_classifier.requires_grad_(False).eval()
    generator_parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        generator_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.style_encoder.eval()
        model.style_classifier.eval()
        total = 0.0
        seen = 0
        for batch in loader:
            target = batch["target_landmarks"].to(device)
            reference = batch["reference_landmarks"].to(device)
            prototype = batch["prototype"].to(device)
            labels = batch["target"].to(device)
            voice_labels = batch["voice"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                style = model.encode_style(reference)
                target_style = model.encode_style(target)
            generated = model.generate_from_style(prototype, labels, style)
            terms = trajectory_terms(generated, target)
            generated_style = model.encode_style(generated)
            loss = (
                terms["spatial"]
                + args.velocity_weight * terms["velocity"]
                + args.acceleration_weight * terms["acceleration"]
                + args.voice_weight * F.cross_entropy(
                    model.style_classifier(generated_style), voice_labels
                )
                + args.content_weight * F.cross_entropy(content_model(generated), labels)
                + args.emitted_style_weight * emitted_style_loss(
                    generated_style, target_style, voice_labels, labels
                )
                + args.style_consistency_weight * (
                    1.0 - F.cosine_similarity(generated_style, style.detach())
                ).mean()
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite signing-voice loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                generator_parameters, args.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach()) * len(target)
            seen += len(target)
        value = total / seen
        history.append({"phase": "generator", "epoch": epoch, "train_loss": value})
        LOG.info("epoch=%d/%d loss=%.5f", epoch, args.epochs, value)
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    centroids = voice_centroids(model, landmarks, signers, dataset.indices, voices, device)
    class_medians = np.asarray([
        np.median(observed_frames[indices[targets[indices] == target]])
        for target in range(100)
    ])
    duration_ratios = []
    for voice in voices:
        rows = indices[signers[indices] == voice]
        duration_ratios.append(float(np.median(
            observed_frames[rows] / class_medians[targets[rows]].clip(min=1)
        )))
    evidence = []
    for path in args.evidence:
        row = json.loads(path.read_text())
        evidence.append({
            "report": path.as_posix(), "sha256": sha256(path),
            "fold": row["fold"], "selected_epoch": row["selected_epoch"],
            "validation": row["validation"],
        })
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "slt_signing_voice_generator_v17",
        "version": 2,
        "model_config": model.config.to_dict(),
        "model_state_dict": state,
        "style_classifier_voices": len(voices),
        "content_prototypes": prototypes,
        "content_prototype_pool_indices": medoids,
        "label_to_index": label_to_index,
        "train_voices": voices,
        "validation_voices": [],
        "train_voice_style_centroids": centroids,
        "train_voice_duration_ratios": torch.tensor(duration_ratios),
        "class_median_observed_frames": torch.tensor(class_medians),
        "fold": None,
        "seed": args.seed,
        "epoch": args.epochs,
        "style_pretrain_epochs": args.style_pretrain_epochs,
        "selection_performed": False,
        "signer_disjoint_evidence": evidence,
        "pool": args.pool.as_posix(),
        "pool_sha256": sha256(args.pool),
        "content_checkpoint": args.content_checkpoint.as_posix(),
        "content_checkpoint_sha256": sha256(args.content_checkpoint),
        "pairing_contract": "style reference always comes from the same signer but a different gloss",
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
    }
    checkpoint_path = args.output / "model.pth"
    torch.save(checkpoint, checkpoint_path)
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    report = {
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "epochs": args.epochs,
        "style_pretrain_epochs": args.style_pretrain_epochs,
        "train_voices": len(voices),
        "train_examples": len(dataset),
        "seconds": time.monotonic() - started,
        "selection_performed": False,
        "signer_disjoint_evidence": evidence,
        "claim_boundary": "train-all artifact inherits fold evidence; it has no independent naturalness score",
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=Path("data/local/signing_voice_v17/train_only_landmark_pool.npz"))
    parser.add_argument("--content-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"))
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/stage2_training_manifest_v17.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/signing_voice_v17_allvoices_final"))
    parser.add_argument("--evidence", type=Path, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--style-pretrain-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=18701)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--style-dim", type=int, default=32)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--decoder-depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--velocity-weight", type=float, default=0.25)
    parser.add_argument("--acceleration-weight", type=float, default=0.15)
    parser.add_argument("--voice-weight", type=float, default=0.10)
    parser.add_argument("--content-weight", type=float, default=0.10)
    parser.add_argument("--cross-gloss-style-weight", type=float, default=0.25)
    parser.add_argument("--emitted-style-weight", type=float, default=0.10)
    parser.add_argument("--style-consistency-weight", type=float, default=0.10)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
