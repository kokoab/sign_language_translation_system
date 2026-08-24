#!/usr/bin/env python3
"""Train a signer-disjoint content/style landmark generator for a real signing voice."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")

import argparse
import copy
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_signing_voice_v17 import (
    SigningVoiceGeneratorV17,
    SigningVoiceV17Config,
)
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.train_transition_inpainter_v17 import sha256


LOG = logging.getLogger("train_signing_voice_v17")
FOLDS = {
    0: {
        "citizen:P11", "citizen:P27", "citizen:P33",
        "semlex:17", "semlex:49", "semlex:62", "asllrp:RACHEL",
    },
    1: {
        "citizen:P29", "citizen:P31", "citizen:P37",
        "semlex:30", "semlex:39", "semlex:84", "asllrp:CORY",
    },
    2: {
        "citizen:P40", "citizen:P50", "citizen:P52",
        "semlex:11", "semlex:35", "semlex:88", "asllrp:BENJAMIN_JAMES_BAHAN",
    },
}


class VoicePairDataset(Dataset):
    """Pair every target with a different gloss from the same signer."""

    def __init__(
        self,
        landmarks: np.ndarray,
        targets: np.ndarray,
        signers: np.ndarray,
        indices: np.ndarray,
        prototypes: torch.Tensor,
        voice_to_index: dict[str, int] | None,
        *,
        seed: int,
        fixed: bool,
    ):
        self.landmarks = landmarks
        self.targets = targets
        self.signers = signers
        self.indices = indices.astype(np.int64)
        self.prototypes = prototypes
        self.voice_to_index = voice_to_index
        self.seed = seed
        self.fixed = fixed
        self.by_signer = {}
        for signer in sorted(set(signers[self.indices].tolist())):
            rows = self.indices[signers[self.indices] == signer]
            if len(set(targets[rows].tolist())) >= 2:
                self.by_signer[str(signer)] = rows
        self.indices = np.asarray([
            index for index in self.indices if str(signers[index]) in self.by_signer
        ], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, object]:
        target_index = int(self.indices[item])
        signer = str(self.signers[target_index])
        candidates = self.by_signer[signer]
        different = candidates[self.targets[candidates] != self.targets[target_index]]
        if not len(different):
            raise RuntimeError("style reference has no different-gloss candidate")
        if self.fixed:
            reference_index = int(different[(self.seed + item * 104729) % len(different)])
        else:
            reference_index = int(different[np.random.randint(len(different))])
        target = int(self.targets[target_index])
        return {
            "target_landmarks": torch.from_numpy(
                self.landmarks[target_index].astype(np.float32)
            ),
            "reference_landmarks": torch.from_numpy(
                self.landmarks[reference_index].astype(np.float32)
            ),
            "prototype": self.prototypes[target],
            "target": target,
            "voice": -1 if self.voice_to_index is None else self.voice_to_index[signer],
            "signer": signer,
            "target_index": target_index,
            "reference_index": reference_index,
        }


def build_prototypes(
    landmarks: np.ndarray, targets: np.ndarray, indices: np.ndarray
) -> tuple[torch.Tensor, list[int]]:
    prototypes = []
    medoids = []
    for target in range(100):
        rows = indices[targets[indices] == target]
        if not len(rows):
            raise ValueError(f"training split has no content for class {target}")
        values = landmarks[rows].astype(np.float32)
        present = values[..., 3:4] > 0
        count = present.sum(axis=0).clip(min=1)
        mean_xyz = (values[..., :3] * present).sum(axis=0) / count
        error = []
        for value in values:
            valid = (value[..., 3] > 0) & (count[..., 0] > 0)
            error.append(float(np.square(value[..., :3] - mean_xyz)[valid].mean()))
        choice = int(rows[int(np.argmin(error))])
        prototypes.append(landmarks[choice].astype(np.float32))
        medoids.append(choice)
    return torch.from_numpy(np.stack(prototypes)), medoids


def load_content_model(path: Path, device: torch.device) -> tuple[SLTStage1V17, dict[str, int]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_unified_multimodal_v17":
        raise ValueError("unexpected Stage-1 content checkpoint")
    config = dict(checkpoint["landmark_model_config"])
    if "phonology_head_sizes" in config:
        config["phonology_head_sizes"] = tuple(
            tuple(value) for value in config["phonology_head_sizes"]
        )
    model = SLTStage1V17(Stage1V17Config(**config))
    model.load_state_dict(checkpoint["landmark_model_state_dict"])
    model.eval().requires_grad_(False).to(device)
    return model, {str(key): int(value) for key, value in checkpoint["label_to_index"].items()}


def trajectory_terms(
    generated: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    present = (generated[..., 3] > 0) & (target[..., 3] > 0)
    spatial = F.smooth_l1_loss(generated[..., :3][present], target[..., :3][present])
    pair = present[:, 1:] & present[:, :-1]
    generated_velocity = generated[:, 1:, :, :3] - generated[:, :-1, :, :3]
    target_velocity = target[:, 1:, :, :3] - target[:, :-1, :, :3]
    velocity = F.smooth_l1_loss(generated_velocity[pair], target_velocity[pair])
    triple = pair[:, 1:] & pair[:, :-1]
    generated_accel = generated_velocity[:, 1:] - generated_velocity[:, :-1]
    target_accel = target_velocity[:, 1:] - target_velocity[:, :-1]
    acceleration = F.smooth_l1_loss(generated_accel[triple], target_accel[triple])
    return {"spatial": spatial, "velocity": velocity, "acceleration": acceleration}


def cross_gloss_style_loss(
    reference_style: torch.Tensor,
    target_style: torch.Tensor,
    voices: torch.Tensor,
    margin: float = 0.20,
) -> torch.Tensor:
    """Pull same-signer/different-gloss pairs together and separate other voices."""
    positive = 1.0 - F.cosine_similarity(reference_style, target_style)
    similarity = reference_style @ target_style.transpose(0, 1)
    different = voices[:, None] != voices[None, :]
    if different.any():
        negative = F.relu(similarity[different] - margin).mean()
    else:
        negative = similarity.sum() * 0.0
    return positive.mean() + 0.25 * negative


def emitted_style_loss(
    generated_style: torch.Tensor,
    target_style: torch.Tensor,
    voices: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.20,
) -> torch.Tensor:
    """Make generated motion match its signer under exact-content controls."""
    positive = 1.0 - F.cosine_similarity(generated_style, target_style)
    similarity = generated_style @ target_style.transpose(0, 1)
    negative_mask = (
        (voices[:, None] != voices[None, :])
        & (targets[:, None] == targets[None, :])
    )
    if negative_mask.any():
        negative = F.relu(similarity[negative_mask] - margin).mean()
    else:
        negative = similarity.sum() * 0.0
    return positive.mean() + 0.25 * negative


def rank_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.concatenate((positive, negative))
    labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    positive_rank = ranks[labels == 1].sum()
    return float(
        (positive_rank - len(positive) * (len(positive) + 1) / 2)
        / (len(positive) * len(negative))
    )


def signer_aware_style_scores(
    reference_style: np.ndarray,
    target_style: np.ndarray,
    signers: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return same-signer positives and same-gloss/different-signer negatives."""
    if (
        reference_style.shape != target_style.shape
        or len(reference_style) != len(signers)
        or len(reference_style) != len(targets)
    ):
        raise ValueError("style embeddings, signer identities, and targets must align")
    if reference_style.ndim != 2:
        raise ValueError("style embeddings must be two-dimensional")
    similarity = reference_style @ target_style.T
    different = (
        (signers[:, None] != signers[None, :])
        & (targets[:, None] == targets[None, :])
    )
    eligible_positive = different.any(axis=1)
    positive = np.diag(similarity)[eligible_positive]
    negative = similarity[different]
    if not len(positive) or not len(negative):
        raise ValueError("style verification requires same- and different-signer pairs")
    return positive, negative


@torch.inference_mode()
def evaluate(
    model: SigningVoiceGeneratorV17,
    loader: DataLoader,
    content_model: SLTStage1V17,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    totals = {
        "generated_spatial": 0.0, "prototype_spatial": 0.0,
        "generated_velocity": 0.0, "prototype_velocity": 0.0,
        "generated_acceleration": 0.0, "prototype_acceleration": 0.0,
        "style_effect": 0.0, "generated_condition_style_cosine": 0.0,
    }
    seen = 0
    generated_correct = 0
    prototype_correct = 0
    target_correct = 0
    reference_styles = []
    conditioning_styles = []
    target_styles = []
    style_signers = []
    style_targets = []
    for batch in loader:
        target = batch["target_landmarks"].to(device)
        reference = batch["reference_landmarks"].to(device)
        prototype = batch["prototype"].to(device)
        labels = batch["target"].to(device)
        generated, style = model(prototype, labels, reference)
        target_style = model.encode_style(target)
        generated_style = model.encode_style(generated)
        reference_styles.append(generated_style.cpu().numpy())
        conditioning_styles.append(style.cpu().numpy())
        target_styles.append(target_style.cpu().numpy())
        style_signers.extend(str(value) for value in batch["signer"])
        style_targets.extend(int(value) for value in labels.cpu().tolist())
        generated_terms = trajectory_terms(generated, target)
        prototype_terms = trajectory_terms(prototype, target)
        count = len(target)
        seen += count
        for key in ("spatial", "velocity", "acceleration"):
            totals[f"generated_{key}"] += float(generated_terms[key]) * count
            totals[f"prototype_{key}"] += float(prototype_terms[key]) * count
        totals["style_effect"] += float(
            (generated[..., :3] - prototype[..., :3]).abs().mean()
        ) * count
        totals["generated_condition_style_cosine"] += float(
            F.cosine_similarity(generated_style, style).mean()
        ) * count
        generated_correct += int((content_model(generated).argmax(1) == labels).sum())
        prototype_correct += int((content_model(prototype).argmax(1) == labels).sum())
        target_correct += int((content_model(target).argmax(1) == labels).sum())
    same_scores, different_scores = signer_aware_style_scores(
        np.concatenate(reference_styles),
        np.concatenate(target_styles),
        np.asarray(style_signers),
        np.asarray(style_targets),
    )
    condition_same, condition_different = signer_aware_style_scores(
        np.concatenate(conditioning_styles),
        np.concatenate(target_styles),
        np.asarray(style_signers),
        np.asarray(style_targets),
    )
    result = {"examples": seen}
    result.update({key: value / seen for key, value in totals.items()})
    for key in ("spatial", "velocity", "acceleration"):
        base = result[f"prototype_{key}"]
        result[f"relative_{key}_improvement"] = (
            base - result[f"generated_{key}"]
        ) / max(base, 1e-8)
    result.update({
        "generated_content_accuracy": generated_correct / seen,
        "prototype_content_accuracy": prototype_correct / seen,
        "target_content_accuracy": target_correct / seen,
        "same_voice_cosine": float(np.mean(same_scores)),
        "different_voice_cosine": float(np.mean(different_scores)),
        "style_verification_auc": rank_auc(same_scores, different_scores),
        "conditioning_style_verification_auc": rank_auc(
            condition_same, condition_different
        ),
        "style_positive_pairs": int(len(same_scores)),
        "style_negative_pairs": int(len(different_scores)),
        "style_negative_contract": "same gloss, different signer",
    })
    return result


def selection_score(metrics: dict[str, float | int]) -> float:
    content_regression = max(
        0.0,
        float(metrics["prototype_content_accuracy"])
        - float(metrics["generated_content_accuracy"]),
    )
    return (
        float(metrics["relative_spatial_improvement"])
        + 0.25 * float(metrics["relative_velocity_improvement"])
        + 0.10 * (float(metrics["style_verification_auc"]) - 0.5)
        - 2.0 * content_regression
    )


@torch.inference_mode()
def voice_centroids(
    model: SigningVoiceGeneratorV17,
    landmarks: np.ndarray,
    signers: np.ndarray,
    indices: np.ndarray,
    voices: list[str],
    device: torch.device,
) -> torch.Tensor:
    output = []
    model.eval()
    for signer in voices:
        rows = indices[signers[indices] == signer]
        embeddings = []
        for start in range(0, len(rows), 128):
            value = torch.from_numpy(
                landmarks[rows[start:start + 128]].astype(np.float32)
            ).to(device)
            embeddings.append(model.encode_style(value).cpu())
        output.append(F.normalize(torch.cat(embeddings).mean(dim=0), dim=0))
    return torch.stack(output)


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
        pool_metadata = json.loads(str(payload["metadata_json"]))
    eligible = {
        signer for signer in set(signers.tolist())
        if len(set(targets[signers == signer].tolist())) >= 2
    }
    if args.fold not in FOLDS:
        raise ValueError("fold must be 0, 1, or 2")
    validation_voices = FOLDS[args.fold]
    if not validation_voices <= eligible:
        raise ValueError("held-out voice split is unavailable")
    train_voices = sorted(eligible - validation_voices)
    validation_voices = sorted(validation_voices)
    all_indices = np.arange(len(landmarks))
    train_indices = all_indices[np.isin(signers, train_voices)]
    validation_indices = all_indices[np.isin(signers, validation_voices)]
    prototypes, medoids = build_prototypes(landmarks, targets, train_indices)
    voice_to_index = {voice: index for index, voice in enumerate(train_voices)}
    train_dataset = VoicePairDataset(
        landmarks, targets, signers, train_indices, prototypes, voice_to_index,
        seed=args.seed, fixed=False,
    )
    validation_dataset = VoicePairDataset(
        landmarks, targets, signers, validation_indices, prototypes, None,
        seed=args.seed + 1000, fixed=True,
    )
    signer_counts = {
        signer: int((signers[train_dataset.indices] == signer).sum())
        for signer in train_voices
    }
    weights = np.asarray([
        0.5 / len(train_dataset)
        + 0.5 / (len(train_voices) * signer_counts[str(signers[index])])
        for index in train_dataset.indices
    ])
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights).double(),
        num_samples=len(train_dataset), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )

    content_model, label_to_index = load_content_model(args.content_checkpoint, device)
    manifest_labels = json.loads(args.manifest.read_text())["label_to_index"]
    if label_to_index != {str(key): int(value) for key, value in manifest_labels.items()}:
        raise ValueError("Stage-1 and signing-voice class maps differ")
    model = SigningVoiceGeneratorV17(SigningVoiceV17Config(
        dim=args.dim, style_dim=args.style_dim,
        encoder_depth=args.encoder_depth, decoder_depth=args.decoder_depth,
        heads=args.heads, dropout=args.dropout,
    ))
    model.install_style_classifier(len(train_voices))
    model.to(device)
    if model.style_classifier is None:
        raise RuntimeError("style classifier is unavailable")
    started = time.monotonic()
    pretrain_parameters = list(model.style_encoder.parameters()) + list(
        model.style_classifier.parameters()
    )
    pretrain_optimizer = torch.optim.AdamW(
        pretrain_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    history = []
    for epoch in range(1, args.style_pretrain_epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch in train_loader:
            target = batch["target_landmarks"].to(device)
            reference = batch["reference_landmarks"].to(device)
            voices = batch["voice"].to(device)
            pretrain_optimizer.zero_grad(set_to_none=True)
            style = model.encode_style(reference)
            target_style = model.encode_style(target)
            voice_loss = 0.5 * (
                F.cross_entropy(model.style_classifier(style), voices)
                + F.cross_entropy(model.style_classifier(target_style), voices)
            )
            cross_gloss = cross_gloss_style_loss(style, target_style, voices)
            loss = (
                args.voice_weight * voice_loss
                + args.cross_gloss_style_weight * cross_gloss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                pretrain_parameters, args.gradient_clip, error_if_nonfinite=True
            )
            pretrain_optimizer.step()
            total += float(loss.detach()) * len(target)
            seen += len(target)
        history.append({
            "phase": "style_pretrain", "epoch": epoch,
            "train_loss": total / seen,
        })
        LOG.info(
            "style_pretrain=%d/%d loss=%.5f",
            epoch, args.style_pretrain_epochs, total / seen,
        )
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    model.style_encoder.requires_grad_(False).eval()
    model.style_classifier.requires_grad_(False).eval()
    generator_parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        generator_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    initial = evaluate(model, validation_loader, content_model, device)
    best_metrics = initial
    best_score = selection_score(initial)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history.append({
        "phase": "generator", "epoch": 0,
        "validation": initial, "selection_score": best_score,
    })
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.style_encoder.eval()
        model.style_classifier.eval()
        total = 0.0
        seen = 0
        for batch in train_loader:
            target = batch["target_landmarks"].to(device)
            reference = batch["reference_landmarks"].to(device)
            prototype = batch["prototype"].to(device)
            labels = batch["target"].to(device)
            voices = batch["voice"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                style = model.encode_style(reference)
                target_style = model.encode_style(target)
            generated = model.generate_from_style(prototype, labels, style)
            terms = trajectory_terms(generated, target)
            content_loss = F.cross_entropy(content_model(generated), labels)
            generated_style = model.encode_style(generated)
            voice_loss = F.cross_entropy(model.style_classifier(generated_style), voices)
            emitted_style = emitted_style_loss(
                generated_style, target_style, voices, labels
            )
            consistency = (1.0 - F.cosine_similarity(
                generated_style, style.detach()
            )).mean()
            loss = (
                terms["spatial"]
                + args.velocity_weight * terms["velocity"]
                + args.acceleration_weight * terms["acceleration"]
                + args.voice_weight * voice_loss
                + args.content_weight * content_loss
                + args.emitted_style_weight * emitted_style
                + args.style_consistency_weight * consistency
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
        metrics = evaluate(model, validation_loader, content_model, device)
        score = selection_score(metrics)
        history.append({
            "phase": "generator", "epoch": epoch, "train_loss": total / seen,
            "validation": metrics, "selection_score": score,
        })
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        LOG.info(
            "epoch=%d loss=%.5f spatial=%+.2f%% content=%.2f%% emitted_auc=%.3f condition_auc=%.3f score=%.4f best=%d",
            epoch, total / seen,
            100 * float(metrics["relative_spatial_improvement"]),
            100 * float(metrics["generated_content_accuracy"]),
            float(metrics["style_verification_auc"]),
            float(metrics["conditioning_style_verification_auc"]),
            score, best_epoch,
        )
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        if patience >= args.patience:
            break

    model.load_state_dict(best_state)
    train_centroids = voice_centroids(
        model, landmarks, signers, train_dataset.indices, train_voices, device
    )
    duration_ratios = []
    class_medians = np.asarray([
        np.median(observed_frames[train_indices[targets[train_indices] == target]])
        for target in range(100)
    ])
    for voice in train_voices:
        rows = train_indices[signers[train_indices] == voice]
        duration_ratios.append(float(np.median(
            observed_frames[rows] / class_medians[targets[rows]].clip(min=1)
        )))

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "slt_signing_voice_generator_v17",
        "version": 2,
        "model_config": model.config.to_dict(),
        "model_state_dict": best_state,
        "style_classifier_voices": len(train_voices),
        "content_prototypes": prototypes,
        "content_prototype_pool_indices": medoids,
        "label_to_index": label_to_index,
        "train_voices": train_voices,
        "validation_voices": validation_voices,
        "train_voice_style_centroids": train_centroids,
        "train_voice_duration_ratios": torch.tensor(duration_ratios),
        "class_median_observed_frames": torch.tensor(class_medians),
        "fold": args.fold,
        "seed": args.seed,
        "epoch": best_epoch,
        "style_pretrain_epochs": args.style_pretrain_epochs,
        "validation_metrics": best_metrics,
        "pool": args.pool.as_posix(),
        "pool_sha256": sha256(args.pool),
        "content_checkpoint": args.content_checkpoint.as_posix(),
        "content_checkpoint_sha256": sha256(args.content_checkpoint),
        "pairing_contract": (
            "style reference always comes from the same signer but a different gloss"
        ),
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
    }
    checkpoint_path = args.output / "best_model.pth"
    torch.save(checkpoint, checkpoint_path)
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    report = {
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "fold": args.fold,
        "selected_epoch": best_epoch,
        "selection_score": best_score,
        "train_voices": len(train_voices),
        "validation_voices": validation_voices,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "initial_validation": initial,
        "validation": best_metrics,
        "seconds": time.monotonic() - started,
        "claim_boundary": (
            "held-out landmark reconstruction/content/style evidence is not "
            "fluent-signer-rated linguistic naturalness"
        ),
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
    parser.add_argument(
        "--pool", type=Path,
        default=Path("data/local/signing_voice_v17/train_only_landmark_pool.npz"),
    )
    parser.add_argument(
        "--content-checkpoint", type=Path,
        default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("active/v17/stage2_training_manifest_v17.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/models/signing_voice_v17_fold0"),
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=18701)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--style-pretrain-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
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
