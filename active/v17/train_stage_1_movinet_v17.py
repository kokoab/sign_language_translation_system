#!/usr/bin/env python3
"""Train a hand-aware MoViNet-A0 jointly with frozen Apple landmark evidence.

This is deliberately not a generic full-frame action-classification head. A shared
pretrained MoViNet processes anatomical left-hand, right-hand, and union/context video
streams. Explicit masks and box trajectories preserve hand identity and interaction
context. A zero-initialized residual fusion path starts at the exact frozen Apple
baseline while a visual-only auxiliary objective forces the video branch to learn signs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import tensorflow as tf
import tf_keras

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.movinet_data_v17 import (
        FRAMES,
        VIEWS,
        augment_sign_views,
        decode_crop_archive,
        load_aligned_records,
    )
else:
    from .movinet_data_v17 import (
        FRAMES,
        VIEWS,
        augment_sign_views,
        decode_crop_archive,
        load_aligned_records,
    )

from official.projects.movinet.modeling import movinet, movinet_model


LOG = logging.getLogger("stage1_movinet_v17")
NUM_CLASSES = 100
LANDMARK_DIM = 256


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def balanced_subset(records, count: int):
    remaining = {target: count for target in range(NUM_CLASSES)}
    selected = []
    for record in records:
        if remaining[record.target] > 0:
            selected.append(record)
            remaining[record.target] -= 1
    if any(remaining.values()):
        raise ValueError("balanced subset exceeds available examples")
    return selected


def make_dataset(records, resolution, batch_size, training, seed):
    def generator():
        rng = np.random.default_rng(seed)
        order = np.arange(len(records))
        if training:
            rng.shuffle(order)
        for index in order:
            record = records[int(index)]
            pixels, valid, boxes = decode_crop_archive(record.crop_path, resolution)
            if training:
                pixels, valid, boxes = augment_sign_views(pixels, valid, boxes, rng)
            yield (
                pixels.astype(np.float32) / 255.0,
                valid,
                boxes,
                record.landmark_feature,
                record.base_logits,
                np.int64(record.target),
            )

    signature = (
        tf.TensorSpec((FRAMES, VIEWS, resolution, resolution, 3), tf.float32),
        tf.TensorSpec((FRAMES, VIEWS), tf.bool),
        tf.TensorSpec((FRAMES, VIEWS, 4), tf.float32),
        tf.TensorSpec((LANDMARK_DIM,), tf.float32),
        tf.TensorSpec((NUM_CLASSES,), tf.float32),
        tf.TensorSpec((), tf.int64),
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=signature)
    return dataset.batch(batch_size, drop_remainder=False).prefetch(1)


class SignMoViNetFusion(tf_keras.Model):
    """Shared-view MoViNet plus identity-preserving landmark/RGB residual fusion."""

    def __init__(self, backbone, dim=256, dropout=0.25):
        super().__init__(name="sign_movinet_fusion_v17")
        self.backbone = backbone
        self.view_embedding = self.add_weight(
            name="view_embedding",
            shape=(VIEWS, 32),
            initializer=tf_keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.box_projection = tf_keras.Sequential(
            [
                tf_keras.layers.Dense(96, activation=tf.nn.gelu),
                tf_keras.layers.LayerNormalization(),
            ],
            name="box_trajectory_projection",
        )
        self.view_projection = tf_keras.Sequential(
            [
                tf_keras.layers.Dense(dim, activation=tf.nn.gelu),
                tf_keras.layers.LayerNormalization(),
                tf_keras.layers.Dropout(dropout),
            ],
            name="view_projection",
        )
        self.view_attention = tf_keras.Sequential(
            [
                tf_keras.layers.Dense(64, activation=tf.nn.gelu),
                tf_keras.layers.Dense(1),
            ],
            name="view_attention",
        )
        self.visual_classifier = tf_keras.layers.Dense(NUM_CLASSES, name="visual_classifier")
        self.landmark_projection = tf_keras.Sequential(
            [
                tf_keras.layers.LayerNormalization(),
                tf_keras.layers.Dense(dim, activation=tf.nn.gelu),
            ],
            name="landmark_projection",
        )
        self.fusion_hidden = tf_keras.Sequential(
            [
                tf_keras.layers.Dense(dim * 2, activation=tf.nn.gelu),
                tf_keras.layers.Dropout(dropout),
                tf_keras.layers.Dense(dim, activation=tf.nn.gelu),
            ],
            name="fusion_hidden",
        )
        self.fusion_gate = tf_keras.layers.Dense(
            1,
            activation="sigmoid",
            bias_initializer=tf_keras.initializers.Constant(-2.0),
            name="fusion_gate",
        )
        self.residual_classifier = tf_keras.layers.Dense(
            NUM_CLASSES,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="zero_initialized_residual",
        )

    def call(self, inputs, training=False):
        pixels, valid, boxes, landmark_features, base_logits = inputs
        batch = tf.shape(pixels)[0]

        # [B,T,V,H,W,C] -> [B*V,T,H,W,C], preserving temporal order within each view.
        videos = tf.transpose(pixels, (0, 2, 1, 3, 4, 5))
        videos = tf.reshape(
            videos,
            (-1, tf.shape(pixels)[1], tf.shape(pixels)[3], tf.shape(pixels)[4], 3),
        )
        endpoints, _ = self.backbone({"image": videos}, training=training)
        visual = tf.reshape(endpoints["head"], (batch, VIEWS, -1))

        valid_by_view = tf.transpose(valid, (0, 2, 1))
        stream_valid = tf.reduce_any(valid_by_view, axis=-1)
        box_sequence = tf.transpose(boxes, (0, 2, 1, 3))
        box_sequence *= tf.cast(valid_by_view[..., None], boxes.dtype)
        box_sequence = tf.reshape(box_sequence, (batch, VIEWS, FRAMES * 4))
        mask_sequence = tf.cast(valid_by_view, boxes.dtype)
        geometry = self.box_projection(
            tf.concat((box_sequence, mask_sequence), axis=-1), training=training
        )
        view_ids = tf.broadcast_to(self.view_embedding[None, :, :], (batch, VIEWS, 32))
        tokens = self.view_projection(
            tf.concat((visual, geometry, view_ids), axis=-1), training=training
        )
        attention = tf.squeeze(self.view_attention(tokens, training=training), axis=-1)
        attention = tf.where(stream_valid, attention, tf.cast(-1e4, attention.dtype))
        weights = tf.nn.softmax(attention, axis=1)
        visual_feature = tf.reduce_sum(tokens * weights[..., None], axis=1)
        visual_logits = self.visual_classifier(visual_feature)

        landmark = self.landmark_projection(landmark_features, training=training)
        cross_modal = tf.concat(
            (landmark, visual_feature, landmark * visual_feature, tf.abs(landmark - visual_feature)),
            axis=-1,
        )
        hidden = self.fusion_hidden(cross_modal, training=training)
        gate = self.fusion_gate(cross_modal)
        residual = self.residual_classifier(hidden)
        fused_logits = base_logits + gate * residual
        return {"fused_logits": fused_logits, "visual_logits": visual_logits, "gate": gate}


def build_pretrained_backbone(checkpoint_dir: Path):
    backbone = movinet.Movinet(model_id="a0", causal=False)
    classifier = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
    classifier.build([None, None, None, None, 3])
    checkpoint_path = tf.train.latest_checkpoint(str(checkpoint_dir))
    if checkpoint_path is None:
        raise FileNotFoundError(f"no MoViNet checkpoint under {checkpoint_dir}")
    status = tf.train.Checkpoint(model=classifier).restore(checkpoint_path)
    status.assert_existing_objects_matched()
    status.expect_partial()  # The archive also contains an unused save counter.
    return backbone, checkpoint_path


def classification_loss(targets, logits, label_smoothing):
    one_hot = tf.one_hot(targets, NUM_CLASSES)
    return tf.reduce_mean(
        tf_keras.losses.categorical_crossentropy(
            one_hot, logits, from_logits=True, label_smoothing=label_smoothing
        )
    )


def confusion_metrics(targets, logits):
    targets = np.concatenate(targets).astype(np.int64)
    logits = np.concatenate(logits)
    predictions = logits.argmax(axis=1)
    top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "top1": 100 * float((predictions == targets).mean()),
        "top5": 100 * float((top5 == targets[:, None]).any(axis=1).mean()),
        "macro_f1": 100 * float(f1.mean()),
        "samples": int(len(targets)),
    }


def evaluate(model, dataset, max_batches=0):
    targets = []
    fused_logits = []
    visual_logits = []
    gates = []
    for batch_index, (pixels, valid, boxes, landmark, base, target) in enumerate(dataset):
        if max_batches and batch_index >= max_batches:
            break
        output = model((pixels, valid, boxes, landmark, base), training=False)
        targets.append(target.numpy())
        fused_logits.append(output["fused_logits"].numpy())
        visual_logits.append(output["visual_logits"].numpy())
        gates.append(output["gate"].numpy())
    return {
        "fused": confusion_metrics(targets, fused_logits),
        "visual": confusion_metrics(targets, visual_logits),
        "mean_gate": float(np.concatenate(gates).mean()),
    }


def run(args):
    if args.split_test:
        raise ValueError("the official Citizen test split is frozen and unavailable")
    if args.device == "cpu":
        # TensorFlow Metal 1.2 cannot run MoViNet's XLA-compiled grouped Conv3D ops.
        # Official Model Garden CPU execution supports them and is the reproducible
        # training path on this host. This does not change later TFLite deployment.
        tf.config.set_visible_devices([], "GPU")
    elif args.device == "cuda":
        if not tf.test.is_built_with_cuda():
            raise RuntimeError("--device cuda requires a CUDA-enabled TensorFlow build")
        devices = tf.config.list_physical_devices("GPU")
        if not devices:
            raise RuntimeError("--device cuda requested but TensorFlow found no GPU")
        LOG.info("using CUDA device: %s", devices[0].name)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    train_records = load_aligned_records(
        args.crop_root, args.landmark_train, "train"
    )
    val_records = load_aligned_records(args.crop_root, args.landmark_val, "val")
    if args.smoke:
        train_records = balanced_subset(train_records, 1)
        val_records = balanced_subset(val_records, 1)
        if not args.max_val_batches:
            args.max_val_batches = 2

    backbone, pretrained_checkpoint = build_pretrained_backbone(args.pretrained)
    model = SignMoViNetFusion(backbone, dim=args.dim, dropout=args.dropout)
    sample = next(
        iter(make_dataset(train_records[:1], args.resolution, 1, False, args.seed))
    )
    pixels, valid, boxes, landmark, base, _ = sample
    initial = model((pixels, valid, boxes, landmark, base), training=False)
    if not np.array_equal(initial["fused_logits"].numpy(), base.numpy()):
        raise RuntimeError("fusion must begin as the exact Apple baseline")

    args.output.mkdir(parents=True, exist_ok=True)
    phases = [
        ("warmup", args.warmup_epochs, False, args.head_lr),
        ("joint_finetune", args.finetune_epochs, True, args.backbone_lr),
    ]
    if args.smoke:
        phases = [("smoke", 1, False, args.head_lr)]

    best = -1.0
    stale = 0
    history = []
    epoch_number = 0
    best_weights = args.output / "best.weights.h5"
    for phase_name, phase_epochs, train_backbone, learning_rate in phases:
        model.backbone.trainable = train_backbone
        optimizer = tf_keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=args.weight_decay, clipnorm=1.0
        )

        @tf.function(reduce_retracing=True)
        def train_step(batch):
            pixels, valid, boxes, landmark, base, targets = batch
            with tf.GradientTape() as tape:
                output = model((pixels, valid, boxes, landmark, base), training=True)
                fused_loss = classification_loss(
                    targets, output["fused_logits"], args.label_smoothing
                )
                visual_loss = classification_loss(
                    targets, output["visual_logits"], args.label_smoothing
                )
                loss = fused_loss + args.visual_loss_weight * visual_loss
                if model.losses:
                    loss += tf.add_n(model.losses)
            variables = model.trainable_variables
            gradients = tape.gradient(loss, variables)
            pairs = [(gradient, variable) for gradient, variable in zip(gradients, variables) if gradient is not None]
            optimizer.apply_gradients(pairs)
            return loss, fused_loss, visual_loss

        for _ in range(phase_epochs):
            epoch_number += 1
            started = time.monotonic()
            losses = []
            train_dataset = make_dataset(
                train_records, args.resolution, args.batch_size, True, args.seed + epoch_number
            )
            for batch_index, batch in enumerate(train_dataset):
                if args.max_train_batches and batch_index >= args.max_train_batches:
                    break
                losses.append([float(value) for value in train_step(batch)])
            val_dataset = make_dataset(
                val_records, args.resolution, args.batch_size, False, args.seed
            )
            metrics = evaluate(model, val_dataset, args.max_val_batches)
            row = {
                "epoch": epoch_number,
                "phase": phase_name,
                "train_loss": float(np.mean(losses, axis=0)[0]),
                "train_fused_loss": float(np.mean(losses, axis=0)[1]),
                "train_visual_loss": float(np.mean(losses, axis=0)[2]),
                **metrics,
                "seconds": time.monotonic() - started,
            }
            history.append(row)
            LOG.info(
                "epoch=%d phase=%s loss=%.4f fused=%.2f visual=%.2f gate=%.3f seconds=%.1f",
                epoch_number,
                phase_name,
                row["train_loss"],
                metrics["fused"]["top1"],
                metrics["visual"]["top1"],
                metrics["mean_gate"],
                row["seconds"],
            )
            (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")

            score = metrics["fused"]["top1"]
            if score > best:
                best = score
                stale = 0
                model.save_weights(best_weights)
                metadata = {
                    "format": "slt_stage1_sign_movinet_fusion_v17",
                    "epoch": epoch_number,
                    "phase": phase_name,
                    "validation_metrics": metrics,
                    "pretrained_checkpoint": pretrained_checkpoint,
                    "pretrained_archive_sha256": sha256_file(args.pretrained_archive),
                    "resolution": args.resolution,
                    "frames": FRAMES,
                    "views": ["left", "right", "union"],
                    "test_evaluated": False,
                }
                (args.output / "best_metadata.json").write_text(
                    json.dumps(metadata, indent=2) + "\n"
                )
            else:
                stale += 1
            if not args.smoke and phase_name == "joint_finetune" and stale >= args.patience:
                LOG.info("early stopping after %d stale joint epochs", stale)
                break
        if not args.smoke and phase_name == "joint_finetune" and stale >= args.patience:
            break

    model.load_weights(best_weights)
    final = evaluate(
        model,
        make_dataset(val_records, args.resolution, args.batch_size, False, args.seed),
        args.max_val_batches,
    )
    result = {
        "best_validation_top1": best,
        "best_checkpoint_metrics": final,
        "epochs_completed": len(history),
        "parameters": int(model.count_params()),
        "backbone_parameters": int(model.backbone.count_params()),
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-root", type=Path, default=Path("data/local/citizen100_v17/hand_rgb"))
    parser.add_argument("--landmark-train", type=Path, default=Path("artifacts/generated/fusion_v17/landmark_train.npz"))
    parser.add_argument("--landmark-val", type=Path, default=Path("artifacts/generated/fusion_v17/landmark_val.npz"))
    parser.add_argument("--pretrained", type=Path, default=Path("artifacts/model_assets/movinet/movinet_a0_base"))
    parser.add_argument("--pretrained-archive", type=Path, default=Path("artifacts/model_assets/movinet/movinet_a0_base.tar.gz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/stage1_v17_sign_movinet_fusion"))
    parser.add_argument("--resolution", type=int, default=172)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--visual-loss-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--max-val-batches", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--split-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
