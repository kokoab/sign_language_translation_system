#!/usr/bin/env python3
"""Validate packed RGB crops through all three 100-gloss Stage-2 Core ML models."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import logging
from pathlib import Path
import statistics
import sys
import time

import coremltools as ct
import numpy as np
from PIL import Image
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.export_stage1_coreml_v17 import sha256_file, tree_sha256
from active.v17.export_stage2_frozen_encoder_coreml_v17 import archive_paths, paired_arrays
from active.v17.extract_hand_rgb_v17 import decode_packed_crops
from active.v17.model_stage2_v17 import load_stage2_context_adapted
from active.v17.schema_hand_rgb_v17 import CROP_SIZE
from active.v17.train_stage_2_v17 import collapse_ctc, edit_distance
from active.v17.stage2_stage3_contract_v17 import (
    CONTRACT_PATH,
    load_contract,
    make_stage2_output,
)
from scripts.evaluate_stage2_coreml_pipeline_v17 import frozen_metadata, model_names


LOG = logging.getLogger("evaluate_mobile_100gloss_coreml_v17")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def decode(logits: np.ndarray, windows: int, tokens_per_window: int) -> list[int]:
    value = np.asarray(logits).reshape(-1, logits.shape[-1])
    return collapse_ctc(value[: windows * tokens_per_window].argmax(-1))


def crop_embeddings(
    path: Path,
    image_model: ct.models.MLModel,
    image_input: str,
    image_output: str,
    maximum_windows: int,
) -> tuple[np.ndarray, int, int, list[float]]:
    """Decode the exact stored JPEGs and run every valid crop through Core ML."""
    with np.load(path, allow_pickle=False) as payload:
        offsets = payload["hand_jpeg_offsets"]
        valid = payload["hand_valid"].astype(np.bool_)
        windows = int(offsets.shape[0])
        if not 1 <= windows <= maximum_windows:
            raise ValueError(f"{path}: invalid window count {windows}")
        flat_offsets = offsets.reshape(windows * 16, 3, 2)
        crops = decode_packed_crops(payload["hand_jpeg_blob"], flat_offsets, CROP_SIZE)
    flat_valid = valid.reshape(windows * 16, 3)
    embeddings = np.zeros((1, maximum_windows, 16, 3, 512), dtype=np.float32)
    timings: list[float] = []
    for frame, view in np.argwhere(flat_valid):
        started = time.perf_counter()
        prediction = image_model.predict(
            {image_input: Image.fromarray(crops[int(frame), int(view)])}
        )
        timings.append((time.perf_counter() - started) * 1_000.0)
        window, within = divmod(int(frame), 16)
        embeddings[0, window, within, int(view)] = np.asarray(
            prediction[image_output], dtype=np.float32
        ).reshape(512)
    valid_count = int(flat_valid.sum())
    del crops, flat_valid
    gc.collect()
    return embeddings, windows, valid_count, timings


def run(args: argparse.Namespace) -> dict[str, object]:
    checked = (
        args.image_package,
        args.encoder_package,
        args.head_package,
        args.phrase_rgb_root,
        args.context_rgb_root,
    )
    if any("test" in {part.lower() for part in path.parts} for path in checked):
        raise ValueError("the mobile 100-gloss parity gate is validation-only")
    image_model = ct.models.MLModel(str(args.image_package), compute_units=ct.ComputeUnit.ALL)
    encoder = ct.models.MLModel(str(args.encoder_package), compute_units=ct.ComputeUnit.ALL)
    head = ct.models.MLModel(str(args.head_package), compute_units=ct.ComputeUnit.ALL)
    image_inputs, image_output = model_names(image_model)
    encoder_inputs, encoder_output = model_names(encoder)
    head_inputs, head_output = model_names(head)
    if len(image_inputs) != 1:
        raise ValueError("the MobileCLIP2 image package must have exactly one input")
    pytorch_head, _ = load_stage2_context_adapted(args.checkpoint)
    pytorch_head.eval()
    maximum_windows = pytorch_head.base.config.max_windows
    tokens_per_window = pytorch_head.base.config.tokens_per_window
    stage3_contract = load_contract()
    domains = (
        (
            args.phrase_rgb_root,
            args.phrase_hand_root,
            args.phrase_frozen_root,
        ),
        (
            args.context_rgb_root,
            args.context_hand_root,
            args.context_frozen_root,
        ),
    )
    paths: list[tuple[Path, Path, Path, Path]] = []
    for rgb_root, hand_root, frozen_root in domains:
        paths.extend(
            (path, rgb_root, hand_root, frozen_root) for path in archive_paths(rgb_root)
        )
    if args.maximum_samples:
        positions = np.rint(
            np.linspace(0, len(paths) - 1, min(args.maximum_samples, len(paths)))
        ).astype(int)
        paths = [paths[int(index)] for index in positions]

    stats = defaultdict(lambda: {"edits": 0, "tokens": 0, "exact": 0, "samples": 0})
    samples = windows_total = valid_crops = contract_outputs_validated = 0
    regenerated_vs_cached_decode_mismatches = 0
    regenerated_vs_pytorch_decode_mismatches = 0
    max_embedding_abs = 0.0
    minimum_embedding_cosine = 1.0
    max_frozen_abs = 0.0
    max_head_abs = 0.0
    crop_timings: list[float] = []
    pipeline_timings: list[float] = []
    started_all = time.monotonic()
    for index, (rgb_path, rgb_root, hand_root, frozen_root) in enumerate(paths, start=1):
        arrays, cached_frozen, windows = paired_arrays(
            rgb_path,
            rgb_root=rgb_root,
            hand_root=hand_root,
            frozen_root=frozen_root,
            maximum_windows=maximum_windows,
        )
        reference_tokens, metadata = frozen_metadata(rgb_path, rgb_root, frozen_root)
        started_pipeline = time.perf_counter()
        regenerated, regenerated_windows, crop_count, timings = crop_embeddings(
            rgb_path,
            image_model,
            image_inputs[0],
            image_output,
            maximum_windows,
        )
        if regenerated_windows != windows:
            raise ValueError(f"{rgb_path}: regenerated window mismatch")
        crop_timings.extend(timings)
        valid_crops += crop_count
        mask = arrays[2].astype(np.bool_)
        cached_embeddings = arrays[1]
        if mask.any():
            current = regenerated[mask]
            cached = cached_embeddings[mask]
            max_embedding_abs = max(
                max_embedding_abs, float(np.max(np.abs(current - cached)))
            )
            cosines = np.sum(current * cached, axis=-1) / np.maximum(
                np.linalg.norm(current, axis=-1) * np.linalg.norm(cached, axis=-1),
                1e-12,
            )
            minimum_embedding_cosine = min(minimum_embedding_cosine, float(cosines.min()))
        regenerated_arrays = (arrays[0], regenerated, arrays[2], arrays[3], arrays[4])
        regenerated_frozen = np.asarray(
            encoder.predict(dict(zip(encoder_inputs, regenerated_arrays)))[encoder_output],
            dtype=np.float32,
        ).reshape(cached_frozen.shape)
        max_frozen_abs = max(
            max_frozen_abs,
            float(np.max(np.abs(regenerated_frozen[:, :windows] - cached_frozen[:, :windows]))),
        )
        regenerated_logits = np.asarray(
            head.predict(
                {head_inputs[0]: regenerated_frozen, head_inputs[1]: arrays[-1]}
            )[head_output],
            dtype=np.float32,
        )
        with torch.inference_mode():
            pytorch_logits, _ = pytorch_head(
                torch.from_numpy(cached_frozen), torch.from_numpy(arrays[-1]) > 0.5
            )
        pytorch_value = pytorch_logits.numpy()
        regenerated_value = regenerated_logits.reshape(pytorch_value.shape)
        max_head_abs = max(
            max_head_abs, float(np.max(np.abs(regenerated_value - pytorch_value)))
        )
        hypothesis = decode(regenerated_value, windows, tokens_per_window)
        make_stage2_output(
            utterance_id=str(metadata["source_item_id"]),
            token_indices=[value + 1 for value in hypothesis],
            window_count=windows,
            contract=stage3_contract,
        )
        contract_outputs_validated += 1
        reference_hypothesis = decode(pytorch_value, windows, tokens_per_window)
        cached_coreml_frozen = np.asarray(
            encoder.predict(dict(zip(encoder_inputs, arrays)))[encoder_output], dtype=np.float32
        ).reshape(cached_frozen.shape)
        cached_logits = np.asarray(
            head.predict(
                {head_inputs[0]: cached_coreml_frozen, head_inputs[1]: arrays[-1]}
            )[head_output],
            dtype=np.float32,
        ).reshape(pytorch_value.shape)
        cached_hypothesis = decode(cached_logits, windows, tokens_per_window)
        regenerated_vs_cached_decode_mismatches += int(hypothesis != cached_hypothesis)
        regenerated_vs_pytorch_decode_mismatches += int(hypothesis != reference_hypothesis)
        source = str(metadata["source"])
        row = stats[source]
        row["edits"] += edit_distance(reference_tokens, hypothesis)
        row["tokens"] += len(reference_tokens)
        row["exact"] += int(reference_tokens == hypothesis)
        row["samples"] += 1
        samples += 1
        windows_total += windows
        pipeline_timings.append((time.perf_counter() - started_pipeline) * 1_000.0)
        if index == 1 or index % 10 == 0 or index == len(paths):
            LOG.info(
                "%d/%d samples, %d valid crops, %.1fs elapsed",
                index,
                len(paths),
                valid_crops,
                time.monotonic() - started_all,
            )
        del arrays, cached_frozen, regenerated, regenerated_frozen
        gc.collect()

    if regenerated_vs_cached_decode_mismatches:
        raise ValueError(
            f"fresh RGB embeddings changed {regenerated_vs_cached_decode_mismatches} decoded sequences"
        )
    metrics = {
        source: {
            "edits": value["edits"],
            "tokens": value["tokens"],
            "wer": value["edits"] / max(1, value["tokens"]),
            "exact_sequences": value["exact"],
            "samples": value["samples"],
            "sequence_accuracy": value["exact"] / max(1, value["samples"]),
        }
        for source, value in sorted(stats.items())
    }
    result = {
        "format": "slt_mobile_100gloss_coreml_validation_v17",
        "version": 1,
        "image_package": args.image_package.as_posix(),
        "image_package_tree_sha256": tree_sha256(args.image_package),
        "encoder_package": args.encoder_package.as_posix(),
        "encoder_package_tree_sha256": tree_sha256(args.encoder_package),
        "head_package": args.head_package.as_posix(),
        "head_package_tree_sha256": tree_sha256(args.head_package),
        "validation_samples": samples,
        "validation_windows": windows_total,
        "valid_rgb_crops_encoded": valid_crops,
        "embedding_max_abs_vs_cached_float16": max_embedding_abs,
        "embedding_min_cosine_vs_cached_float16": minimum_embedding_cosine,
        "frozen_feature_max_abs_vs_cached_float16": max_frozen_abs,
        "logit_max_abs_vs_cached_pytorch": max_head_abs,
        "fresh_rgb_vs_cached_coreml_decode_mismatches": regenerated_vs_cached_decode_mismatches,
        "fresh_rgb_vs_cached_pytorch_decode_mismatches": regenerated_vs_pytorch_decode_mismatches,
        "stage2_to_stage3_contract": CONTRACT_PATH.as_posix(),
        "stage2_to_stage3_contract_sha256": sha256_file(CONTRACT_PATH),
        "stage2_to_stage3_outputs_validated": contract_outputs_validated,
        "metrics": metrics,
        "crop_coreml_latency_ms_median": statistics.median(crop_timings),
        "crop_coreml_latency_ms_p90": percentile(crop_timings, 0.9),
        "packed_crop_to_ctc_latency_ms_median": statistics.median(pipeline_timings),
        "packed_crop_to_ctc_latency_ms_p90": percentile(pipeline_timings, 0.9),
        "wall_seconds": time.monotonic() - started_all,
        "execution_environment": "mac_host_coreml",
        "pipeline_boundary": "packed_stage2_rgb_crops_and_landmarks_to_ctc_gloss_indices",
        "camera_to_gloss_end_to_end": False,
        "all_mobile_neural_models_in_coreml": True,
        "hardware_performance_claim": False,
        "thermals_interpretable": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "test_evaluated": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-package", type=Path, default=Path("artifacts/coreml/MobileCLIP2S0ImageEncoderV17FP32.mlpackage"))
    parser.add_argument("--encoder-package", type=Path, default=Path("artifacts/coreml/Stage2FrozenEncoderV17FP32.mlpackage"))
    parser.add_argument("--head-package", type=Path, default=Path("artifacts/coreml/Stage2CompactContextV17FP32.mlpackage"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/stage2_v17_compact_context_student_v1/model.pth"))
    parser.add_argument("--phrase-rgb-root", type=Path, default=Path("data/local/stage2_v17_multimodal"))
    parser.add_argument("--phrase-hand-root", type=Path, default=Path("data/local/stage2_v17_hand_mobileclip2"))
    parser.add_argument("--phrase-frozen-root", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    parser.add_argument("--context-rgb-root", type=Path, default=Path("data/local/stage2_v17_asllrp_segmented_validation_multimodal"))
    parser.add_argument("--context-hand-root", type=Path, default=Path("data/local/stage2_v17_asllrp_segmented_validation_hand_mobileclip2"))
    parser.add_argument("--context-frozen-root", type=Path, default=Path("data/local/stage2_v17_asllrp_segmented_validation_frozen_features"))
    parser.add_argument("--maximum-samples", type=int, default=0)
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/mobile_100gloss_v17/full_coreml_validation.json"))
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
