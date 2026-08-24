#!/usr/bin/env python3
"""Run the frozen v17 orientation benchmark in an actual iOS Simulator runtime.

The measurements exercise the iOS app, Apple Vision, and compiled Core ML program,
but they run on Mac hardware. The generated evidence therefore rejects physical-
iPhone latency, ANE, thermal, or sustained-performance claims by construction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from active.v17.extract_v17 import (
    AppleVisionDetector, extract_video_v17, save_v17_result,
)
from active.v17.schema_hand_rgb_v17 import HandRGBV17Config
from active.v17.schema_stage2_features_v17 import Stage2FeatureV17Config
from active.v17.schema_v17 import V17Config
from scripts.extract_stage2_multimodal_v17 import extract_row as extract_stage2_row

PROJECT = REPO_ROOT / "mobile_benchmark/OrientationBenchmarkV17/OrientationBenchmarkV17.xcodeproj"
APP_BUILD_ROOT = REPO_ROOT / "artifacts/generated/orientation_benchmark_v17_release_simulator"
APP_BUNDLE = APP_BUILD_ROOT / "OrientationBenchmarkV17.app"
MODEL_MANIFEST = (
    REPO_ROOT
    / "mobile_benchmark/OrientationBenchmarkV17/OrientationBenchmarkV17/Stage2MobileV17_manifest.json"
)
STAGE3_MANIFEST = REPO_ROOT / "active/v17/stage3_mobile_naturalizer_manifest_v17.json"
DEFAULT_SOURCE = (
    REPO_ROOT
    / "data/local/citizen100_v17/raw/val/HELLO/020030442376253177-HELLO.mp4"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/reports/orientation_v17_simulator_benchmark"
DEFAULT_GENERATED_ROOT = REPO_ROOT / "artifacts/generated/orientation_v17_simulator_inputs"
ANGLES = (0.0, 17.0, 37.0, 73.0, 90.0, 123.0, 180.0, 270.0)
EXACT_CORRECTIONS = {0.0: 0, 90.0: 270, 180.0: 180, 270.0: 90}
BUNDLE_ID = "com.local.slt.OrientationBenchmarkV17"
DEVICE_NAME = "SLT Orientation Benchmark iPhone 13"
DEVICE_TYPE = "com.apple.CoreSimulator.SimDeviceType.iPhone-13"
PREFERRED_RUNTIME_VERSION = "26.2"
SIMULATOR_VISION_LIMITATION = (
    "The installed iOS Simulator runtime omits Apple Vision pose Espresso weight files; "
    "features were extracted by the same v17 Apple Vision pipeline on the macOS host."
)


class SimulatorBenchmarkError(RuntimeError):
    """A fail-closed simulator setup, execution, or evidence error."""


def run(
    command: list[str],
    *,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def command_output(command: list[str]) -> str:
    result = run(command, capture=True)
    return result.stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def angle_slug(angle: float) -> str:
    return str(int(angle)) if angle.is_integer() else f"{angle:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def residual_roll(angle: float, correction: float) -> float:
    value = (angle + correction) % 360.0
    if value > 180.0:
        value -= 360.0
    return value


def rotation_filter(angle: float) -> str | None:
    normalized = angle % 360.0
    if normalized == 0.0:
        return None
    if normalized == 90.0:
        return "transpose=clock"
    if normalized == 180.0:
        return "hflip,vflip"
    if normalized == 270.0:
        return "transpose=cclock"
    radians = normalized * math.pi / 180.0
    value = f"{radians:.15f}"
    return (
        f"rotate={value}:ow=ceil(rotw({value})/2)*2:"
        f"oh=ceil(roth({value})/2)*2:c=black"
    )


def generate_video(source: Path, destination: Path, angle: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    transform = rotation_filter(angle)
    if transform is None:
        shutil.copy2(source, destination)
        return
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            transform,
            "-map_metadata",
            "-1",
            "-metadata:s:v:0",
            "rotate=0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    probe = json.loads(
        command_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames",
                "-of",
                "json",
                str(destination),
            ]
        )
    )
    streams = probe.get("streams", [])
    if len(streams) != 1 or int(streams[0].get("width", 0)) < 1 or int(streams[0].get("height", 0)) < 1:
        raise SimulatorBenchmarkError(f"generated video failed decode probe: {destination}")


def simulator_runtimes() -> list[dict[str, Any]]:
    payload = json.loads(command_output(["xcrun", "simctl", "list", "runtimes", "-j"]))
    return list(payload.get("runtimes", []))


def version_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def find_runtime() -> tuple[str, str] | None:
    compatible: list[tuple[tuple[int, ...], str, str]] = []
    preferred = version_key(PREFERRED_RUNTIME_VERSION)
    for runtime in simulator_runtimes():
        name = str(runtime.get("name", ""))
        version = str(runtime.get("version", name.removeprefix("iOS ")))
        parsed = version_key(version)
        if (
            name.startswith("iOS ")
            and runtime.get("isAvailable", True)
            and parsed >= preferred
            and parsed[0] == preferred[0]
        ):
            compatible.append((parsed, str(runtime["identifier"]), version))
    if not compatible:
        return None
    _, identifier, version = min(compatible)
    return identifier, version


def ensure_runtime(allow_install: bool) -> tuple[str, str]:
    runtime = find_runtime()
    if runtime:
        return runtime
    if not allow_install:
        raise SimulatorBenchmarkError(
            f"a compatible iOS {PREFERRED_RUNTIME_VERSION} runtime is not installed"
        )
    exact = run(
        ["xcodebuild", "-downloadPlatform", "iOS", "-buildVersion", PREFERRED_RUNTIME_VERSION],
        check=False,
    )
    if exact.returncode != 0:
        print(
            f"iOS {PREFERRED_RUNTIME_VERSION} is unavailable in Apple's catalog; "
            "installing Xcode's current compatible iOS runtime.",
            flush=True,
        )
        run(["xcodebuild", "-downloadPlatform", "iOS"])
    runtime = find_runtime()
    if not runtime:
        raise SimulatorBenchmarkError(
            f"Xcode finished but no compatible iOS {PREFERRED_RUNTIME_VERSION} runtime is available"
        )
    return runtime


def ensure_device(runtime_id: str) -> tuple[str, str]:
    payload = json.loads(command_output(["xcrun", "simctl", "list", "devices", "-j"]))
    devices = payload.get("devices", {}).get(runtime_id, [])
    for device in devices:
        if device.get("name") == DEVICE_NAME and device.get("isAvailable", True):
            return str(device["udid"]), str(device.get("state", "Shutdown"))
    udid = command_output(
        ["xcrun", "simctl", "create", DEVICE_NAME, DEVICE_TYPE, runtime_id]
    ).strip()
    if not udid:
        raise SimulatorBenchmarkError("simctl did not return a new device UDID")
    return udid, "Shutdown"


def boot_device(udid: str, state: str) -> None:
    if state != "Booted":
        run(["xcrun", "simctl", "boot", udid])
    run(["xcrun", "simctl", "bootstatus", udid, "-b"])


def build_and_install(udid: str) -> None:
    run(
        [
            "xcodebuild",
            "-project",
            str(PROJECT),
            "-target",
            "OrientationBenchmarkV17",
            "-sdk",
            "iphonesimulator26.2",
            "-configuration",
            "Release",
            "CODE_SIGNING_ALLOWED=NO",
            f"CONFIGURATION_BUILD_DIR={APP_BUILD_ROOT}",
            "build",
        ]
    )
    if not APP_BUNDLE.is_dir():
        raise SimulatorBenchmarkError(f"missing built app: {APP_BUNDLE}")
    run(["xcrun", "simctl", "install", udid, str(APP_BUNDLE)])


def extract_host_features(
    video: Path,
    destination: Path,
    landmark_archive: Path,
    detector: AppleVisionDetector,
) -> dict[str, Any]:
    result = extract_video_v17(
        video,
        V17Config(),
        rotation="auto",
        detector=detector,
        vision_auto_orient=True,
    )
    if result is None:
        raise SimulatorBenchmarkError(f"host Apple Vision extraction failed: {video}")
    features = np.asarray(result.features, dtype="<f4")
    if features.shape != (32, 61, 5) or not np.isfinite(features).all():
        raise SimulatorBenchmarkError(
            f"host feature contract failed for {video}: {features.shape}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.tofile(destination)
    save_v17_result(landmark_archive, result, V17Config())
    metadata = result.metadata
    diagnostics = result.diagnostics
    score_groups = metadata.get("vision_orientation_scores") or {}
    scores = {
        f"{angle}_{name}": float(value)
        for angle, group in score_groups.items()
        for name, value in group.items()
    }
    return {
        "sourceFrames": int(metadata["source_frames_before_hand_trim"]),
        "trimmedFrames": int(metadata["source_frames_processed"]),
        "observedHandFrames": int(diagnostics["observed_hand_frames"]),
        "handPresenceFraction": float(diagnostics["hand_presence_fraction"]),
        "facePresenceFraction": float(diagnostics["face_presence_fraction"]),
        "bodyPresenceFraction": float(diagnostics["body_presence_fraction"]),
        "extractionMilliseconds": float(diagnostics["elapsed_seconds"]) * 1_000.0,
        "visionCoarseRotationClockwise": int(
            round(float(metadata.get("vision_coarse_rotation_clockwise", 0.0)))
        ),
        "visionOrientationScores": scores,
    }


def write_stage2_crop_bundle(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write the small fail-closed crop container consumed by the Swift app."""
    offsets = np.asarray(arrays["hand_jpeg_offsets"], dtype=np.int64)
    valid = np.asarray(arrays["hand_valid"], dtype=np.float32)
    boxes = np.asarray(arrays["hand_boxes_normalized"], dtype=np.float32)
    blob = np.asarray(arrays["hand_jpeg_blob"], dtype=np.uint8)
    windows = int(offsets.shape[0])
    if not 1 <= windows <= 8 or offsets.shape != (windows, 16, 3, 2) or valid.shape != (windows, 16, 3) or boxes.shape != (windows, 16, 3, 4):
        raise SimulatorBenchmarkError("unexpected Stage-2 hand-crop archive shape")
    payload = bytearray(b"SLTHRGB1")
    payload.extend(struct.pack("<I", windows))
    payload.extend(valid.astype("<f4", copy=False).tobytes(order="C"))
    payload.extend(boxes.astype("<f4", copy=False).tobytes(order="C"))
    for window in range(windows):
        for frame in range(16):
            for view in range(3):
                start, length = [int(value) for value in offsets[window, frame, view]]
                if not bool(valid[window, frame, view]):
                    if start != -1 or length != 0:
                        raise SimulatorBenchmarkError("invalid crop has nonempty JPEG offsets")
                    payload.extend(struct.pack("<I", 0))
                    continue
                if start < 0 or length <= 0 or start + length > len(blob):
                    raise SimulatorBenchmarkError("valid crop has invalid JPEG offsets")
                payload.extend(struct.pack("<I", length))
                payload.extend(blob[start : start + length].tobytes())
    path.write_bytes(payload)


def stage2_host_diagnostics(metadata: dict[str, Any]) -> dict[str, Any]:
    diagnostics = list(metadata["window_diagnostics"])
    observed = sum(int(item.get("observed_hand_frames", 0)) for item in diagnostics)
    return {
        "sourceFrames": int(metadata["sampled_source_frames"]),
        "trimmedFrames": int(metadata["sampled_source_frames"]),
        "observedHandFrames": observed,
        "handPresenceFraction": float(np.mean([
            item.get("hand_presence_fraction", 0.0) for item in diagnostics
        ])),
        "facePresenceFraction": float(np.mean([
            item.get("face_presence_fraction", 0.0) for item in diagnostics
        ])),
        "bodyPresenceFraction": float(np.mean([
            item.get("body_presence_fraction", 0.0) for item in diagnostics
        ])),
        "extractionMilliseconds": 1000.0 * float(sum(
            item.get("elapsed_seconds", 0.0) for item in diagnostics
        )),
        "visionCoarseRotationClockwise": int(
            round(float(metadata.get("vision_coarse_rotation_clockwise", 0.0)))
        ),
        "visionOrientationScores": {
            f"{angle}_{name}": float(value)
            for angle, group in (metadata.get("vision_orientation_scores") or {}).items()
            for name, value in group.items()
        },
    }


def prepare_suite(
    source: Path,
    generated_root: Path,
    suite_id: str,
    iterations: int,
) -> tuple[dict[str, Any], Path]:
    if "test" in {part.lower() for part in source.parts}:
        raise SimulatorBenchmarkError("test paths are forbidden")
    if not source.is_file():
        raise SimulatorBenchmarkError(f"missing source video: {source}")
    suite_root = generated_root / suite_id
    entries: list[dict[str, Any]] = []
    landmark_detector = AppleVisionDetector(V17Config().minimum_point_confidence)
    hand_detector = AppleVisionDetector(V17Config().minimum_point_confidence)
    stage2_config = Stage2FeatureV17Config(maximum_source_frames=256)
    hand_config = HandRGBV17Config()
    for angle in ANGLES:
        filename = f"hello-roll-{angle_slug(angle)}.mp4"
        destination = suite_root / filename
        generate_video(source, destination, angle)
        feature_filename = f"hello-roll-{angle_slug(angle)}.v17.f32"
        feature_path = suite_root / feature_filename
        generated_sha = sha256_file(destination)
        stage2_arrays, stage2_metadata = extract_stage2_row(
            {
                "source_item_id": f"hello-roll-{angle_slug(angle)}",
                "source": "citizen_val_simulator_orientation",
                "source_group": "citizen_val_simulator_orientation",
                "role": "validation",
                "video_path": destination.as_posix(),
                "video_sha256": generated_sha,
                "target_sequence": ["HELLO"],
                "target_indices": [14],
                "zero_lip_nodes": False,
                "lip_supervision": "full",
            },
            landmark_detector,
            hand_detector,
            stage2_config,
            hand_config,
            "simulator_orientation_not_a_training_manifest",
        )
        np.asarray(stage2_arrays["landmarks"], dtype="<f4").tofile(feature_path)
        host_diagnostics = stage2_host_diagnostics(stage2_metadata)
        crop_bundle_filename = f"hello-roll-{angle_slug(angle)}.stage2_hand_rgb.bin"
        crop_bundle_path = suite_root / crop_bundle_filename
        write_stage2_crop_bundle(crop_bundle_path, stage2_arrays)
        entries.append(
            {
                "angleDegreesClockwise": angle,
                "relativeVideoPath": f"simulator_benchmark_inputs/{suite_id}/{filename}",
                "generatedVideoSHA256": generated_sha,
                "relativeFeaturePath": (
                    f"simulator_benchmark_inputs/{suite_id}/{feature_filename}"
                ),
                "featureSHA256": sha256_file(feature_path),
                "relativeHandCropBundlePath": (
                    f"simulator_benchmark_inputs/{suite_id}/{crop_bundle_filename}"
                ),
                "handCropBundleSHA256": sha256_file(crop_bundle_path),
                "hostDiagnostics": host_diagnostics,
            }
        )
    payload = {
        "format": "slt_v17_ios_simulator_benchmark_suite",
        "version": 1,
        "suiteID": suite_id,
        "expectedLabel": "HELLO",
        "iterations": iterations,
        "sourceVideoSHA256": sha256_file(source),
        "citizenTestAccessed": False,
        "semlexTestAccessed": False,
        "featureExtractionEnvironment": "host_macos_apple_vision",
        "endToEndPipeline": True,
        "videoFileToGlossEndToEnd": False,
        "cameraToGlossEndToEnd": False,
        "allMobileNeuralModelsInCoreML": True,
        "simulatorVisionLimitation": SIMULATOR_VISION_LIMITATION,
        "entries": entries,
    }
    manifest_path = suite_root / f"simulator-suite-{suite_id}.json"
    write_json(manifest_path, payload)
    return payload, manifest_path


def install_suite_in_container(
    udid: str,
    suite: dict[str, Any],
    manifest_path: Path,
    generated_root: Path,
) -> tuple[Path, str]:
    data_container = Path(
        command_output(
            ["xcrun", "simctl", "get_app_container", udid, BUNDLE_ID, "data"]
        ).strip()
    )
    documents = data_container / "Documents"
    suite_id = str(suite["suiteID"])
    target_inputs = documents / "simulator_benchmark_inputs" / suite_id
    target_inputs.mkdir(parents=True, exist_ok=True)
    for entry in suite["entries"]:
        for key in ("relativeVideoPath", "relativeFeaturePath", "relativeHandCropBundlePath"):
            filename = Path(str(entry[key])).name
            source = generated_root / suite_id / filename
            shutil.copy2(source, target_inputs / filename)
    manifest_name = manifest_path.name
    shutil.copy2(manifest_path, documents / manifest_name)
    reports = documents / "benchmark_reports"
    reports.mkdir(parents=True, exist_ok=True)
    return reports, manifest_name


def wait_for_aggregate(
    reports_directory: Path,
    suite_id: str,
    timeout_seconds: int,
) -> Path:
    aggregate = reports_directory / f"simulator-suite-{suite_id}-aggregate.json"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if aggregate.is_file():
            try:
                json.loads(aggregate.read_text())
                return aggregate
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(2)
    raise SimulatorBenchmarkError(
        f"simulator suite did not finish within {timeout_seconds} seconds"
    )


def validate_and_collect(
    suite: dict[str, Any],
    aggregate_path: Path,
    destination: Path,
    runtime_id: str,
    udid: str,
) -> dict[str, Any]:
    aggregate = json.loads(aggregate_path.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(aggregate_path, destination / aggregate_path.name)
    app_manifest = json.loads(MODEL_MANIFEST.read_text())
    stage3_manifest_sha256 = sha256_file(STAGE3_MANIFEST)
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    if aggregate.get("format") != "slt_v17_ios_simulator_benchmark_aggregate":
        errors.append("aggregate format mismatch")
    if aggregate.get("executionEnvironment") != "simulator":
        errors.append("aggregate is not simulator-labeled")
    if aggregate.get("simulatorDeviceName") != DEVICE_NAME:
        errors.append(
            f"aggregate device {aggregate.get('simulatorDeviceName')!r} is not {DEVICE_NAME!r}"
        )
    if aggregate.get("simulatorModelIdentifier") != "iPhone14,5":
        errors.append("aggregate simulator model is not iPhone 13 (iPhone14,5)")
    if aggregate.get("hardwarePerformanceClaim") is not False:
        errors.append("aggregate makes a hardware-performance claim")
    if aggregate.get("thermalsInterpretable") is not False:
        errors.append("aggregate treats simulator thermals as interpretable")
    if aggregate.get("citizenTestAccessed") is not False or aggregate.get("semlexTestAccessed") is not False:
        errors.append("aggregate test-access flags are not false")
    aggregate_entries = aggregate.get("entries", [])
    if len(aggregate_entries) != len(ANGLES):
        errors.append(f"expected {len(ANGLES)} aggregate entries, found {len(aggregate_entries)}")
    for entry in aggregate_entries:
        angle = float(entry.get("angleDegreesClockwise", float("nan")))
        if entry.get("success") is not True or not entry.get("reportFilename"):
            errors.append(f"{angle:g}° app run failed: {entry.get('error')}")
            continue
        source_report = aggregate_path.parent / str(entry["reportFilename"])
        if not source_report.is_file():
            errors.append(f"{angle:g}° report is missing")
            continue
        destination_report = destination / source_report.name
        shutil.copy2(source_report, destination_report)
        report = json.loads(destination_report.read_text())
        reports.append(report)
        prefix = f"{angle:g}°"
        if report.get("executionEnvironment") != "simulator":
            errors.append(f"{prefix} is not simulator-labeled")
        if report.get("hardwarePerformanceClaim") is not False:
            errors.append(f"{prefix} makes a hardware-performance claim")
        if report.get("thermalsInterpretable") is not False:
            errors.append(f"{prefix} treats thermals as interpretable")
        if report.get("iterations") != suite["iterations"]:
            errors.append(f"{prefix} iteration count mismatch")
        if report.get("extractionSucceeded") is not True:
            errors.append(f"{prefix} did not extract")
        if report.get("extractionExecutionEnvironment") != "host_macos_apple_vision":
            errors.append(f"{prefix} extraction environment is not host Apple Vision")
        if report.get("endToEndPipeline") is not True:
            errors.append(f"{prefix} did not run the complete mobile neural pipeline")
        if report.get("allMobileNeuralModelsInCoreML") is not True:
            errors.append(f"{prefix} did not run all three neural models in Core ML")
        if report.get("videoFileToGlossEndToEnd") is not False:
            errors.append(f"{prefix} incorrectly claims simulator file-video extraction evidence")
        if report.get("cameraToGlossEndToEnd") is not False:
            errors.append(f"{prefix} incorrectly claims simulator camera-to-gloss evidence")
        if report.get("predictionAggregation") != "mean_logits_across_timed_iterations_then_greedy_ctc":
            errors.append(f"{prefix} prediction aggregation is not the pinned sustained rule")
        if sum(int(value) for value in report.get("sequenceVoteCounts", {}).values()) != suite["iterations"]:
            errors.append(f"{prefix} decoded-sequence vote count mismatch")
        if report.get("simulatorVisionLimitation") != SIMULATOR_VISION_LIMITATION:
            errors.append(f"{prefix} simulator Vision limitation is missing")
        if report.get("expectedLabel") != "HELLO" or report.get("predictedLabel") != "HELLO":
            errors.append(f"{prefix} prediction is not HELLO")
        if report.get("stage3NaturalEnglish") != "Hello.":
            errors.append(f"{prefix} Stage-3 natural English is not Hello.")
        if report.get("stage3LiteralEnglish") != "Hello.":
            errors.append(f"{prefix} Stage-3 literal English is not Hello.")
        if report.get("stage3RenderingMode") != "reviewed_template":
            errors.append(f"{prefix} Stage-3 did not use the reviewed HELLO template")
        if report.get("stage3SafeFallbackUsed") is not False:
            errors.append(f"{prefix} Stage-3 unexpectedly used the fallback")
        if report.get("stage3NaturalizerManifestSHA256") != stage3_manifest_sha256:
            errors.append(f"{prefix} Stage-3 manifest hash mismatch")
        if report.get("correct") is not True:
            errors.append(f"{prefix} is not correct")
        if report.get("checkpointSHA256") != app_manifest["checkpointSHA256"]:
            errors.append(f"{prefix} checkpoint hash mismatch")
        expected_hashes = {
            "imageEncoderPackageTreeSHA256": "imageEncoderPackageTreeSHA256",
            "frozenEncoderPackageTreeSHA256": "frozenEncoderPackageTreeSHA256",
            "contextHeadPackageTreeSHA256": "contextHeadPackageTreeSHA256",
            "vocabularyManifestSHA256": "vocabularyManifestSHA256",
        }
        for report_key, manifest_key in expected_hashes.items():
            if report.get(report_key) != app_manifest[manifest_key]:
                errors.append(f"{prefix} {report_key} mismatch")
        if report.get("sourceVideoSHA256") != suite["sourceVideoSHA256"]:
            errors.append(f"{prefix} source hash mismatch")
        expected_entry = next(
            candidate for candidate in suite["entries"]
            if float(candidate["angleDegreesClockwise"]) == angle
        )
        if report.get("generatedVideoSHA256") != expected_entry["generatedVideoSHA256"]:
            errors.append(f"{prefix} generated-video hash mismatch")
        if report.get("featureSHA256") != expected_entry["featureSHA256"]:
            errors.append(f"{prefix} host-feature hash mismatch")
        if report.get("handCropBundleSHA256") != expected_entry["handCropBundleSHA256"]:
            errors.append(f"{prefix} hand-crop bundle hash mismatch")
        correction = int(report.get("diagnostics", {}).get("visionCoarseRotationClockwise", -1))
        if angle in EXACT_CORRECTIONS and correction != EXACT_CORRECTIONS[angle]:
            errors.append(
                f"{prefix} correction {correction} != {EXACT_CORRECTIONS[angle]}"
            )
        actual_residual = float(report.get("residualRollDegrees", float("inf")))
        expected_residual = residual_roll(angle, correction)
        if not math.isclose(actual_residual, expected_residual, abs_tol=1e-6):
            errors.append(f"{prefix} residual-roll value mismatch")
        if abs(actual_residual) > 45.0 + 1e-6:
            errors.append(f"{prefix} residual roll exceeds 45 degrees")
    report_angles = sorted(float(item["inputRotationDegreesClockwise"]) for item in reports)
    if report_angles != sorted(ANGLES):
        errors.append(f"report angle inventory mismatch: {report_angles}")
    medians = [float(item["medianInferenceMilliseconds"]) for item in reports]
    p90s = [float(item["p90InferenceMilliseconds"]) for item in reports]
    extractions = [float(item["diagnostics"]["extractionMilliseconds"]) for item in reports]
    result = {
        "format": "slt_v17_ios_simulator_benchmark_result",
        "version": 1,
        "createdUTC": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "executionEnvironment": "simulator",
        "hardwarePerformanceClaim": False,
        "thermalsInterpretable": False,
        "limitations": [
            "Core ML executes on Mac simulator hardware, not an iPhone SoC or Neural Engine.",
            "Simulator latency, resident memory, and thermal state are not physical-iPhone measurements.",
            SIMULATOR_VISION_LIMITATION,
        ],
        "runtimeIdentifier": runtime_id,
        "simulatorUDID": udid,
        "simulatorDeviceName": aggregate.get("simulatorDeviceName"),
        "simulatorModelIdentifier": aggregate.get("simulatorModelIdentifier"),
        "simulatorRuntimeVersion": aggregate.get("simulatorRuntimeVersion"),
        "hostArchitecture": platform.machine(),
        "sourceVideo": str(DEFAULT_SOURCE.relative_to(REPO_ROOT)),
        "sourceVideoSHA256": suite["sourceVideoSHA256"],
        "split": "citizen_official_val",
        "citizenTestAccessed": False,
        "semlexTestAccessed": False,
        "expectedLabel": "HELLO",
        "anglesDegreesClockwise": list(ANGLES),
        "iterationsPerAngle": suite["iterations"],
        "conditionsCompleted": len(reports),
        "conditionsCorrect": sum(bool(item.get("correct")) for item in reports),
        "meanMedianInferenceMilliseconds": sum(medians) / len(medians) if medians else None,
        "maximumP90InferenceMilliseconds": max(p90s) if p90s else None,
        "meanExtractionMilliseconds": sum(extractions) / len(extractions) if extractions else None,
        "featureExtractionEnvironment": "host_macos_apple_vision",
        "endToEndPipeline": True,
        "videoFileToGlossEndToEnd": False,
        "cameraToGlossEndToEnd": False,
        "allMobileNeuralModelsInCoreML": True,
        "candidateID": app_manifest["candidateID"],
        "checkpointSHA256": app_manifest["checkpointSHA256"],
        "imageEncoderPackageTreeSHA256": app_manifest["imageEncoderPackageTreeSHA256"],
        "frozenEncoderPackageTreeSHA256": app_manifest["frozenEncoderPackageTreeSHA256"],
        "contextHeadPackageTreeSHA256": app_manifest["contextHeadPackageTreeSHA256"],
        "vocabularyManifestSHA256": app_manifest["vocabularyManifestSHA256"],
        "stage3NaturalEnglish": "Hello.",
        "stage3LiteralEnglish": "Hello.",
        "stage3RenderingMode": "reviewed_template",
        "stage3SafeFallbackUsed": False,
        "stage3NaturalizerManifestSHA256": stage3_manifest_sha256,
        "aggregateFilename": aggregate_path.name,
        "reportFilenames": [item["reportFilename"] for item in aggregate_entries if item.get("reportFilename")],
        "acceptanceErrors": errors,
    }
    write_json(destination / "result.json", result)
    if errors:
        raise SimulatorBenchmarkError("; ".join(errors))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument(
        "--no-install-runtime",
        action="store_true",
        help="Fail instead of asking Xcode to install a compatible iOS runtime",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 20:
        raise SimulatorBenchmarkError("iterations must be at least 20")
    if args.timeout_seconds < 60:
        raise SimulatorBenchmarkError("timeout must be at least 60 seconds")
    source = args.source.resolve()
    runtime_id, runtime_version = ensure_runtime(
        allow_install=not args.no_install_runtime
    )
    suite_id = datetime.now(timezone.utc).strftime(
        f"orientation-v17-ios{runtime_version.replace('.', '-')}-%Y%m%dT%H%M%SZ"
    )
    suite, manifest_path = prepare_suite(
        source, args.generated_root.resolve(), suite_id, args.iterations
    )
    udid, state = ensure_device(runtime_id)
    boot_device(udid, state)
    build_and_install(udid)
    reports_directory, manifest_name = install_suite_in_container(
        udid, suite, manifest_path, args.generated_root.resolve()
    )
    log_root = args.output_root.resolve() / suite_id
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / "simulator_stdout.log"
    stderr_path = log_root / "simulator_stderr.log"
    run(
        [
            "xcrun",
            "simctl",
            "launch",
            "--terminate-running-process",
            f"--stdout={stdout_path}",
            f"--stderr={stderr_path}",
            udid,
            BUNDLE_ID,
            "--benchmark-suite",
            manifest_name,
        ]
    )
    aggregate_path = wait_for_aggregate(
        reports_directory, suite_id, args.timeout_seconds
    )
    try:
        result = validate_and_collect(
            suite, aggregate_path, log_root, runtime_id, udid
        )
    finally:
        run(["xcrun", "simctl", "terminate", udid, BUNDLE_ID], check=False)
    write_json(args.output_root.resolve() / "latest_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (SimulatorBenchmarkError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
