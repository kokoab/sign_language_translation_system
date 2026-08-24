#!/usr/bin/env python3
"""Validate the complete Swift video-to-English path on Mac-host Apple frameworks.

This is functional parity evidence for code compiled into the iOS app. It is not an
iPhone latency, memory, thermal, camera, or ANE measurement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    ROOT
    / "artifacts/generated/orientation_v17_simulator_inputs"
    / "orientation-v17-ios26-3-1-20260824T071611Z"
)
DEFAULT_APP = (
    ROOT
    / "mobile_benchmark/OrientationBenchmarkV17/artifacts/generated"
    / "stage3_mobile_release_simulator/OrientationBenchmarkV17.app"
)
DEFAULT_BINARY = ROOT / "artifacts/generated/ios_preprocessor_validation/validate-ios-preprocessor"
DEFAULT_REPORT = ROOT / "artifacts/reports/stage3_mobile_v17/swift_video_to_english_validation.json"
ANGLES = (0, 17, 37, 73, 90, 123, 180, 270)
EXACT_CORRECTIONS = {0: 0, 90: 270, 180: 180, 270: 90}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)


def compile_harness(binary: Path) -> None:
    binary.parent.mkdir(parents=True, exist_ok=True)
    run([
        "xcrun", "swiftc", "-O", "-parse-as-library",
        "mobile_benchmark/OrientationBenchmarkV17/OrientationBenchmarkV17/V17Pipeline.swift",
        "mobile_benchmark/OrientationBenchmarkV17/OrientationBenchmarkV17/Stage2MobileV17.swift",
        "mobile_benchmark/OrientationBenchmarkV17/OrientationBenchmarkV17/Stage3MobileV17.swift",
        "scripts/validate_ios_stage2_preprocessor_v17.swift",
        "-framework", "AVFoundation", "-framework", "Vision", "-framework", "CoreML",
        "-framework", "CoreImage", "-framework", "ImageIO", "-framework", "CoreVideo",
        "-framework", "CryptoKit", "-o", str(binary),
    ])


def residual(angle: int, correction: int) -> float:
    value = float((angle + correction) % 360)
    return value - 360 if value > 180 else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--app", type=Path, default=DEFAULT_APP)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    compile_harness(args.binary)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for angle in ANGLES:
        video = args.inputs / f"hello-roll-{angle}.mp4"
        completed = run(["/usr/bin/time", "-l", str(args.binary), str(video), str(args.app)])
        result = json.loads(completed.stdout)
        rss_match = re.search(r"^\s*(\d+)\s+maximum resident set size$", completed.stderr, re.M)
        peak_rss = int(rss_match.group(1)) if rss_match else None
        correction = int(result["coarseRotationClockwise"])
        row = {
            "angleDegreesClockwise": angle,
            "video": str(video.relative_to(ROOT)),
            "videoSHA256": sha256_file(video),
            **result,
            "residualRollDegrees": residual(angle, correction),
            "peakResidentBytes": peak_rss,
        }
        rows.append(row)
        if result["predictedGlosses"] != ["HELLO"]:
            errors.append(f"{angle} degrees predicted {result['predictedGlosses']}")
        if result["stage3NaturalEnglish"] != "Hello.":
            errors.append(f"{angle} degrees produced {result['stage3NaturalEnglish']!r}")
        if angle in EXACT_CORRECTIONS and correction != EXACT_CORRECTIONS[angle]:
            errors.append(f"{angle} degrees correction {correction} is wrong")
        if abs(row["residualRollDegrees"]) > 45:
            errors.append(f"{angle} degrees residual roll exceeds 45 degrees")
        if result["windows"] < 1 or result["windows"] > 8 or result["cropCount"] < 1:
            errors.append(f"{angle} degrees produced invalid bounded inputs")
    manifest = ROOT / "active/v17/stage3_mobile_naturalizer_manifest_v17.json"
    report = {
        "format": "slt_v17_swift_video_to_english_validation",
        "version": 1,
        "createdUTC": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "executionEnvironment": "macos_host_swift_apple_vision_coreml",
        "hardwarePerformanceClaim": False,
        "thermalsInterpretable": False,
        "physicalIPhoneEvidence": False,
        "cameraToGlossEndToEnd": False,
        "swiftVideoFileToEnglishEndToEnd": True,
        "conditions": len(rows),
        "conditionsCorrect": sum(row["predictedGlosses"] == ["HELLO"] for row in rows),
        "maximumPeakResidentBytes": max(
            (row["peakResidentBytes"] or 0 for row in rows), default=0
        ),
        "naturalizerManifest": str(manifest.relative_to(ROOT)),
        "naturalizerManifestSHA256": sha256_file(manifest),
        "stage2ContractSHA256": "8be66a44d337dd99484d3ee3140f3124c2e121abe20e93ce7f09b94d96ecc30d",
        "recognizerCheckpointSHA256": "623f9b56141643704b3562a8d2fdcebe44269985b2f618eb8f0a471e857a2cf5",
        "citizenSplit": "official_validation",
        "citizenTestAccessed": False,
        "semlexTestAccessed": False,
        "localTestAccessed": False,
        "twoMFloresDevtestAccessed": False,
        "acceptanceErrors": errors,
        "rows": rows,
        "limitations": [
            "This runs the Swift sources used by the iOS app but executes Apple frameworks on Mac hardware.",
            "Peak resident size, extraction time, and Core ML execution are not physical-iPhone measurements.",
            "The input is a video file rather than a live camera capture.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.report)
    print(json.dumps({key: report[key] for key in (
        "status", "conditions", "conditionsCorrect", "maximumPeakResidentBytes",
        "acceptanceErrors",
    )}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
