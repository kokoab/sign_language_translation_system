#!/usr/bin/env python3
"""Fail-closed audit for the locked-100 mobile Stage 2 + bounded Stage 3 app."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "artifacts/reports/stage3_mobile_v17"
DEVICE_APP = (
    ROOT
    / "artifacts/generated/stage3_mobile_release_device/OrientationBenchmarkV17.app"
)
STAGE3_MANIFEST = ROOT / "active/v17/stage3_mobile_naturalizer_manifest_v17.json"
SIMULATOR_RESULT = (
    ROOT
    / "artifacts/reports/orientation_v17_simulator_benchmark/latest_result.json"
)
SWIFT_RESULT = REPORT_DIR / "swift_video_to_english_validation.json"
DATA_AUDIT = REPORT_DIR / "data_and_coverage_audit.json"
EXPECTED_MODELS = {
    "MobileCLIP2S0ImageEncoderV17FP32.mlmodelc",
    "Stage2FrozenEncoderV17FP32.mlmodelc",
    "Stage2CompactContextV17FP32.mlmodelc",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    errors: list[str] = []
    required = [DEVICE_APP, STAGE3_MANIFEST, SIMULATOR_RESULT, SWIFT_RESULT, DATA_AUDIT]
    for path in required:
        if not path.exists():
            errors.append(f"missing required path: {path.relative_to(ROOT)}")
    if errors:
        payload = {"status": "fail", "acceptanceErrors": errors}
    else:
        manifest_hash = sha256(STAGE3_MANIFEST)
        bundled_manifest = DEVICE_APP / STAGE3_MANIFEST.name
        simulator = load(SIMULATOR_RESULT)
        swift = load(SWIFT_RESULT)
        data = load(DATA_AUDIT)
        compiled_models = {path.name for path in DEVICE_APP.glob("*.mlmodelc")}
        binary_description = subprocess.check_output(
            ["/usr/bin/file", str(DEVICE_APP / "OrientationBenchmarkV17")], text=True
        ).strip()

        if compiled_models != EXPECTED_MODELS:
            errors.append(f"compiled models differ: {sorted(compiled_models)}")
        if not bundled_manifest.is_file() or sha256(bundled_manifest) != manifest_hash:
            errors.append("bundled Stage 3 manifest does not match source")
        if "arm64" not in binary_description or "Mach-O" not in binary_description:
            errors.append("generic device binary is not arm64 Mach-O")
        if not (
            simulator.get("status") == "pass"
            and simulator.get("conditionsCompleted") == 8
            and simulator.get("conditionsCorrect") == 8
            and simulator.get("iterationsPerAngle") == 200
            and simulator.get("acceptanceErrors") == []
        ):
            errors.append("final iPhone 13 simulator acceptance failed")
        if simulator.get("stage3NaturalizerManifestSHA256") != manifest_hash:
            errors.append("simulator Stage 3 manifest hash mismatch")
        if simulator.get("videoFileToGlossEndToEnd") is not False:
            errors.append("simulator incorrectly claims in-app file extraction")
        if simulator.get("cameraToGlossEndToEnd") is not False:
            errors.append("simulator incorrectly claims live-camera extraction")
        if simulator.get("hardwarePerformanceClaim") is not False:
            errors.append("simulator incorrectly claims hardware performance")
        if simulator.get("thermalsInterpretable") is not False:
            errors.append("simulator incorrectly claims thermal evidence")
        if simulator.get("citizenTestAccessed") or simulator.get("semlexTestAccessed"):
            errors.append("simulator reports test-split access")
        if not (
            swift.get("status") == "pass"
            and swift.get("conditions") == 8
            and swift.get("conditionsCorrect") == 8
            and swift.get("acceptanceErrors") == []
        ):
            errors.append("Swift video-to-English validation failed")
        if swift.get("naturalizerManifestSHA256") != manifest_hash:
            errors.append("Swift Stage 3 manifest hash mismatch")
        if swift.get("hardwarePerformanceClaim") is not False:
            errors.append("Mac-host Swift report incorrectly claims hardware performance")
        if swift.get("citizenTestAccessed") or swift.get("localTestAccessed"):
            errors.append("Swift validation reports test-split access")
        if not (
            data.get("status") == "pass"
            and data.get("genuinePairs", {}).get("total") == 1165
            and data.get("genuinePairs", {}).get("fullyLocked100") == 0
        ):
            errors.append("Stage 3 data-suitability audit failed")
        if any(
            data.get(key)
            for key in (
                "citizenTestAccessed",
                "semlexTestAccessed",
                "localTestAccessed",
                "twoMFloresDevtestAccessed",
            )
        ):
            errors.append("Stage 3 data audit reports test-split access")

        payload = {
            "format": "slt_v17_stage3_mobile_deployment_audit",
            "version": 1,
            "createdUTC": datetime.now(timezone.utc).isoformat(),
            "status": "pass" if not errors else "fail",
            "acceptanceErrors": errors,
            "hostArchitecture": platform.machine(),
            "deviceApp": str(DEVICE_APP.relative_to(ROOT)),
            "deviceAppBytes": directory_bytes(DEVICE_APP),
            "deviceBinaryDescription": binary_description,
            "compiledCoreMLModels": sorted(compiled_models),
            "stage3NaturalizerManifestSHA256": manifest_hash,
            "stage2ContractSHA256": manifest_hash and load(STAGE3_MANIFEST)[
                "stage2_contract_sha256"
            ],
            "simulatorResultSHA256": sha256(SIMULATOR_RESULT),
            "swiftVideoToEnglishResultSHA256": sha256(SWIFT_RESULT),
            "dataAuditSHA256": sha256(DATA_AUDIT),
            "simulator": {
                "device": simulator.get("simulatorDeviceName"),
                "model": simulator.get("simulatorModelIdentifier"),
                "runtime": simulator.get("simulatorRuntimeVersion"),
                "conditions": simulator.get("conditionsCompleted"),
                "correct": simulator.get("conditionsCorrect"),
                "iterationsPerAngle": simulator.get("iterationsPerAngle"),
                "hardwarePerformanceClaim": simulator.get("hardwarePerformanceClaim"),
            },
            "swiftVideoToEnglish": {
                "conditions": swift.get("conditions"),
                "correct": swift.get("conditionsCorrect"),
                "maximumPeakResidentBytes": swift.get("maximumPeakResidentBytes"),
                "hardwarePerformanceClaim": swift.get("hardwarePerformanceClaim"),
            },
            "stage3Data": {
                "genuinePairsAudited": data.get("genuinePairs", {}).get("total"),
                "genuinePairsFullyLocked100": data.get("genuinePairs", {}).get(
                    "fullyLocked100"
                ),
                "reviewedTemplates": data.get("reviewedTemplates"),
            },
            "physicalIPhoneEvidence": False,
            "liveCameraPathImplemented": False,
            "interactiveVideoFilePathImplemented": True,
            "citizenTestAccessed": False,
            "semlexTestAccessed": False,
            "localTestAccessed": False,
            "twoMFloresDevtestAccessed": False,
        }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "deployment_audit.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
