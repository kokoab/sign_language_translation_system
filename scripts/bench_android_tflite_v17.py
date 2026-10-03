#!/usr/bin/env python3
"""Benchmark the Android-family TFLite models on a USB-connected phone with LiteRT benchmark_model.

Configurations: 1 big core, 4 big cores (cores pinned with taskset), 4 little cores, GPU delegate.
Raw logs and a summary JSON go to --output. Only /data/local/tmp/slt_bench on the phone is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ADB = ROOT / "artifacts/generated/android_tools/platform-tools/adb"
REMOTE = "/data/local/tmp/slt_bench"
MODELS = {
    "hand_detector": "hand_landmarkerhand_detector.tflite",
    "hand_landmarks": "hand_landmarkerhand_landmarks_detector.tflite",
    "pose_detector": "pose_landmarker_litepose_detector.tflite",
    "pose_landmarks": "pose_landmarker_litepose_landmarks_detector.tflite",
    "face_detector": "face_landmarkerface_detector.tflite",
    "face_landmarks": "face_landmarkerface_landmarks_detector.tflite",
    "boundary_fp32": "boundary_mediapipe_fp32.tflite",
    "span_b8_fp32": "span_recognizer_mediapipe_b8_fp32.tflite",
    "span_b8_int8": "span_recognizer_mediapipe_b8_int8.tflite",
    "span_landmark_b8_fp32": "span_recognizer_landmark_only_b8_fp32.tflite",
    "mobileclip_fp32": "mobileclip2_s0_image_fp32.tflite",
    "t5_encoder_fp32": "stage3_t5_encoder_fp32.tflite",
    "t5_decoder_fp32": "stage3_t5_decoder_step_fp32.tflite",
    "t5_encoder_int8": "stage3_t5_encoder_int8.tflite",
    "t5_decoder_int8": "stage3_t5_decoder_step_int8.tflite",
}
# Token-id inputs must stay inside the vocabulary; positions inside the 64-token window.
INT_RANGES = {
    "t5_encoder": ("--input_layer=serving_default_args_0,serving_default_args_1 --input_layer_shape=1,64:1,64 "
                   "--input_layer_value_range=serving_default_args_0,0,200:serving_default_args_1,1,1"),
    "t5_decoder": ("--input_layer=serving_default_args_0,serving_default_args_1,serving_default_args_2,"
                   "serving_default_args_3 --input_layer_shape=1,64:1,64,256:1,64:1 "
                   "--input_layer_value_range=serving_default_args_0,0,200:serving_default_args_2,1,1:"
                   "serving_default_args_3,0,10"),
}
CONFIGS = {
    "cpu_1big": ("taskset 10", "--num_threads=1"),
    "cpu_4big": ("taskset f0", "--num_threads=4"),
    "cpu_4little": ("taskset 0f", "--num_threads=4"),
    "gpu": ("taskset f0", "--use_gpu=true --gpu_precision_loss_allowed=true"),
}


def adb(command: str, timeout: int = 900) -> str:
    result = subprocess.run([str(ADB), "shell", command], capture_output=True, text=True, timeout=timeout)
    return result.stdout + result.stderr


def temperature() -> float | None:
    raw = adb("cat /sys/class/power_supply/battery/temp 2>/dev/null").strip()
    return int(raw) / 10 if raw.isdigit() else None


def parse(log: str) -> dict:
    out = {}
    match = re.search(r"Inference timings in us: Init: (\d+), First inference: (\d+), Warmup \(avg\): ([\d.e+]+), "
                      r"Inference \(avg\): ([\d.e+]+)", log)
    if match:
        out.update(init_ms=int(match[1]) / 1000, first_ms=int(match[2]) / 1000,
                   warmup_ms=float(match[3]) / 1000, avg_ms=round(float(match[4]) / 1000, 2))
    std = re.search(r"count=\d+ first=\d+ curr=\d+ min=(\d+) max=(\d+) avg=[\d.e+]+ std=(\d+)", log)
    if std:
        out.update(min_ms=int(std[1]) / 1000, max_ms=int(std[2]) / 1000, std_ms=int(std[3]) / 1000)
    memory = re.search(r"Overall peak memory footprint \(MB\) via periodic monitoring: ([\d.]+)", log)
    if memory:
        out.update(peak_mem_mb=float(memory[1]))
    if "GPU delegate created" in log or "Created TensorFlow Lite delegate for GPU" in log:
        out["gpu_delegate"] = True
    replaced = re.search(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \(TfLiteGpuDelegate", log)
    if replaced:
        out["gpu_nodes"] = f"{replaced[1]}/{replaced[2]}"
    if not match:
        out["error"] = (log.strip().splitlines() or ["no output"])[-1][:200]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "artifacts/reports/android_device_bench_20261004")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()
    (args.output / "logs").mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in args.models:
        for config in args.configs:
            pin, flags = CONFIGS[config]
            ranges = next((v for k, v in INT_RANGES.items() if name.startswith(k)), None)
            extra = f" {ranges}" if ranges else ""
            command = (f"cd {REMOTE} && {pin} ./benchmark_model --graph={MODELS[name]} {flags} "
                       f"--num_runs={args.runs} --warmup_runs=3 --report_peak_memory_footprint=true{extra}")
            before = temperature()
            log = adb(command)
            (args.output / "logs" / f"{name}__{config}.log").write_text(log)
            row = parse(log)
            row.update(battery_c_before=before, battery_c_after=temperature())
            summary.setdefault(name, {})[config] = row
            summary_path.write_text(json.dumps(summary, indent=1) + "\n")
            print(name, config, {k: row.get(k) for k in ("avg_ms", "init_ms", "peak_mem_mb", "gpu_nodes", "error")},
                  flush=True)


if __name__ == "__main__":
    main()
