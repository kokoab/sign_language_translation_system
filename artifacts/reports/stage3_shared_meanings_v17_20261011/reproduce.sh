#!/bin/sh
set -eu
cd /Volumes/secret/SLT/SLT
report=artifacts/reports/stage3_shared_meanings_v17_20261011
source=/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios/Runner/LiveReel
packages=artifacts/coreml/stage3_multisentence_tiny_v17_20260929
venv/bin/python "$report/build_experiment.py"
venv/bin/python "$report/extend_experiment.py"
swiftc -O -swift-version 5 -target arm64-apple-macos15.0 "$source/LiveReelCore.swift" "$source/LiveReelModels.swift" "$source/LiveReelDecoder.swift" "$source/LiveReelStage3.swift" "$source/LiveReelEngine.swift" "$report/main.swift" -o "$report/probe" > "$report/build.log" 2>&1
for prefix in '' extended_ phone_ locking_; do
    "$report/probe" "$packages" "$report/${prefix}cases.json" "$report/${prefix}results.jsonl" > "$report/${prefix}run.log" 2>&1
done
venv/bin/python "$report/analyze.py" > "$report/analysis.log"
venv/bin/python "$report/check_contracts.py"
venv/bin/python "$report/verify.py"
