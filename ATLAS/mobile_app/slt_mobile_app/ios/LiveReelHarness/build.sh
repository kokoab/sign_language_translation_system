#!/bin/sh
# Build the macOS replay harness from the same LiveReel sources the iPhone app compiles.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
src="$here/../Runner/LiveReel"
swiftc -O -swift-version 5 -target arm64-apple-macos15.0 \
  "$src/LiveReelCore.swift" "$src/LiveReelModels.swift" "$src/LiveReelDecoder.swift" \
  "$src/LiveReelStage3.swift" "$src/LiveReelEngine.swift" "$here/main.swift" \
  -o "$here/${1:-harness}"
echo "built $here/${1:-harness}"
