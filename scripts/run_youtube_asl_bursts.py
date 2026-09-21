#!/usr/bin/env python3
"""Restart the resumable YouTube-ASL downloader after sustained stalls."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def count_json(path: Path) -> int:
    return sum(1 for _ in path.glob("*.json"))


def notify(message: str) -> None:
    script = f'display notification {json.dumps(message)} with title "SLT download"'
    subprocess.run(["osascript", "-e", script], check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bitstreams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--bursts", type=int, default=3)
    parser.add_argument("--stale-seconds", type=int, default=120)
    args = parser.parse_args()

    command = [sys.executable, str(Path(__file__).with_name("download_youtube_asl_pilot.py")),
               "--manifest", str(args.manifest), "--bitstreams", str(args.bitstreams),
               "--output", str(args.output), "--status", str(args.status)]
    start = count_json(args.output)
    for burst in range(1, args.bursts + 1):
        process = subprocess.Popen(command)
        last_count = count_json(args.output)
        last_progress = time.monotonic()
        while process.poll() is None:
            time.sleep(5)
            current = count_json(args.output)
            if current >= 2000:
                process.wait()
                notify("YouTube-ASL pilot complete: 2000/2000 clips")
                return
            if current > last_count:
                last_count = current
                last_progress = time.monotonic()
            elif time.monotonic() - last_progress >= args.stale_seconds:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
        if process.returncode == 0:
            return
        args.status.write_text(json.dumps({"state": "restarting", "completed": last_count,
                                           "total": 2000, "detail": f"burst {burst}/3 stalled"},
                                          indent=2) + "\n")
    completed = count_json(args.output)
    notify(f"YouTube-ASL three bursts ended: {completed}/2000 clips")


if __name__ == "__main__":
    main()
