#!/usr/bin/env python3
"""Prepare a blind, train-only Luna boundary-annotation pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANIFEST = ROOT / "artifacts/reports/confident_supervision_v17_20260920/confident_supervision.json"
COUNT = 24
FRAMES = 24
MARGIN = 0.35


def rank(row):
    return hashlib.sha256(str(row["identity"]).encode()).hexdigest()


def select(rows):
    candidates = [row for row in rows if row["role"] == "train"
                  and row["target_kind"] == "known"
                  and str(row["source"]).startswith("asllrp")
                  and row["source_crop_complete"] is True]
    chosen, labels, items = [], set(), set()
    for source, quota in (("asllrp_contiguous", 8), ("asllrp_other_ctc", 16)):
        pool = sorted((row for row in candidates if row["source"] == source), key=rank)
        for row in pool:
            if row["label"] in labels or row["source_item_id"] in items:
                continue
            chosen.append(row); labels.add(row["label"]); items.add(row["source_item_id"])
            if sum(value["source"] == source for value in chosen) == quota:
                break
    if len(chosen) != COUNT:
        raise RuntimeError(f"selected {len(chosen)} of {COUNT}")
    return chosen


def contact_sheet(row, index):
    video = ROOT / row["video_path"]
    if "test" in {part.casefold() for part in video.parts} or not video.is_file():
        raise ValueError(f"invalid pilot video: {video}")
    start = max(0.0, float(row["start_seconds"]) - MARGIN)
    stop = float(row["end_seconds"]) + MARGIN
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count < 2:
        raise RuntimeError(f"invalid video clock: {video}")
    stop = min(stop, (frame_count - 1) / fps)
    if stop <= start:
        raise RuntimeError(f"empty review interval: {video}")
    times = np.linspace(start, stop, FRAMES)
    cells = []
    for frame_index, second in enumerate(times):
        capture.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"cannot read {video} at {second:.3f}s")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((230, 150))
        cell = Image.new("RGB", (240, 180), "white")
        cell.paste(image, ((240 - image.width) // 2, 0))
        ImageDraw.Draw(cell).text((5, 156), f"frame {frame_index:02d}  {second:.3f}s", fill="black")
        cells.append(cell)
    capture.release()
    sheet = Image.new("RGB", (1440, 760), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 4), f"item {index:02d}   target gloss: {row['label']}   choose first/last target-sign frame", fill="black")
    for frame_index, cell in enumerate(cells):
        sheet.paste(cell, ((frame_index % 6) * 240, 40 + (frame_index // 6) * 180))
    path = HERE / "sheets" / f"item_{index:02d}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=91)
    return times, path


def main():
    payload = json.loads(MANIFEST.read_text())
    chosen = select(payload["rows"])
    blind, reference = [], []
    for index, row in enumerate(chosen, 1):
        times, sheet = contact_sheet(row, index)
        common = dict(item=f"item_{index:02d}", gloss=row["label"], source=row["source"],
                      sheet=str(sheet.relative_to(HERE)), frame_times_seconds=times.tolist())
        blind.append(common)
        reference.append(dict(**common, identity=row["identity"], signer_id=row["signer_id"],
                              source_item_id=row["source_item_id"], video_path=row["video_path"],
                              source_start_seconds=row["start_seconds"], source_end_seconds=row["end_seconds"]))
    (HERE / "blind_manifest.json").write_text(json.dumps(blind, indent=2) + "\n")
    (HERE / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    (HERE / "INSTRUCTIONS.md").write_text(
        "# Blind boundary annotation\n\n"
        "Inspect only `blind_manifest.json` and the corresponding sheet. For each item, choose the first frame clearly belonging to the named target sign and the last frame still belonging to it. Exclude setup, release, and movement into another sign. If the target cannot be isolated confidently, reject it. Do not inspect `reference.json`; it contains the source annotations reserved for agreement analysis.\n\n"
        "Write a JSON array with: `item`, `start_frame`, `end_frame`, `confidence` (`high`, `medium`, or `low`), and `reason`. Rejected items use null frames and confidence `reject`.\n")
    assert len(blind) == COUNT and len({row["gloss"] for row in blind}) == COUNT
    assert all(len(row["frame_times_seconds"]) == FRAMES for row in blind)
    print(f"prepared {COUNT} blind items with {FRAMES} frames each")


if __name__ == "__main__":
    main()
