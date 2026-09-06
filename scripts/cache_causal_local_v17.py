#!/usr/bin/env python3
"""Re-extract only signer-disjoint local development videos with the live normalizer."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.continuous_vision_v17 import CausalVisionFeatures
from active.v17.extract_v17 import AppleVisionDetector, assign_hands, limit_image_side, orient_frame
from active.v17.train_continuous_evidence_v17 import sha256
from active.v17.train_streaming_tcn_ctc_v17 import refuse_protected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/local/stage2_v17_grounded_signer_split"))
    p.add_argument("--output", type=Path, default=Path("artifacts/generated/causal_local_v17_v1"))
    a = p.parse_args(); refuse_protected((a.root, a.output))
    a.output.mkdir(parents=True, exist_ok=False)
    detector = AppleVisionDetector()
    rows = []
    for role in ("train", "validation"):
        for archive in sorted((a.root / role).glob("*/*.npz")):
            with np.load(archive) as data:
                metadata = json.loads(str(data["metadata_json"]))
            if metadata["source"] != "local_phrases":
                continue
            if metadata["role"] != role:
                raise ValueError("source split mismatch")
            video = Path(metadata["video_path"]); refuse_protected((video,))
            if sha256(video) != metadata["video_sha256"]:
                raise ValueError("source video changed")
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise ValueError(f"cannot read {video}")
            capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
            fps = capture.get(cv2.CAP_PROP_FPS)
            # This focused corpus audit contains only 30-Hz recordings. Refuse
            # another timebase instead of silently changing its sequence labels.
            if abs(fps - 30) > .01:
                raise ValueError(f"non-30Hz source requires explicit tick resampling: {video}")
            normalizer = CausalVisionFeatures()
            previous = {"left": None, "right": None}; seen = {"left": -1000, "right": -1000}
            features = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                index = len(features)
                frame = limit_image_side(orient_frame(frame, metadata.get("vision_coarse_rotation_clockwise", 0),
                    metadata["video_metadata"].get("input_mirrored", False)), 640)
                detection = detector.detect(frame, True, True)
                for side in previous:
                    if index - seen[side] > 15:
                        previous[side] = None
                assigned = assign_hands(detection.hands, previous)
                for side, hand in assigned.items():
                    if hand is not None and hand.confidence[0] > 0:
                        previous[side] = hand.xy[0].copy(); seen[side] = index
                features.append(normalizer.add(detection, assigned, frame.shape[1], frame.shape[0]))
            capture.release()
            if len(features) != metadata["video_metadata"]["decoded_frame_count"]:
                raise ValueError(f"incomplete video decode: {video}")
            destination = a.output / archive.relative_to(a.root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            provenance = dict(format="slt_causal_local_features_v17", version=1, role=role,
                source_archive=str(archive), source_archive_sha256=sha256(archive),
                video=str(video), video_sha256=metadata["video_sha256"],
                identity=metadata["source_item_id"], signer=metadata["signer_id"],
                targets=metadata["target_sequence"], fps=fps, future_frames_used=False, test_accessed=False)
            np.savez_compressed(destination, features=np.asarray(features, np.float16), metadata_json=json.dumps(provenance))
            rows.append(dict(path=str(destination), frames=len(features), role=role,
                             hand_fraction=float((np.asarray(features)[:, :42, 3] > 0).any(1).mean())))
            if len(rows) % 20 == 0:
                print(json.dumps(dict(completed=len(rows), role=role)), flush=True)
    result = dict(format="slt_causal_local_cache_v17", counts=dict(Counter(r["role"] for r in rows)), rows=rows, test_accessed=False)
    (a.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"]), flush=True)


if __name__ == "__main__":
    main()
