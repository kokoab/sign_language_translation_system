#!/usr/bin/env python3
"""Assign anonymous, reproducible signer groups to the local phrase videos.

Only face embeddings are used for clustering. The embeddings stay in memory and are
not written to disk; the output contains anonymous cluster IDs and audit scores.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".m4v"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_frames(path: Path, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = np.linspace(0.20, 0.80, count)
    frames = []
    for fraction in positions:
        if total > 1:
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(fraction * (total - 1)))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    capture.release()
    return frames


def face_feature(
    frame: np.ndarray,
    detector: cv2.FaceDetectorYN,
    recognizer: cv2.FaceRecognizerSF,
) -> tuple[np.ndarray | None, float]:
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    if faces is None or not len(faces):
        return None, 0.0
    # Phrase clips can contain a bystander near an edge. Prefer the central face,
    # using area only as a tie-breaker, because the signer is camera-centered.
    def signer_score(row: np.ndarray) -> float:
        center_x = float(row[0] + row[2] / 2) / width
        center_y = float(row[1] + row[3] / 2) / height
        center_distance = (center_x - 0.5) ** 2 + (center_y - 0.42) ** 2
        area = float(row[2] * row[3]) / (width * height)
        return area / (0.03 + center_distance)

    face = max(faces, key=signer_score)
    aligned = recognizer.alignCrop(frame, face)
    feature = recognizer.feature(aligned).reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    if not np.isfinite(feature).all() or norm <= 0:
        return None, 0.0
    return feature / norm, float(face[-1])


def video_feature(
    path: Path,
    detector: cv2.FaceDetectorYN,
    recognizer: cv2.FaceRecognizerSF,
    samples: int,
) -> tuple[np.ndarray | None, int, float]:
    features, scores = [], []
    for frame in sample_frames(path, samples):
        feature, score = face_feature(frame, detector, recognizer)
        if feature is not None:
            features.append(feature)
            scores.append(score)
    if not features:
        return None, 0, 0.0
    value = np.mean(features, axis=0)
    value /= np.linalg.norm(value)
    return value.astype(np.float32), len(features), float(np.mean(scores))


def numbered_take(path: Path) -> bool:
    return bool(re.fullmatch(rf"{re.escape(path.parent.name)}_\d+", path.stem))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, default=Path("data/raw_videos/PHRASES"))
    parser.add_argument(
        "--clusters", default="auto",
        help="signer count, or 'auto' to select it by cosine silhouette (default)",
    )
    parser.add_argument("--max-clusters", type=int, default=10)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--detector", type=Path, default=Path(
        "artifacts/models/opencv_face_signer_clustering_v1/"
        "face_detection_yunet_2023mar.onnx"
    ))
    parser.add_argument("--recognizer", type=Path, default=Path(
        "artifacts/models/opencv_face_signer_clustering_v1/"
        "face_recognition_sface_2021dec.onnx"
    ))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/reports/local_phrase_signer_audit_v17"
    ))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.clusters != "auto":
        try:
            requested_clusters = int(args.clusters)
        except ValueError as error:
            raise ValueError("clusters must be an integer >=2 or 'auto'") from error
        if requested_clusters < 2:
            raise ValueError("clusters must be >=2")
    else:
        requested_clusters = None
    if args.samples < 1 or args.max_clusters < 2:
        raise ValueError("samples must be positive and max-clusters must be >=2")

    paths = sorted(
        path for path in args.video_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES
    )
    if requested_clusters is not None and len(paths) < requested_clusters:
        raise ValueError("fewer videos than requested signer clusters")
    detector = cv2.FaceDetectorYN_create(
        str(args.detector), "", (320, 320), 0.70, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF_create(str(args.recognizer), "")

    rows, embedded = [], []
    for index, path in enumerate(paths, 1):
        feature, detected_frames, detection_score = video_feature(
            path, detector, recognizer, args.samples
        )
        row = {
            "video_path": path.as_posix(),
            "phrase": path.parent.name,
            "numbered_take": numbered_take(path),
            "detected_frames": detected_frames,
            "mean_detection_score": detection_score,
            "feature": feature,
        }
        rows.append(row)
        if feature is not None:
            embedded.append(row)
        if index % 100 == 0:
            print(f"embedded {index}/{len(paths)}", flush=True)
    if requested_clusters is not None and len(embedded) < requested_clusters:
        raise RuntimeError("too few detected faces to cluster")

    matrix = np.stack([row["feature"] for row in embedded])
    candidates = []
    cluster_counts = (
        [requested_clusters] if requested_clusters is not None
        else range(2, min(args.max_clusters, len(embedded) - 1) + 1)
    )
    labels_by_count = {}
    for count in cluster_counts:
        labels = AgglomerativeClustering(
            n_clusters=count, metric="cosine", linkage="average"
        ).fit_predict(matrix)
        sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        score = float(silhouette_score(matrix, labels, metric="cosine"))
        candidates.append({"clusters": count, "silhouette_cosine": score, "sizes": sizes})
        labels_by_count[count] = labels
    selected_clusters = max(candidates, key=lambda row: row["silhouette_cosine"])["clusters"]
    raw_labels = labels_by_count[selected_clusters]
    centroids = []
    for label in range(selected_clusters):
        centroid = matrix[raw_labels == label].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        centroids.append(centroid)
    centroids = np.stack(centroids)

    # Make anonymous IDs stable across runs by sorting on the first member path.
    raw_to_stable = {
        raw: stable for stable, raw in enumerate(sorted(
            range(selected_clusters),
            key=lambda label: min(
                row["video_path"] for row, value in zip(embedded, raw_labels)
                if value == label
            ),
        ), 1)
    }
    for row, raw_label, similarities in zip(embedded, raw_labels, matrix @ centroids.T):
        own = float(similarities[raw_label])
        second = float(np.max(np.delete(similarities, raw_label)))
        row.update({
            "signer_id": f"local_signer_{raw_to_stable[int(raw_label)]:02d}",
            "own_centroid_similarity": own,
            "next_centroid_similarity": second,
            "identity_margin": own - second,
            # Low-confidence clips remain useful for training but must not define a
            # held-out signer split without manual review.
            "identity_confident": own >= 0.363 and own - second >= 0.05,
        })
    for row in rows:
        if row["feature"] is None:
            row.update({
                "signer_id": None,
                "own_centroid_similarity": None,
                "next_centroid_similarity": None,
                "identity_margin": None,
                "identity_confident": False,
            })

    public_rows = [{key: value for key, value in row.items() if key != "feature"} for row in rows]
    counts: dict[str, dict[str, object]] = {}
    for row in public_rows:
        signer = row["signer_id"] or "unresolved"
        bucket = counts.setdefault(signer, {"videos": 0, "phrases": set(), "confident": 0})
        bucket["videos"] += 1
        bucket["phrases"].add(row["phrase"])
        bucket["confident"] += int(row["identity_confident"])
    summary = {
        signer: {
            "videos": value["videos"],
            "phrases": len(value["phrases"]),
            "phrase_names": sorted(value["phrases"]),
            "confident_videos": value["confident"],
        }
        for signer, value in sorted(counts.items())
    }
    numbered_groups = sorted({
        row["signer_id"] for row in public_rows
        if row["numbered_take"] and row["signer_id"] is not None
    })
    payload = {
        "format": "slt_local_phrase_anonymous_signer_clusters_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "OpenCV YuNet face detection + SFace embeddings + average-linkage cosine clustering",
        "requested_signer_clusters": args.clusters,
        "selected_signer_clusters": selected_clusters,
        "cluster_count_candidates": candidates,
        "video_count": len(rows),
        "embedded_video_count": len(embedded),
        "unresolved_video_count": len(rows) - len(embedded),
        "silhouette_cosine": next(
            row["silhouette_cosine"] for row in candidates
            if row["clusters"] == selected_clusters
        ),
        "numbered_take_signer_groups": numbered_groups,
        "numbered_take_single_group": len(numbered_groups) == 1,
        "clusters": summary,
        "rows": public_rows,
        "privacy": "anonymous group IDs only; face embeddings are not persisted",
        "detector_sha256": sha256(args.detector),
        "recognizer_sha256": sha256(args.recognizer),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "signer_clusters.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "signer_clusters.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
