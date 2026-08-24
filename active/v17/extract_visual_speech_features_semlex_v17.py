#!/usr/bin/env python3
"""Cache frozen Auto-AVSR features for non-training RGB diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.extract_hand_rgb_semlex_val_v17 import (
    SOURCE as SEMLEX_SOURCE, SPLIT as SEMLEX_SPLIT, validation_items,
)
from active.v17.local_multimodal_audit_v17 import (
    SOURCE as LOCAL_SOURCE, SPLIT as LOCAL_SPLIT, local_audit_items,
)
from active.v17.model_visual_speech_v17 import AutoAVSRVisualFrontend, load_auto_avsr_frontend
from active.v17.schema_visual_speech_v17 import VIEW_NAMES, VisualSpeechV17Config, schema_fingerprint
from active.v17.train_stage_1_v17 import select_device
from active.v17.train_stage_1_visual_speech_v17 import (
    AUTO_AVSR_MODEL_NAME, AUTO_AVSR_REPORTED_LRS3_WER, AUTO_AVSR_SOURCE,
    AUTO_AVSR_TRAINING_HOURS, decode_view, prepare_pixels, sha256_file,
)


class SemLexVisualSpeechDataset(Dataset):
    def __init__(
        self, root: Path, selection_manifest: Path, citizen_manifest: Path, view: str,
        source: str = SEMLEX_SOURCE,
    ):
        if view not in VIEW_NAMES:
            raise ValueError(f"unknown visual-speech view: {view}")
        self.root = Path(root)
        self.view = view
        self.view_index = VIEW_NAMES.index(view)
        self.expected_schema = schema_fingerprint(VisualSpeechV17Config())
        citizen = json.loads(Path(citizen_manifest).read_text(encoding="utf-8"))
        self.label_to_index = {
            str(row["canonical_label"]): int(row["class_index"])
            for row in citizen["classes"]
        }
        if source == SEMLEX_SOURCE:
            items, _ = validation_items(Path(selection_manifest))
            self.expected_source, self.expected_split = SEMLEX_SOURCE, SEMLEX_SPLIT
        elif source == LOCAL_SOURCE:
            items, _ = local_audit_items(Path(selection_manifest))
            self.expected_source, self.expected_split = LOCAL_SOURCE, LOCAL_SPLIT
        else:
            raise ValueError(f"unsupported visual-speech diagnostic source: {source}")
        self.files = [
            self.root / item.label / f"{item.item_id}.visual_speech_v17.npz"
            for item in items
        ]
        self.targets = torch.tensor(
            [self.label_to_index[item.label] for item in items], dtype=torch.long
        )
        missing = [path for path in self.files if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
            if (
                metadata.get("schema_fingerprint") != self.expected_schema
                or metadata.get("source") != self.expected_source
                or metadata.get("split") != self.expected_split
                or metadata.get("training_eligible") is not False
                or metadata.get("audio_accessed") is not False
                or metadata.get("test_accessed") is not False
            ):
                raise ValueError(f"SemLex visual-speech provenance mismatch: {path}")
            pixels = decode_view(payload["jpeg_blob"], payload["jpeg_offsets"], self.view_index)
            valid = payload["valid"][:, self.view_index].astype(np.bool_, copy=True)
        return (
            torch.from_numpy(pixels).permute(0, 3, 1, 2).contiguous(),
            torch.from_numpy(valid), self.targets[index],
        )


def run(args: argparse.Namespace) -> dict[str, object]:
    dataset = SemLexVisualSpeechDataset(
        args.data_root, args.selection_manifest, args.citizen_manifest, args.view,
        args.source,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = select_device(args.device)
    frontend = AutoAVSRVisualFrontend()
    load_result = load_auto_avsr_frontend(frontend, str(args.pretrained_checkpoint))
    frontend = frontend.to(device).eval()
    output: list[np.ndarray] = []
    validity: list[np.ndarray] = []
    with torch.inference_mode():
        for pixels, valid, _ in loader:
            pixels, valid_device = prepare_pixels(pixels.to(device), valid.to(device), False)
            output.append(frontend(pixels).cpu().numpy().astype(np.float16))
            validity.append(valid_device.cpu().numpy().astype(np.bool_))
    features = np.concatenate(output)
    valid = np.concatenate(validity)
    metadata = {
        "format": "slt_auto_avsr_visual_features_v17",
        "split": dataset.expected_split,
        "source": dataset.expected_source,
        "view": args.view,
        "samples": len(dataset),
        "shape": list(features.shape),
        "crop_schema_fingerprint": dataset.expected_schema,
        "manifest_sha256": sha256_file(args.citizen_manifest),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "pretrained_checkpoint": str(args.pretrained_checkpoint),
        "pretrained_checkpoint_sha256": sha256_file(args.pretrained_checkpoint),
        "pretraining_source": AUTO_AVSR_SOURCE,
        "pretraining_model": AUTO_AVSR_MODEL_NAME,
        "reported_training_hours": AUTO_AVSR_TRAINING_HOURS,
        "reported_lrs3_visual_wer": AUTO_AVSR_REPORTED_LRS3_WER,
        "frontend_load_result": load_result,
        "pixels_augmented": False,
        "crop_mode": "center_88_from_aligned_112",
        "visual_only": True,
        "audio_accessed": False,
        "training_eligible": False,
        "test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary, features=features, valid=valid,
        targets=dataset.targets.numpy(),
        item_ids=np.asarray([
            f"{dataset.expected_source}/{path.parent.name}/{path.name.removesuffix('.visual_speech_v17.npz')}"
            for path in dataset.files
        ]),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(args.output)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=(SEMLEX_SOURCE, LOCAL_SOURCE), default=SEMLEX_SOURCE)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("data/local/semlex_citizen100_val_audit/visual_speech_rgb"),
    )
    parser.add_argument(
        "--selection-manifest", type=Path,
        default=Path("data/local/semlex_citizen100_val_audit/selection_plan.json"),
    )
    parser.add_argument(
        "--citizen-manifest", type=Path,
        default=Path("active/v17/citizen100_manifest.json"),
    )
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--view", choices=("mouth", "lower_face"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
