"""Recover audited 1–3-frame tails into versioned landmark-only phrase caches."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.extract_stage2_multimodal_v17 import save_archive, sha256
from active.v17.extract_v17 import AppleVisionDetector, read_video_frames, extract_frames_v17, rotate_frame_clockwise
from active.v17.schema_stage2_features_v17 import landmark_config
from active.v17.schema_v17 import schema_payload, schema_fingerprint, MOUTH_START, MOUTH_END
from active.v17.train_unified_streaming_ctc_v17 import validate_phrase_archive


def recovery_ranges(old, frames):
    if (old.ndim != 2 or old.shape[1] != 2 or not len(old) or old[0, 0] != 0
            or not np.issubdtype(old.dtype, np.integer)
            or np.any(old[:, 1] - old[:, 0] != 32)
            or np.any(old[1:, 0] != old[:-1, 1]) or not 1 <= frames-int(old[-1, 1]) <= 3):
        raise ValueError('expected contiguous 32-frame windows missing only 1–3 tail frames')
    start = int(old[-1, 0])
    middle = (start + frames) // 2
    return np.concatenate((old[:-1], np.array([[start, middle], [middle, frames]])))


def run():
    start_time = time.monotonic()
    source_report = ROOT / 'artifacts/reports/phrase_contract_repair_v17_20260921/validation.json'
    blocked = {r['path'] for r in json.loads(source_report.read_text())['rejected_archives']}
    destination = ROOT / 'data/local/phrase_tail_recovery_v17_20260921'
    report = ROOT / 'artifacts/reports/phrase_tail_recovery_v17_20260921'
    if destination.exists():
        raise FileExistsError(f'refusing to overwrite {destination}')
    report.mkdir(parents=True, exist_ok=True)
    classes = json.loads((ROOT / 'active/v17/citizen100_manifest.json').read_text())['classes']
    labels = {r['canonical_label']: r['class_index'] for r in classes}
    config = landmark_config()
    detector = AppleVisionDetector()
    roots = ('stage2_v17_grounded_phrases_fixed_20260921', 'stage2_v17_asllrp_other_multimodal',
             'stage2_v17_flores_other_20260921')
    inventory, recovered = [], 0
    for name in roots:
        for role in ('train', 'validation'):
            for source in sorted((ROOT / 'data/local' / name / role).glob('*/*.npz')):
                relative = str(source.relative_to(ROOT))
                output = destination / name / source.relative_to(ROOT / 'data/local' / name)
                output.parent.mkdir(parents=True, exist_ok=True)
                old_hash = sha256(source)
                row = dict(source=relative, source_sha256=old_hash, output=str(output.relative_to(ROOT)))
                if relative not in blocked:
                    output.symlink_to(source)
                    row['action'] = 'unchanged_symlink'
                else:
                    with np.load(source, allow_pickle=False) as d:
                        metadata = json.loads(str(d['metadata_json'].item()))
                        old = d['window_source_ranges']
                        targets = d['target_indices']
                        cached = d['landmarks']
                    video = ROOT / metadata.get('video_path', metadata['video_metadata']['video_path'])
                    if sha256(video) != metadata['video_sha256']:
                        raise ValueError(f'video hash changed: {video}')
                    count = metadata['sampled_source_frames']
                    ranges = recovery_ranges(old, count)
                    frames, video_metadata = read_video_frames(video, count, 1280, rotation='auto', input_mirrored=False)
                    if len(frames) != count or video_metadata['decoded_frame_count'] != count:
                        raise ValueError(f'video no longer matches source-frame timeline: {video}')
                    rotation = metadata['vision_coarse_rotation_clockwise']
                    if rotation:
                        frames = [rotate_frame_clockwise(f, rotation) for f in frames]
                    features, diagnostics = [], []
                    for left, right in ranges[-2:]:
                        result = extract_frames_v17(frames[left:right], config, detector=detector)
                        value = np.zeros((32, 61, 5), dtype=np.float16) if result is None else result.features.copy()
                        if metadata.get('zero_lip_nodes', False):
                            value[:, MOUTH_START:MOUTH_END] = 0
                        features.append(value)
                        diagnostics.append({'no_usable_hand_detections': True} if result is None else result.diagnostics)
                    landmarks = np.concatenate((cached[:-1], np.asarray(features, dtype=np.float16)))
                    # The output is explicitly landmark-only; old RGB summaries/schema are in the source archive.
                    for key in ('hand_valid_fraction', 'landmark_valid_windows', 'valid_windows', 'window_stride'):
                        metadata.pop(key, None)
                    metadata.update(format='recovered_landmark_only_phrase_v17', schema=schema_payload(config),
                                    schema_fingerprint=schema_fingerprint(config), window_count=len(ranges),
                                    dropped_tail_frames=0, video_path=str(video.relative_to(ROOT)),
                                    window_diagnostics=metadata['window_diagnostics'][:-1] + diagnostics,
                                    recovery=dict(source_archive=relative, source_archive_sha256=old_hash,
                                                  old_ranges=old.tolist(), policy='rebalance last full window and tail; preserve preceding windows',
                                                  reextracted_source_frames=int(count-old[-1, 0]),
                                                  invalid_new_windows=sum('no_usable_hand_detections' in d for d in diagnostics)))
                    validate_phrase_archive(landmarks, ranges, targets, metadata, labels)
                    save_archive(output, dict(landmarks=landmarks, window_source_ranges=ranges, target_indices=targets), metadata)
                    recovered += 1
                    row.update(action='recovered', missing_frames=int(count-old[-1, 1]),
                               output_sha256=sha256(output), invalid_new_windows=metadata['recovery']['invalid_new_windows'])
                    print(json.dumps(dict(recovered=recovered, elapsed_seconds=round(time.monotonic()-start_time, 1))), flush=True)
                inventory.append(row)
                (report / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    summary = dict(archives=len(inventory), recovered=recovered, unchanged=len(inventory)-recovered,
                   elapsed_seconds=round(time.monotonic()-start_time, 1),
                   invalid_new_windows=sum(r.get('invalid_new_windows', 0) for r in inventory),
                   source_report_sha256=sha256(source_report), training_started=False, acquisition_started=False)
    (report / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    run()
