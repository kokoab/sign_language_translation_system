"""Reconcile current admitted isolated IDs with the existing MediaPipe family.

This prepares new files only. It neither trains nor changes any admission gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OLD = ROOT / 'artifacts/generated/mp_unfrozen_phrase_adapt_v17_20261004'
CURRENT = ROOT / 'artifacts/reports/citizen_variant_local_filter_v17_20261008/run_v2/raw_cache_citizen_variant'
REPORT = ROOT / 'artifacts/reports/android_alignment_v17_20261013'
EXTRA = ROOT / 'data/local/mediapipe_alignment_v17_20261013'
CACHE = ROOT / 'artifacts/generated/android_alignment_raw_v17_20261013'
DOMAINS = ('citizen', 'semlex', 'local')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def desired(name):
    audit = json.loads((ROOT / 'artifacts/reports/temporal_repair_training_v17_20261011/isolated_role_check.json').read_text())
    with np.load(CURRENT / (name + '.npz'), allow_pickle=False) as data:
        ids, targets = data['item_ids'], data['targets']
    excluded = audit['semlex']['excluded_validation_ids'] if name == 'semlex_val' else []
    keep = ~np.isin(ids, excluded)
    ids, targets = ids[keep], targets[keep]
    if len(set(ids.tolist())) != len(ids):
        raise ValueError('Duplicate admitted ID: ' + name)
    return ids, targets


def prepare():
    from active.v17.extract_hand_rgb_supplement_v17 import selection_items
    from scripts.extract_mediapipe_full_v17 import mirrored, guard_root
    from active.v17.mediapipe_full_v17 import MediaPipeFullV17Config

    REPORT.mkdir(parents=True, exist_ok=True)
    guard_root(EXTRA, MediaPipeFullV17Config())
    summary, jobs = {}, []
    for domain in DOMAINS:
        for split in ('train', 'val'):
            name = domain + '_' + split
            ids, targets = desired(name)
            with np.load(OLD / (name + '.npz'), allow_pickle=False) as data:
                old_ids, old_targets = data['item_ids'], data['targets']
            old = dict(zip(old_ids.tolist(), old_targets.tolist()))
            overlap = set(ids.tolist()) & set(old)
            assert all(old[item] == int(target) for item, target in zip(ids, targets) if item in old)
            missing = sorted(set(ids.tolist()) - set(old))
            summary[name] = {'admitted': len(ids), 'cached': len(overlap),
                             'missing': missing, 'excluded_old_count': len(set(old) - set(ids.tolist()))}
            if domain == 'local':
                kind = 'local_deep_clean' if split == 'train' else 'local_deep_clean_val'
                path = ROOT / 'data/local/web_variant_supplement_v17_20261008' / ('local_' + split + '_citizen_variant_v2.json')
                rows, _ = selection_items(path, kind)
                index = {row.label + '/' + row.item_id: row for row in rows}
                for item in missing:
                    row = index[item]
                    raw, landmark = ROOT / row.raw_path, ROOT / row.landmark_path
                    assert raw.is_file() and landmark.is_file(), item
                    actual = sha(raw)
                    expected = row.source_row.get('raw_sha256') or row.source_row.get('video_sha256')
                    if expected is not None and actual != expected:
                        raise ValueError('Changed admitted source video: ' + item)
                    jobs.append(dict(source=name, label=row.label, item_id=row.item_id,
                                     admitted_id=item, raw=str(raw), raw_sha256=actual,
                                     apple_landmark=str(landmark), orientation_metadata_sha256=sha(landmark),
                                     landmark=str(mirrored(EXTRA, 'landmarks', row.landmark_path, '.v17.npz')),
                                     crops=str(mirrored(EXTRA, 'hand_rgb', row.landmark_path, '.hand_rgb_v17.npz')),
                                     embedding=str(mirrored(EXTRA, 'hand_mobileclip2_s0', row.landmark_path, '.hand_mobileclip2_v17.npz'))))
    for domain in DOMAINS:
        train, _ = desired(domain + '_train'); val, _ = desired(domain + '_val')
        assert not set(train.tolist()) & set(val.tolist()), domain
    (REPORT / 'isolated_reconciliation.json').write_text(json.dumps(summary, indent=2) + '\n')
    (REPORT / 'extra_jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    print(json.dumps({'domains': {k: {x: v[x] for x in ('admitted', 'cached', 'excluded_old_count')} for k, v in summary.items()}, 'extra_jobs': len(jobs)}))


def cache():
    from active.v17.train_stage_1_v17 import load_v17_archive, mask_mouth_nodes_v17
    from active.v17.train_unified_multimodal_student_v17 import load_hand
    from active.v17.mediapipe_full_v17 import MediaPipeFullV17Config, schema_fingerprint, derived_fingerprint
    from active.v17.schema_hand_mobileclip2_v17 import HandMobileCLIP2V17Config, schema_fingerprint as hand_fp

    config = MediaPipeFullV17Config()
    jobs = json.loads((REPORT / 'extra_jobs.json').read_text())
    job_index = {(j['source'], j['admitted_id']): j for j in jobs}
    CACHE.mkdir(parents=True, exist_ok=True)
    summary = {}
    for domain in DOMAINS:
        for split in ('train', 'val'):
            name = domain + '_' + split
            target_path = CACHE / (name + '.npz')
            if target_path.exists():
                raise FileExistsError(target_path)
            ids, targets = desired(name)
            with np.load(OLD / (name + '.npz'), allow_pickle=False) as old:
                index = {item: i for i, item in enumerate(old['item_ids'].tolist())}
                cached = [item for item in ids.tolist() if item in index]
                ix = [index[item] for item in cached]
                arrays = {k: old[k][ix] for k in ('landmarks', 'hand_embeddings', 'hand_valid', 'hand_boxes')}
            target_index = dict(zip(ids.tolist(), targets.tolist()))
            additions = {k: [] for k in arrays}
            added, missing = [], []
            for item in ids.tolist():
                if item in index:
                    continue
                job = job_index.get((name, item))
                if job is None or not Path(job['landmark']).is_file() or not Path(job['embedding']).is_file():
                    missing.append(item); continue
                lm = load_v17_archive(Path(job['landmark']), schema_fingerprint(config))
                if domain == 'local':
                    lm = mask_mouth_nodes_v17(lm)
                he, hv, hb = load_hand(Path(job['embedding']), derived_fingerprint(hand_fp(HandMobileCLIP2V17Config()), config))
                for key, value in zip(arrays, (lm, he, hv, hb)):
                    additions[key].append(value.numpy())
                added.append(item)
            for key in arrays:
                if added:
                    arrays[key] = np.concatenate((arrays[key], np.stack(additions[key]).astype(arrays[key].dtype)))
            included = cached + added
            np.savez(target_path, **arrays, targets=np.asarray([target_index[item] for item in included]), item_ids=np.asarray(included))
            summary[name] = {'expected_n': len(ids), 'actual_n': len(included), 'new': len(added),
                             'missing': missing, 'sha256': sha(target_path)}
            print(name, summary[name]['actual_n'], 'of', len(ids), flush=True)
    (REPORT / 'cache_receipt.json').write_text(json.dumps(summary, indent=2) + '\n')


def embed():
    from types import SimpleNamespace
    from scripts import extract_mediapipe_full_v17 as extraction
    jobs = json.loads((REPORT / 'extra_jobs.json').read_text())
    # Reuse the established embedding implementation on the explicit reviewed list.
    extraction.isolated_jobs = lambda output, sources: jobs
    extraction.run_embed(SimpleNamespace(output=EXTRA, pose_model='pose_landmarker_lite',
                                         sources=['local_train', 'local_val'], device='mps',
                                         mps_memory_fraction=0.25, archives_per_batch=16,
                                         image_batch_size=64))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'embed', 'cache'))
    args = parser.parse_args()
    {'prepare': prepare, 'embed': embed, 'cache': cache}[args.command]()
