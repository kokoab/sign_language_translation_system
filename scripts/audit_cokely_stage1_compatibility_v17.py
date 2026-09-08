#!/usr/bin/env python3
"""Measure frozen Stage-1 agreement on audited Cokely annotation segments."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ''):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.extract_stage2_multimodal_v17 import safe_name


def segment_score(scores, positions, start, end):
    selected = scores[(positions >= start) & (positions < end)]
    if not len(selected):
        selected = scores[[int(np.abs(positions - (start + end) / 2).argmin())]]
    return selected.mean(axis=0)


def run(args):
    manifest = json.loads(args.manifest.read_text())
    index_to_label = {value: key for key, value in manifest['label_to_index'].items()}
    results = []
    for row in manifest['rows']:
        stem = safe_name(row)
        crop_path = (args.crop_root / row['role'] / row['source'] /
                     f'{stem}.stage2_rgb_v17.npz')
        frozen_path = (args.frozen_root / row['role'] / row['source'] /
                       f'{stem}.stage2_frozen_v17.npz')
        with np.load(crop_path, allow_pickle=False) as payload:
            ranges = payload['window_source_ranges'].astype(np.float32)
            metadata = json.loads(str(payload['metadata_json']))
        with np.load(frozen_path, allow_pickle=False) as payload:
            scores = payload['frozen_features'].astype(np.float32)[..., -100:]
            frozen_metadata = json.loads(str(payload['metadata_json']))
        positions = np.concatenate([
            np.linspace(start, end - 1, 32) for start, end in ranges
        ])
        flat_scores = scores.reshape(-1, 100)
        duration = row['end_ms'] - row['start_ms']
        frame_count = metadata['sampled_source_frames']
        for token_index, (label, target, interval) in enumerate(zip(
                row['target_sequence'], row['target_indices'], row['token_intervals_ms'])):
            start = (interval[0] - row['start_ms']) / duration * frame_count
            end = (interval[1] - row['start_ms']) / duration * frame_count
            value = segment_score(flat_scores, positions, start, end)
            order = np.argsort(value)[::-1]
            probabilities = np.exp(value - value.max())
            probabilities /= probabilities.sum()
            repeated = ((token_index and row['target_sequence'][token_index - 1] == label) or
                        (token_index + 1 < len(row['target_sequence']) and
                         row['target_sequence'][token_index + 1] == label))
            results.append({
                'source_item_id': row['source_item_id'],
                'signer_id': row['signer_id'],
                'annotation_id': row['annotation_ids'][token_index],
                'raw_gloss': row['raw_glosses'][token_index],
                'canonical_label': label,
                'asl_lex_code': row['target_asl_lex_codes'][token_index],
                'top1': index_to_label[int(order[0])],
                'top5': [index_to_label[int(value)] for value in order[:5]],
                'target_probability': float(probabilities[target]),
                'target_rank': int(np.where(order == target)[0][0]) + 1,
                'repeated_annotation_run': bool(repeated),
                'variant_verified': False,
                'training_eligible': False,
                'review_status': ('repetition_tokenization_review_required' if repeated
                                  else 'exact_asl_lex_variant_review_required'),
                'stage1_checkpoint_sha256': frozen_metadata['stage1_checkpoint_sha256'],
            })
    by_signer = defaultdict(Counter)
    if not results:
        raise ValueError('manifest contains no candidate tokens')
    for item in results:
        by_signer[item['signer_id']]['tokens'] += 1
        by_signer[item['signer_id']]['top1'] += item['top1'] == item['canonical_label']
        by_signer[item['signer_id']]['top5'] += item['canonical_label'] in item['top5']
    report = {
        'format': 'slt_cokely_stage1_compatibility_audit_v17',
        'manifest': args.manifest.as_posix(),
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'stage1_checkpoint_sha256': results[0]['stage1_checkpoint_sha256'],
        'tokens': len(results),
        'top1_correct': sum(x['top1'] == x['canonical_label'] for x in results),
        'top5_correct': sum(x['canonical_label'] in x['top5'] for x in results),
        'repeated_annotation_tokens': sum(x['repeated_annotation_run'] for x in results),
        'variant_verified': 0,
        'training_eligible': 0,
        'by_signer': {key: dict(value) for key, value in sorted(by_signer.items())},
        'citizen_test_accessed': False,
        'semlex_test_accessed': False,
        'local_test_accessed': False,
        'rows': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--crop-root', type=Path, required=True)
    parser.add_argument('--frozen-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps({key: report[key] for key in (
        'tokens', 'top1_correct', 'top5_correct', 'repeated_annotation_tokens',
        'variant_verified', 'training_eligible')}, indent=2))


if __name__ == '__main__':
    main()
