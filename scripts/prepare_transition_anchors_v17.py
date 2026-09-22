"""Pin conservative interval anchors within already-admitted ASLLRP clips."""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.approved_phrase_data_v17 import digest, verify_manifest
from scripts.build_combined_dataset_v17 import load_features
from scripts.diagnose_combined_transitions_v17 import token_times


def main():
    verify_manifest()
    manifest_path = ROOT / 'data/local/combined_dataset_v17_20260922/manifest.json'
    annotations_path = ROOT / 'artifacts/reports/clean_boundary_subset_20260920/curated_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    annotations = json.loads(annotations_path.read_text())
    labels = manifest['label_to_index']
    records, counts = {}, Counter()
    for row in manifest['records']:
        if row['source'] != 'asllrp_contiguous':
            continue
        features, _ = load_features(row, labels)
        with np.load(ROOT / row['feature_path'], allow_pickle=False) as z:
            metadata = json.loads(str(z['metadata_json'].item()))
            times = token_times(z['window_source_ranges'], metadata['video_metadata']['fps'])
        assert len(times) == len(features) * 32 and np.all(np.diff(times) > 0)
        events = sorted((e for e in annotations['events'] if e['video'] == row['video_path']), key=lambda e: e['start'])
        assert all(e['role'] == row['role'] for e in events)
        regions = []
        for event in events:
            if event['kind'] != 'known' or not event['eligible']:
                continue
            assert event['label'] in row['target_sequence']
            margin = (event['end'] - event['start']) * .25
            indices = np.flatnonzero((times >= event['start'] + margin) & (times <= event['end'] - margin)).tolist()
            if indices:
                regions.append(dict(kind='sign_core', target=labels[event['label']] + 1, indices=indices, label=event['label'], start=event['start'] + margin, end=event['end'] - margin))
        for left, right in zip(events, events[1:]):
            if not (left['kind'] == right['kind'] == 'known' and left['eligible'] and right['eligible']):
                continue
            start, end = left['end'] + .05, right['start'] - .05
            if end <= start or any(e['start'] < end and e['end'] > start for e in events):
                continue
            indices = np.flatnonzero((times >= start) & (times <= end)).tolist()
            if indices:
                regions.append(dict(kind='internal_gap', target=0, indices=indices, start=start, end=end))
        used = set()
        for region in regions:
            assert not used.intersection(region['indices'])
            used.update(region['indices'])
            counts[f"{row['role']}:{region['kind']}"] += 1
        if regions:
            records[row['feature_path']] = dict(role=row['role'], feature_sha256=row['feature_sha256'], regions=regions)
    output = ROOT / 'artifacts/reports/transition_anchors_v17_20260922/anchors.json'
    value = dict(format='transition_interval_anchors_v17', inputs={str(p.relative_to(ROOT)): digest(p) for p in (manifest_path, annotations_path, Path(__file__), ROOT / 'scripts/diagnose_combined_transitions_v17.py')}, records=records, counts=dict(counts), limitation='Internal unannotated gaps in admitted complete ASLLRP sequences are a bounded blank-alignment hypothesis, not proof of physical rest. Ambiguous edges, padding and OTHER ignored. No local-phrase timing inferred.')
    with output.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(records=len(records), counts=dict(counts), sha256=digest(output))))


if __name__ == '__main__':
    main()
