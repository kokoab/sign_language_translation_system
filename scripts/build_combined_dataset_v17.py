"""Combine finalized supervision without changing splits, labels, or source files."""
import json
from collections import Counter
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.approved_phrase_data_v17 import digest, verify_manifest
from active.v17.train_unified_streaming_ctc_v17 import validate_phrase_archive
from scripts.finalize_supplements_v17 import feature_check

OUT = ROOT/'data/local/combined_dataset_v17_20260922/manifest.json'
REPORT = ROOT/'artifacts/reports/combined_dataset_v17_20260922'
SUPPLEMENTS = ROOT/'artifacts/reports/supplement_finalization_v17_20260922'
BASELINE = ROOT/'active/v17/approved_phrase_manifest_20260921_v2.json'


def load_features(row, labels):
    """Return model-compatible windows and explicit nonblank CTC targets."""
    path = ROOT/row['feature_path']
    if digest(path) != row['feature_sha256']:
        raise ValueError(f'feature hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as d:
        if row['representation']=='windowed_landmarks_v17':
            x = d['landmarks']
            metadata = json.loads(str(d['metadata_json'].item()))
            validate_phrase_archive(x,d['window_source_ranges'],d['target_indices'],metadata,labels)
            if (d['target_indices']+1).tolist()!=row['ctc_targets']:
                raise ValueError('CTC targets differ from archive')
        elif row['representation'] in ('isolated_v17','timestamp_normalized_positive_core'):
            x = d['features'][None]
            if row['ctc_targets'] != [labels[row['canonical_label']]+1]:
                raise ValueError('isolated target mismatch')
        else:
            raise ValueError('unsupported feature representation')
        # Approved phrases retain natural rest/transition windows; the positive-core
        # per-window hand floor applies only to isolated/segmented sign supervision.
        feature_check(x,require_hands=row['supervision']=='single_verified_sign')
    return x, row['ctc_targets']


def build():
    verify_manifest(BASELINE)
    baseline = json.loads(BASELINE.read_text())
    summary = json.loads((SUPPLEMENTS/'summary.json').read_text())
    frozen = ROOT/'active/v17/citizen100_manifest.json'
    labels = {r['canonical_label']:r['class_index'] for r in json.loads(frozen.read_text())['classes']}
    pins = {str(BASELINE.relative_to(ROOT)):digest(BASELINE),str(frozen.relative_to(ROOT)):digest(frozen)}
    records=[]
    for name,expected in summary['manifests'].items():
        path=SUPPLEMENTS/name
        if digest(path)!=expected:raise ValueError(f'supplement manifest changed: {name}')
        pins[str(path.relative_to(ROOT))]=expected
        data=json.loads(path.read_text())
        for evidence,sha in data['evidence_sha256'].items():
            if digest(ROOT/evidence)!=sha:raise ValueError(f'evidence changed: {evidence}')
        for r in data['records']:
            row=dict(r,ctc_targets=[r['ctc_target']],target_sequence=[r['canonical_label']],
                     supervision='single_verified_sign',origin_manifest=str(path.relative_to(ROOT)))
            row.pop('ctc_target')
            records.append(row)
    for r in baseline['admitted']:
        with np.load(ROOT/r['path'],allow_pickle=False) as d:
            metadata=json.loads(str(d['metadata_json'].item()))
        records.append(dict(source=r['source'],role=r['role'],feature_path=r['path'],
                            feature_sha256=r['sha256'],video_sha256=r['video_sha256'],
                            video_path=metadata['video_path'],signer_id=r.get('signer'),
                            source_item_id=r['identity'],representation='windowed_landmarks_v17',
                            target_sequence=metadata['target_sequence'],ctc_targets=[t+1 for t in r['target_indices']],
                            supervision='approved_phrase_or_subspan',origin_manifest=str(BASELINE.relative_to(ROOT))))
    seen=set();windows=0
    raw_hashes={}
    for row in records:
        path=row['feature_path']
        if path in seen:raise ValueError('duplicate combined feature path')
        seen.add(path)
        if row['role'] not in ('train','validation'):raise ValueError('invalid split')
        video=row['video_path']
        if video not in raw_hashes:raw_hashes[video]=digest(ROOT/video)
        if raw_hashes[video]!=row['video_sha256']:raise ValueError('video hash mismatch')
        x,_=load_features(row,labels);windows+=len(x)
    overlap=json.loads((SUPPLEMENTS/'verification.json').read_text())['asllrp_shared_parent_with_baseline']
    counts=dict(Counter(r['role'] for r in records))
    if counts!={'train':4547,'validation':1874}:raise ValueError(f'unexpected combined counts: {counts}')
    manifest=dict(format='combined_supervised_dataset_v17',version=1,data_ready=True,
                  training_started=False,training_recipe=None,
                  scope='Existing approved phrases and finalized supervised supplements; no test inputs',
                  counts=counts,record_count=len(records),label_to_index=labels,ctc_blank=0,ctc_other=101,
                  source_counts={s:dict(Counter(r['role'] for r in records if r['source']==s)) for s in sorted({r['source'] for r in records})},
                  source_manifests_sha256=pins,
                  policy=dict(splits='preserve existing train/validation roles; validation is not training',
                              signer_overlap='allowed; familiar-signer evaluation must be identified',
                              unknown='only existing approved OTHER targets; no inferred OOV or background labels',
                              sampling='mix training sources within a run; shared-parent segments are correlated',
                              temporal='windowed archives retain source ranges; do not concatenate isolated signs into purported real phrases'),
                  baseline_parent_overlap=overlap,records=records)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(manifest,indent=2)+'\n')
    # Reopen the saved manifest and all listed feature files, not just in-memory rows.
    saved=json.loads(OUT.read_text())
    for row in saved['records']:load_features(row,labels)
    REPORT.mkdir(parents=True,exist_ok=True)
    result=dict(manifest=str(OUT.relative_to(ROOT)),sha256=digest(OUT),records=len(records),
                counts=counts,windows_verified=windows,source_counts=manifest['source_counts'],
                shared_parent_records=len(overlap),training_started=False)
    (REPORT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':build()
