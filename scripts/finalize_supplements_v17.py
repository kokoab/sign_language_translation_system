"""Pin existing supervised supplements; never acquire, train, or access test features."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.train_stage_1_v17 import Citizen100V17Dataset, load_v17_archive
from active.v17.schema_v17 import V17Config, schema_fingerprint, MOUTH_START, MOUTH_END
from active.v17.train_unified_streaming_ctc_v17 import validate_phrase_archive
from active.v17.train_stage1_window_v17 import _load_sequence
from active.v17.stage1_window_v17 import normalize_time_window

OUT = ROOT / 'artifacts/reports/supplement_finalization_v17_20260922'
CORES = ROOT / 'data/local/finalized_supplements_v17_20260922/o5s5'


def read(path):
    return json.loads((ROOT / path).read_text())


@lru_cache(None)
def digest(path):
    with (ROOT / path).open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def feature_check(x, *, require_hands=True):
    if x.shape[-3:] != (32, 61, 5) or not np.isfinite(x).all():
        raise ValueError('invalid feature shape or nonfinite values')
    if not np.isin(x[..., 3], (0, 1)).all():
        raise ValueError('nonbinary presence')
    if np.any(x[..., 4] < 0) or np.any(x[..., 4] > 1):
        raise ValueError('invalid confidence')
    if np.any(x[..., [0, 1, 2, 4]][x[..., 3] == 0] != 0):
        raise ValueError('nonzero missing nodes')
    windows = x.reshape(-1, 32, 61, 5)
    if require_hands and np.any((windows[:, :, :42, 3] > 0).any(axis=2).sum(axis=1) < 2):
        raise ValueError('insufficient visible hand frames')


def recover_tail(path):
    from scripts.recover_phrase_tails_v17 import recovery_ranges
    from scripts.extract_stage2_multimodal_v17 import save_archive
    from scripts.extract_stage2_transition_adapt_v17 import read_interval_frames
    from active.v17.extract_v17 import AppleVisionDetector, read_video_frames, extract_frames_v17, rotate_frame_clockwise
    from active.v17.schema_stage2_features_v17 import landmark_config
    from active.v17.schema_v17 import schema_payload
    with np.load(ROOT/path, allow_pickle=False) as d:
        m = json.loads(str(d['metadata_json'].item()))
        old, x, targets = d['window_source_ranges'], d['landmarks'], d['target_indices']
    count = m['sampled_source_frames']
    ranges = recovery_ranges(old, count)
    video = ROOT/m['video_path']
    if digest(m['video_path']) != m['video_sha256']:
        raise ValueError('recovery source hash mismatch')
    if 'interval_sampling' in m:
        t = m['interval_sampling']
        frames, timing = read_interval_frames(video,t['source_start_frame_inclusive'],t['source_end_frame_inclusive'])
        if timing['source_frame_indices'] != t['source_frame_indices']:
            raise ValueError('interval recovery changed sampled source indices')
    else:
        frames, timing = read_video_frames(video,count,1280,rotation='auto',input_mirrored=False)
        if timing['decoded_frame_count'] != count:
            raise ValueError('recovery decoded count changed')
    if len(frames)!=count:
        raise ValueError('recovery frame count mismatch')
    frames = [rotate_frame_clockwise(f,m['vision_coarse_rotation_clockwise']) for f in frames]
    config, detector = landmark_config(), AppleVisionDetector()
    features, diagnostics = [], []
    for left,right in ranges[-2:]:
        result = extract_frames_v17(frames[left:right],config,detector=detector)
        if result is None:
            raise ValueError('recovered window has insufficient hand observations')
        value = result.features.copy()
        if m.get('zero_lip_nodes'):
            value[:,MOUTH_START:MOUTH_END] = 0
        feature_check(value)
        features.append(value)
        diagnostics.append(result.diagnostics)
    x = np.concatenate((x[:-1],np.asarray(features)))
    for key in ('hand_valid_fraction','landmark_valid_windows','valid_windows','window_stride'):
        m.pop(key,None)
    m.update(format='recovered_landmark_only_phrase_v17',schema=schema_payload(config),
             schema_fingerprint=schema_fingerprint(config),window_count=len(ranges),dropped_tail_frames=0,
             window_diagnostics=m['window_diagnostics'][:-1]+diagnostics,
             recovery=dict(source_archive=path,source_archive_sha256=digest(path),old_ranges=old.tolist()))
    dest = CORES.parent/'recovered'/m['role']/Path(path).name
    dest.parent.mkdir(parents=True,exist_ok=True)
    save_archive(dest,dict(landmarks=x,window_source_ranges=ranges,target_indices=targets),m)
    return str(dest.relative_to(ROOT))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    classes = read('active/v17/citizen100_manifest.json')['classes']
    frozen = {r['canonical_label']: r for r in classes}
    labels = {k: v['class_index'] for k, v in frozen.items()}
    accepted, excluded, evidence = defaultdict(list), [], {}

    def pin(path):
        evidence[str(path)] = digest(path)
        return read(path)

    def attempt(source, role, path, annotation, representation, extra=None):
        base = dict(source=source, role=role, feature_path=str(path), **(extra or {}))
        try:
            label = annotation['canonical_label']
            if annotation['citizen_asl_lex_code'] != frozen[label]['citizen_asl_lex_code']:
                raise ValueError('exact ASL-LEX code mismatch')
            p = ROOT / path
            if representation == 'windowed_landmarks_v17':
                with np.load(p, allow_pickle=False) as d:
                    old_meta = json.loads(str(d['metadata_json'].item()))
                    incomplete = int(d['window_source_ranges'][-1,1]) != old_meta['sampled_source_frames']
                if incomplete:
                    base['original_feature_path'] = path
                    path = recover_tail(path)
                    p = ROOT/path
                    base['feature_path'] = path
            with np.load(p, allow_pickle=False) as d:
                meta = json.loads(str(d['metadata_json'].item()))
                if representation == 'isolated_v17':
                    x = load_v17_archive(p, schema_fingerprint(V17Config())).numpy()
                else:
                    x = d['landmarks']
                    validate_phrase_archive(x, d['window_source_ranges'], d['target_indices'], meta, labels)
                    if meta['target_sequence'] != [label] or meta['role'] != role:
                        raise ValueError('target or role mismatch')
                feature_check(x)
            video = meta['video_path']
            raw_hash = digest(video)
            expected = annotation.get('sha256', annotation.get('video_sha256', meta.get('video_sha256')))
            if expected and expected != raw_hash:
                raise ValueError('source video hash mismatch')
            base.update(canonical_label=label, class_index=labels[label], ctc_target=labels[label]+1,
                        citizen_asl_lex_code=annotation['citizen_asl_lex_code'], representation=representation,
                        feature_sha256=digest(path), video_path=video, video_sha256=raw_hash,
                        schema_fingerprint=meta['schema_fingerprint'], windows=int(x.size//(32*61*5)))
            accepted[source].append(base)
        except (ValueError, KeyError, FileNotFoundError) as error:
            excluded.append(dict(base, reason=str(error)))

    pin('active/v17/citizen100_manifest.json')
    rejection = 'data/local/citizen100_v17/rejections.csv'
    evidence[rejection] = digest(rejection)
    for split, role in [('train', 'train'), ('val', 'validation')]:
        official = f'data/local/dataset_metadata/asl_citizen_{split}.csv'
        evidence[official] = digest(official)
        lookup = {r['Video file']: r for r in csv.DictReader((ROOT / official).open())}
        ds = Citizen100V17Dataset(ROOT/'data/local/citizen100_v17/landmarks', split,
                                 ROOT/'active/v17/citizen100_manifest.json', ROOT/rejection, cache=False)
        selected = set(ds.files)
        for p in sorted((ds.root/split).glob('*/*.v17.npz')):
            if p not in selected:
                excluded.append(dict(source='citizen', role=role, feature_path=str(p.relative_to(ROOT)), reason='existing rejection ledger: incompletely segmented sign'))
                continue
            a = frozen[p.parent.name]
            official_row = lookup[p.name.removesuffix('.v17.npz')+'.mp4']
            assert official_row['Gloss'] == a['citizen_raw_gloss'] and official_row['ASL-LEX Code'] == a['citizen_asl_lex_code']
            attempt('citizen', role, str(p.relative_to(ROOT)), a, 'isolated_v17', {'signer_id':official_row['Participant ID']})
        print('citizen', role, len(accepted['citizen']), flush=True)

    provenance = pin('artifacts/generated/kaggle_stage1_orientation_robust_pull_v2/stage1_v17_orientation_robust_v1/training_data_provenance.json')
    assert provenance['semlex_train_only_approved'] is True
    for role, folder, selection, feature_dir in [
        ('train','train','full_clean_train_candidates.json','full_clean_landmarks_v17'),
        ('validation','val','selection_plan.json','landmarks_v17')]:
        prefix = f'data/local/semlex_citizen100_{folder}_audit'
        manifest = prefix+'/'+selection
        rows = pin(manifest)['videos']
        if role == 'train':
            assert digest(manifest) == provenance['semlex_manifest_sha256']
        for a in rows:
            path = f"{prefix}/{feature_dir}/{a['canonical_label']}/{a['semlex_video_id']}.v17.npz"
            attempt('semlex',role,path,a,'isolated_v17',dict(signer_id=a['semlex_signer_id'], source_item_id=a['semlex_video_id'], approval_basis='established exact mapping; frozen base explicitly approved train supplement'))
        print('semlex',role,len(accepted['semlex']),flush=True)

    asllrp = pin('data/local/asllrp_segmented_citizen100_v17/manifest.json')
    by_video = {a['path']:a for a in asllrp['videos']}
    review = 'artifacts/reports/asl_stem_wiki_manual_expansion_admission_v17/expert_review_queue.final.csv'
    evidence[review] = digest(review)
    reviewed = list(csv.DictReader((ROOT/review).open()))
    for source, root in [('asllrp_segmented','stage2_v17_asllrp_segmented_train_multimodal'),
                         ('asllrp_segmented','stage2_v17_asllrp_segmented_validation_multimodal'),
                         ('stem','stage2_v17_transition_adapt_v2/multimodal')]:
        for role in ['train','validation']:
            for p in sorted((ROOT/'data/local'/root/role).glob('*/*.npz')):
                with np.load(p, allow_pickle=False) as d:
                    m = json.loads(str(d['metadata_json'].item()))
                extra = dict(signer_id=m['signer_id'],source_item_id=m['source_item_id'])
                if source == 'asllrp_segmented':
                    a = by_video[m['video_path']]
                    assert a['occurrence'].rstrip('+') == a['signbank_annotation_id'].rstrip('+')
                    extra.update(parent_video=a['utterance_video_filename'], sign_start_frame=int(a['sign_start_frame']), sign_end_frame=int(a['sign_end_frame']))
                else:
                    timing = m['interval_sampling']
                    matches = [a for a in reviewed if a['video_sha256']==m['video_sha256'] and a['canonical_label']==m['target_sequence'][0] and a['verified_start_frame']==str(timing['source_start_frame_inclusive']) and a['verified_end_frame']==str(timing['source_end_frame_inclusive'])]
                    assert len(matches)==1
                    a = matches[0]
                    assert all(a[k]=='True' for k in ['training_eligible','variant_verified','boundary_verified','signer_quality_verified'])
                    extra['interval_sampling'] = timing
                attempt(source,role,str(p.relative_to(ROOT)),a,'windowed_landmarks_v17',extra)
        print(source,len(accepted[source]),flush=True)

    supervision = pin('artifacts/reports/o5s5_citizen100_v17/apple_vision_supervision.json')
    occurrences = 'artifacts/reports/o5s5_citizen100_v17/exact_occurrences.csv'
    evidence[occurrences] = digest(occurrences)
    assert digest(occurrences)==supervision['exact_occurrences_sha256']
    sequences = {r['source_item_id']: r for r in supervision['rows']}
    cache = {}
    for a in csv.DictReader((ROOT/occurrences).open()):
        source_id = a['source_item_id']
        s = sequences[source_id]
        row = dict(source='o5s5',role=a['role'],signer_id=a['signer_id'],source_item_id=source_id,ordinal=int(a['ordinal']))
        try:
            assert a['positive_training_eligible']=='True' and a['background_training_eligible']=='False'
            assert a['citizen_asl_lex_code']==frozen[a['canonical_label']]['citizen_asl_lex_code']
            assert a['role']==s['role']
            if digest(a['video_path']) != s['video_sha256']:
                raise ValueError('O5S5 video differs from verified supervision source')
            if source_id not in cache:
                cache[source_id] = _load_sequence(ROOT/s['archive_path'])
            raw,times = cache[source_id]
            start,end = float(a['start_seconds']),float(a['end_seconds'])
            assert any(i['exact_locked_match'] and i['label']==a['canonical_label'] and i['start_seconds']==start and i['end_seconds']==end for i in s['intervals'])
            if np.count_nonzero((times>=start-1e-9)&(times<=end+1e-9))<2:
                raise ValueError('fewer than two sampled frames inside annotated interval')
            x,diagnostics = normalize_time_window(raw,times,end,end-start)
            if diagnostics['observed_hand_frames']<2:
                raise ValueError('fewer than two actual hand observations inside annotated interval')
            feature_check(x)
            p = CORES/a['role']/f"{source_id}_{a['ordinal']}.npz"
            p.parent.mkdir(parents=True,exist_ok=True)
            row.update(canonical_label=a['canonical_label'],class_index=int(a['class_index']),ctc_target=int(a['class_index'])+1,
                       citizen_asl_lex_code=a['citizen_asl_lex_code'],representation='timestamp_normalized_positive_core',
                       start_seconds=start,end_seconds=end,video_path=a['video_path'],video_sha256=digest(a['video_path']),
                       raw_feature_path=s['archive_path'],raw_feature_sha256=digest(s['archive_path']),diagnostics=diagnostics,
                       background_training_eligible=False,feature_path=str(p.relative_to(ROOT)),windows=1)
            np.savez_compressed(p,features=x.astype(np.float32),metadata_json=np.array(json.dumps(row)))
            row['feature_sha256']=digest(str(p.relative_to(ROOT)))
            accepted['o5s5'].append(row)
        except ValueError as error:
            excluded.append(dict(row,reason=str(error)))
    print('o5s5',len(accepted['o5s5']),flush=True)

    # Exact duplicate videos are different from shared signer identities.
    # Retain established training copies; do not score the same bytes as validation.
    seen = {}
    for source, rows in accepted.items():
        keep = []
        for r in sorted(rows,key=lambda r:r['role']!='train'):
            interval = (r.get('start_seconds'),r.get('end_seconds'),json.dumps(r.get('interval_sampling'),sort_keys=True))
            key = (r['video_sha256'],interval)
            if key in seen:
                prior = seen[key]
                if prior['canonical_label'] != r['canonical_label']:
                    raise ValueError('duplicate video has conflicting labels')
                excluded.append(dict(r,reason='exact duplicate source bytes and interval',duplicate_of=prior['feature_path']))
            else:
                seen[key] = r
                keep.append(r)
        accepted[source] = keep

    # Signer overlap is descriptive, never an admission filter (user instruction).
    for source, rows in accepted.items():
        train_ids={r.get('signer_id') for r in rows if r['role']=='train'}
        for r in rows:
            r['signer_overlap_with_source_train']=r['role']=='validation' and r.get('signer_id') in train_ids
        output=dict(format='finalized_supervised_supplement_v17',source=source,
                    signer_policy='preserve existing roles; overlap does not disqualify',
                    counts=dict(Counter(r['role'] for r in rows)),records=rows,
                    evidence_sha256=evidence,exclusions=[r for r in excluded if r['source']==source])
        (OUT/f'{source}.json').write_text(json.dumps(output,indent=2)+'\n')
    summary=dict(counts={s:dict(Counter(r['role'] for r in rows)) for s,rows in accepted.items()},
                 accepted=sum(map(len,accepted.values())),excluded=len(excluded),
                 exclusions_by_reason=dict(Counter(r['reason'] for r in excluded)),
                 manifests={s+'.json':digest(str((OUT/(s+'.json')).relative_to(ROOT))) for s in accepted},
                 training_started=False,combined_manifest_created=False,protected_test_features_accessed=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'exclusions.json').write_text(json.dumps(excluded,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    run()
