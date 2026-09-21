"""Audit existing annotation identities and prepare development-only sign cores."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_asllrp_continuous_citizen100_v17 import read_sentence_csv, occurrence_matches
from scripts.finalize_asllrp_other_ctc_manifest_v17 import sha256
from active.v17.train_streaming_tcn_ctc_v17 import restore_source_frames


def classify_identity(variant, occurrence, kind, index, locked_codes, locked_families):
    matches = index.get(variant, [])
    codes = {r['Code'] for r in matches}
    if not occurrence_matches(variant, occurrence):
        return 'unresolved', None, 'occurrence differs from official variant'
    if len(codes) != 1 or codes & {'', 'NA'}:
        return 'unresolved', None, 'missing or ambiguous official identity'
    code = next(iter(codes))
    if code in locked_codes:
        return 'known', code, 'exact official identity'
    if kind != 'Lexical Signs':
        return 'unresolved', code, 'nonlexical category not admitted as lexical OOV'
    if any(r['LemmaID'] in {'', 'NA'} or r['LemmaID'] in locked_families for r in matches):
        return 'unresolved', code, 'missing lemma or variant of locked lemma'
    return 'oov', code, 'official identity and lemma outside locked vocabulary'


def run():
    output = ROOT / 'artifacts/reports/annotation_identity_audit_v17_20260921'
    output.mkdir(exist_ok=True)
    classes_path = ROOT / 'active/v17/citizen100_manifest.json'
    lex_path = ROOT / 'data/local/dataset_metadata/asllex2_official/signdata.csv'
    source_path = ROOT / 'data/local/dataset_metadata/asllrp_signbank/asllrp_sentence_signs_2025_06_28.csv'
    acquisition_path = ROOT / 'data/local/asllrp_other_ctc_v17/manifest.json'
    integrity_path = ROOT / 'artifacts/reports/phrase_contract_repair_v17_20260921/validation.json'
    classes = json.loads(classes_path.read_text())['classes']
    locked_codes = {r['citizen_asl_lex_code'] for r in classes}
    code_targets = {r['citizen_asl_lex_code']: r['class_index'] + 1 for r in classes}
    with lex_path.open(encoding='latin1', newline='') as f:
        lex = list(csv.DictReader(f))
    index = defaultdict(list)
    for row in lex:
        if row['SignBankAnnotationID'].strip() not in {'', 'NA'}:
            index[row['SignBankAnnotationID'].strip()].append(row)
    families = {r['LemmaID'] for r in lex if r['Code'] in locked_codes}
    source, invalid_source = read_sentence_csv(source_path)
    parents = defaultdict(list)
    for row in source:
        if row.get('Hidden') != 'T':
            parents[row['Utterance video filename']].append(row)
    spans = {f"asllrp_other_ctc:{Path(s['utterance_video_filename']).stem}:span{int(s['span_index_in_utterance']):02d}": s
             for s in json.loads(acquisition_path.read_text())['spans']}
    blocked = {r['path'] for r in json.loads(integrity_path.read_text())['rejected_archives']}
    roots = ['stage2_v17_grounded_phrases_fixed_20260921', 'stage2_v17_asllrp_other_multimodal',
             'stage2_v17_flores_other_20260921']
    events, clips = [], []
    training_codes = set()
    for root in roots:
        for role in ('train', 'validation'):
            for archive in sorted((ROOT / 'data/local' / root / role).glob('*/*.npz')):
                with np.load(archive, allow_pickle=False) as d:
                    m = json.loads(str(d['metadata_json'].item()))
                    targets = (d['target_indices'] + 1).tolist()
                path = str(archive.relative_to(ROOT))
                item = dict(archive=path, identity=m['source_item_id'], source=m['source'], role=role,
                            complete_cache=path not in blocked, stored_targets=targets)
                if m['source'] != 'asllrp_other_ctc':
                    trusted = m['source'] == 'asllrp_contiguous' or (
                        m['source'] == 'local_phrases' and bool(m.get('phrase_review')))
                    item.update(mapping_status='existing_reviewed_known' if trusted else 'unresolved_cross_corpus_mapping',
                                full_sequence_eligible=trusted and path not in blocked)
                    clips.append(item)
                    continue
                span = spans[m['source_item_id']]
                left, right = int(span['crop_start_frame_local']), int(span['crop_end_frame_local']) + 1
                selected = []
                for row in parents[span['utterance_video_filename']]:
                    origin = int(row['Start frame of the containing utterance'])
                    start = int(row['Start frame of the sign video']) - origin
                    end = int(row['End frame of the sign video']) - origin + 1
                    if start >= right or end <= left:
                        continue
                    status, code, reason = classify_identity(row['Entry/variant gloss label'], row['Occurrence label'],
                                                            row['Sign type'], index, locked_codes, families)
                    event = dict(identity=item['identity'], archive=path, role=role, signer=m['signer_id'],
                                 video_path=m['video_path'], source_video_sha256=m['video_sha256'],
                                 parent_utterance=span['utterance_video_filename'],
                                 annotation_id=row['Video ID number'], variant=row['Entry/variant gloss label'],
                                 occurrence=row['Occurrence label'], sign_type=row['Sign type'],
                                 status=status, asllex_code=code, reason=reason,
                                 start_frame=start-left, end_frame_exclusive=end-left,
                                 complete_event=start >= left and end <= right,
                                 complete_cache=item['complete_cache'])
                    selected.append(event)
                    if role == 'train' and code:
                        training_codes.add(code)
                selected.sort(key=lambda e: (e['start_frame'], e['end_frame_exclusive'], e['annotation_id']))
                rebuilt = []
                for event in selected:
                    token = code_targets.get(event['asllex_code'], 101)
                    if not rebuilt or token != 101 or rebuilt[-1] != 101:
                        rebuilt.append(token)
                    event['overlapping_event'] = any(
                        other is not event and min(event['end_frame_exclusive'], other['end_frame_exclusive']) >
                        max(event['start_frame'], other['start_frame']) for other in selected)
                item.update(mapping_status='event_ledger', event_counts=dict(Counter(e['status'] for e in selected)),
                            target_reconstruction_matches=rebuilt == targets,
                            full_sequence_eligible=bool(selected) and item['complete_cache'] and rebuilt == targets and
                            all(e['status'] != 'unresolved' and e['complete_event'] and not e['overlapping_event'] for e in selected))
                clips.append(item)
                events.extend(selected)

    # Development cores only: retain existing validation signer, never claim new test truth.
    candidates = [e for e in events if e['role'] == 'validation' and e['status'] in {'known', 'oov'}
                  and e['complete_cache'] and e['complete_event'] and not e['overlapping_event']]
    cores, seen, features = [], set(), {}
    for event in candidates:
        key = (event['parent_utterance'], event['annotation_id'])
        if key in seen:
            continue
        seen.add(key)
        archive = event['archive']
        if archive not in features:
            with np.load(ROOT / archive, allow_pickle=False) as d:
                features[archive] = restore_source_frames(d['landmarks'], d['window_source_ranges'])
        core = features[archive][event['start_frame']:event['end_frame_exclusive']]
        hand_frames = (core[:, :42, 3] > 0).any(axis=1)
        if hand_frames.sum() < 2 or hand_frames.mean() < .8:
            continue
        cores.append({**event, 'expected_ctc_index': code_targets.get(event['asllex_code'], 101),
                      'hand_frame_fraction': float(hand_frames.mean()),
                      'identity_exposure': 'seen_in_asllrp_train' if event['asllex_code'] in training_codes
                      else 'unresolved_other_training_sources', 'evaluation_role': 'reused_development_only'})
    counts = Counter((e['role'], e['status']) for e in events)
    summary = dict(archives=len(clips), events=len(events), event_counts={':'.join(k): v for k,v in counts.items()},
                   full_sequence_eligible=dict(Counter(c['source']+':'+c['role'] for c in clips if c['full_sequence_eligible'])),
                   development_cores=dict(Counter(c['status'] for c in cores)),
                   oov_development_identities=len({c['asllex_code'] for c in cores if c['status']=='oov'}),
                   oov_exposure=dict(Counter(c['identity_exposure'] for c in cores if c['status']=='oov')),
                   development_signers=sorted({c['signer'] for c in cores}),
                   invalid_source_rows=len(invalid_source), independent_unseen_oov_benchmark_ready=False,
                   acquisition_started=False, training_started=False, source_annotations_changed=False,
                   input_hashes={str(p.relative_to(ROOT)): sha256(p) for p in
                                 (classes_path, lex_path, source_path, acquisition_path, integrity_path)})
    for name, value in [('summary.json', summary), ('clip_admission.json', clips),
                        ('event_ledger.json', events), ('development_cores.json', cores)]:
        (output / name).write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()
