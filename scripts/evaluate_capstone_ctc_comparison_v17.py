"""Frozen live CTC replay on the existing streaming comparison membership.

No fitting or selection. Uses the saved live CTC defaults and runtime window/rollover
functions. Different pipelines retain their own native sampling rates. Scores compare
whole-pipeline transcript output, not an isolated decoder replacement or phone speed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(65536), b''): h.update(b)
    return h.hexdigest()


def audit_common_validation(output):
    """Filter every pipeline by the same historical manifest roles, never by scores."""
    import torch
    from scripts.evaluate_temporal_boundary_v17 import edit_counts, summarize
    manifest_path = ROOT / 'active/v17/stage2_training_manifest_v17.json'
    specialist = torch.load(ROOT / 'artifacts/models/stage2_v17_signer_voice_ctc_pilot_v1/best_model.pth', map_location='cpu', weights_only=False)
    assert digest(manifest_path) == specialist['real_training_manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads((output/'protocol.json').read_text())
    by_hash = {r['video_sha256']: r for r in manifest['rows']}
    membership = [dict(id=r['source_item_id'], subset=r['subset'], historical_role=by_hash.get(r['video_sha256'], {}).get('role','not_matched_in_this_manifest')) for r in protocol['input_videos']]
    (output/'training_membership_audit.json').write_text(json.dumps(dict(scope='Direct video SHA256 match to saved specialist real-training manifest; other ancestral sources not exhaustively audited',manifest=str(manifest_path.relative_to(ROOT)),records=membership),indent=2)+'\n')
    ids = {r['id'] for r in membership if r['historical_role']=='validation'}
    assert len(ids)==21
    ctc=json.loads((output/'result.json').read_text())
    swift=json.loads((ROOT/'artifacts/reports/phone_speed_v17_20260929/score_test_C.json').read_text())
    earlier=json.loads((ROOT/'artifacts/reports/boundary_expanded_eval_v17_20260922/evaluation.json').read_text())
    earlier=next(r for r in earlier['runs'] if r['arm']=='frozen_pretrained_bio')
    result=dict(scope='Common historical-validation subset, selected by the saved CTC training manifest, not prediction scores. Excludes 51 known train-member videos; other ancestral sources are not exhaustively audited.',manifest_sha256=digest(manifest_path),ids=sorted(ids),results={})
    expected=None
    for name,d in [('CTC',ctc),('Earlier boundary-based recognition',earlier),('Current ATLAS',swift)]:
        records=[dict(id=r['id'],reference=r['reference'],hypothesis=r['hypothesis'],metrics=edit_counts(r['reference'],r['hypothesis'])) for r in d['records'] if r['id'] in ids]
        refs={r['id']:r['reference'] for r in records}
        assert len(refs)==len(ids)
        if expected is not None:assert refs==expected
        expected=refs
        result['results'][name]=dict(overall=summarize(records),records=records)
    (output/'common_validation.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    from scripts import live_stage2_ctc_v17 as ctc
    from scripts.segmental_lab_v17 import rows_for
    from scripts.evaluate_temporal_boundary_v17 import observations, edit_counts, summarize
    from active.v17.extract_v17 import AppleVisionDetector
    from active.v17.train_stage_2_other_ctc_v17 import directory_sha256
    reference_path = ROOT / 'artifacts/reports/phone_speed_v17_20260929/score_test_C.json'
    swift = json.loads(reference_path.read_text())
    rows = rows_for('test')
    reference = {r['id']: r['reference'] for r in swift['records']}
    assert len(rows) == len(reference) == 72
    assert {r['source_item_id']: r['target_sequence'] for r in rows} == reference
    for row in rows:
        assert digest(ROOT / row['video_path']) == row['video_sha256']
    args = ctc.parser().parse_args(['--no-display', '--no-speech', '--no-live-lips'])
    paths = {k: getattr(args, k) for k in ['image_encoder','stage2_encoder','stage2_primary','stage2_specialist','stage2_selector','selector_report','vocabulary']}
    hashes = {k: directory_sha256(p) if p.is_dir() else digest(p) for k,p in paths.items()}
    protocol = dict(model='saved general live CTC selector', models={k:str(p.relative_to(ROOT)) for k,p in paths.items()}, sha256=hashes,
                    processing_fps=args.processing_fps, window_seconds=args.window_seconds,
                    selector_config=json.loads(args.selector_report.read_text())['selector_config'],
                    swift_reference_sha256=digest(reference_path), native_pipeline_comparison=True,
                    scope='same recorded videos and transcripts; CPU/Core ML Mac replay; not phone timing',
                    checkpoint_selection='fixed live-script defaults before inference',
                    protected_citizen_test_accessed=False, input_videos=[{k:r[k] for k in ['source_item_id','video_path','video_sha256','target_sequence','subset']} for r in rows])
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    classifier = ctc.LiveStage2CTC(args)
    detector = AppleVisionDetector(args.minimum_point_confidence)
    records=[]; start=time.monotonic()
    for row in rows:
        frozen=[]; locked=[]; context=[]; positions=[]; raw=[]; previous=0; hyp=[]; accepted=0; rejected=0
        buf=ctc.ElapsedWindowBuffer(args.window_seconds)
        def submit(chunk):
            nonlocal previous,locked,context,positions,raw,hyp,accepted,rejected
            if len(frozen)>=ctc.MAXIMUM_WINDOWS:
                previous=raw[ctc.TOKENS_PER_WINDOW-1]
                del raw[:ctc.TOKENS_PER_WINDOW]
                locked,context,positions=ctc.roll_ctc_prefix(locked,context,positions)
                frozen.pop(0);hyp=locked+context
            feature,result=classifier.classify_window(chunk,list(frozen),previous)
            if feature is None:
                rejected+=1;return
            frozen.append(feature);accepted+=1
            context=list(result['hypothesis']);positions=list(result['token_positions']);raw=list(result['ctc_argmax'])
            hyp=locked+context
        for obs in observations(row,args,detector):
            chunk=buf.add(obs)
            if chunk is not None: submit(chunk)
        chunk=buf.finish()
        if chunk is not None: submit(chunk)
        metrics=edit_counts(row['target_sequence'],hyp)
        record=dict(id=row['source_item_id'],subset=row['subset'],reference=row['target_sequence'],hypothesis=hyp,metrics=metrics,accepted_windows=accepted,rejected_windows=rejected)
        records.append(record)
        with (a.output/'predictions.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(f'{len(records)}/{len(rows)} elapsed={time.monotonic()-start:.1f}s',flush=True)
    result=dict(overall=summarize(records),subsets={name:summarize([r for r in records if r['subset']==name]) for name in sorted({r['subset'] for r in records})},records=records,protocol='protocol.json')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    audit_common_validation(a.output)
    print(json.dumps(result['overall']),flush=True)


if __name__=='__main__':main()
