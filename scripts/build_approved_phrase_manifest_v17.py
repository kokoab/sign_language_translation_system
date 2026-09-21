"""Build a non-destructive, whole-sequence admission view of recovered phrase caches."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.approved_phrase_data_v17 import APPROVED_ROOT, DEFAULT_MANIFEST, digest, verify_manifest
from active.v17.train_unified_streaming_ctc_v17 import validate_phrase_archive


def run():
    if APPROVED_ROOT.exists() or DEFAULT_MANIFEST.exists():
        raise FileExistsError('approved version already exists; create a new version instead')
    audit = ROOT/'artifacts/reports/annotation_identity_audit_v17_20260921'
    recovery = ROOT/'artifacts/reports/phrase_tail_recovery_v17_20260921'
    inputs = [audit/'clip_admission.json',audit/'event_ledger.json',recovery/'inventory.json',
              recovery/'verification.json',ROOT/'active/v17/citizen100_manifest.json']
    clips = json.loads(inputs[0].read_text())
    events = defaultdict(list)
    for row in json.loads(inputs[1].read_text()): events[row['archive']].append(row)
    inventory = {r['source']:r for r in json.loads(inputs[2].read_text())}
    labels = {r['canonical_label']:r['class_index'] for r in json.loads(inputs[4].read_text())['classes']}
    admitted, excluded = [], []
    for root in ('phrases','other'):
        for role in ('train','validation'): (APPROVED_ROOT/root/role).mkdir(parents=True)
    for clip in clips:
        old = clip['archive']; repaired = inventory[old]; path = ROOT/repaired['output']
        selected = events[old]
        reason = None
        if repaired.get('invalid_new_windows',0): reason='insufficient hand detections in rebuilt window'
        elif clip['mapping_status']=='existing_reviewed_known': pass
        elif clip['source']=='asllrp_other_ctc':
            if not selected or any(e['status']=='unresolved' for e in selected): reason='unresolved annotation identity'
            elif any(not e['complete_event'] or e['overlapping_event'] for e in selected): reason='clipped or overlapping annotation'
            elif not clip['target_reconstruction_matches']: reason='targets disagree with established identities'
        else: reason='unresolved cross-corpus identity mapping'
        if reason:
            excluded.append(dict(original=old,recovered=repaired['output'],source=clip['source'],role=clip['role'],reason=reason))
            continue
        with np.load(path,allow_pickle=False) as d:
            m=json.loads(str(d['metadata_json'].item()))
            validate_phrase_archive(d['landmarks'],d['window_source_ranges'],d['target_indices'],m,labels)
            assert m['role']==clip['role'] and m['source_item_id']==clip['identity']
            targets=d['target_indices'].tolist()
        assert digest(ROOT/old)==repaired['source_sha256']
        if repaired['action']=='recovered': assert digest(path)==repaired['output_sha256']
        group='other' if clip['source']=='asllrp_other_ctc' else 'phrases'
        output=APPROVED_ROOT/group/clip['role']/clip['source']/path.name
        output.parent.mkdir(parents=True,exist_ok=True); output.symlink_to(path)
        admitted.append(dict(path=str(output.relative_to(ROOT)),sha256=digest(path),
                             original=old,recovered=repaired['output'],source=clip['source'],
                             role=clip['role'],signer=m['signer_id'],identity=clip['identity'],
                             video_sha256=m['video_sha256'],target_indices=targets))
    # Preserve signer-disjoint roles within each source and prevent cross-role exact-video overlap.
    for field in ('video_sha256','signer'):
        train={(r['source'] if field=='signer' else '',r[field]) for r in admitted if r['role']=='train'}
        val={(r['source'] if field=='signer' else '',r[field]) for r in admitted if r['role']=='validation'}
        assert not train & val, (field,train & val)
    manifest=dict(format='approved_phrase_data_v17',version=1,training_ready=False,
                  blockers=['Existing recipes require excluded NCSLGR/OTHER validation sources; update objective/selection metrics.',
                            'Blank/rest supervision and unseen-OOV evaluation are not approved; auxiliary training inputs must be pinned before enabling training.'],
                  scope='Phrase admission only; no certification of isolated, RGB, blank/rest or unseen-OOV training readiness.',
                  roots={k:str((APPROVED_ROOT/v).relative_to(ROOT)) for k,v in [('phrase_root','phrases'),('other_root','other')]},
                  evidence_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs},
                  counts=dict(Counter(r['source']+':'+r['role'] for r in admitted)),
                  exclusion_counts=dict(Counter(r['reason'] for r in excluded)),
                  policy='Exclude entire ambiguous/clipped/overlapping sequences; never remove targets while retaining their video. Preserve originals.',
                  unseen_oov_evaluation=[],admitted=admitted,excluded=excluded)
    DEFAULT_MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(verify_manifest(),indent=2));print(json.dumps(manifest['counts'],indent=2))


if __name__=='__main__': run()
