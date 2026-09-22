"""Validate prepared membership and poses; never enable training or modify caches."""
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE=Path(__file__).parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE/'dependencies')]
import numpy as np
from pose_format import Pose
from active.v17.approved_phrase_data_v17 import digest, verify_manifest


def main():
    started=time.perf_counter()
    path=HERE/'prepared_manifest.json'
    manifest_bytes=path.read_bytes()
    data=json.loads(manifest_bytes)
    source=ROOT/data['source_manifest']
    if digest(source)!=data['source_sha256']:raise ValueError('source manifest changed')
    canonical=verify_manifest()
    original=json.loads(source.read_text())
    expected={r['item']:r for r in original['records']}
    if len(expected)!=len(original['records']):raise ValueError('duplicate source identity')
    for file,sha in data['code_sha256'].items():
        if digest(ROOT/file)!=sha:raise ValueError('extraction code changed: '+file)
    records=[];errors=[];seen=set();parents={};signers={}
    for row in data['records']:
        item=row['item'];failures=[]
        if item in seen:failures.append('duplicate prepared identity')
        seen.add(item)
        ref=expected.get(item)
        if ref is None or any(row.get(k)!=v for k,v in ref.items()):failures.append('source membership/contract mismatch')
        try:
            for field in ('video','pose'):
                if digest(ROOT/row[field+'_path'])!=row[field+'_sha256']:failures.append(field+' hash mismatch')
            with (ROOT/row['pose_path']).open('rb') as f:pose=Pose.read(f)
            raw=pose.body.data.filled(0);confidence=pose.body.confidence
            if raw.shape!=(row['pose_frames'],1,50,3):failures.append('pose shape mismatch')
            if not np.isfinite(raw).all() or not np.isfinite(confidence).all():failures.append('nonfinite values')
            if np.any((confidence<0)|(confidence>1)):failures.append('confidence outside0..1')
            if abs(pose.body.fps-row['pose_fps'])>.001:failures.append('fps mismatch')
            if (pose.header.dimensions.width,pose.header.dimensions.height)!=(row['width'],row['height']):failures.append('dimensions mismatch')
            for start,end in row['intervals']:
                if not np.isfinite([start,end]).all() or not -.05<=start<end<=(len(raw)-1)/row['pose_fps']+.05:
                    failures.append('interval outside video clock')
            hand=pose.get_components(['LEFT_HAND_LANDMARKS','RIGHT_HAND_LANDMARKS'])
            visible=np.any(hand.body.confidence[:,0]>.5,axis=-1)
            records.append(dict(item=item,role=row['role'],frames=len(raw),intervals=len(row['intervals']),
                                hand_visible_frame_fraction=float(visible.mean()),errors=failures))
        except Exception as exc:failures.append(type(exc).__name__+': '+str(exc))
        parents.setdefault((row['source'],row['parent']),set()).add(row['role'])
        signers.setdefault(row['signer'],set()).add(row['role'])
        if failures:errors.append(dict(item=item,errors=failures))
    parent_overlap=[list(k) for k,v in parents.items() if len(v)>1]
    signer_overlap=[k for k,v in signers.items() if len(v)>1]
    missing=sorted(set(expected)-seen)
    result=dict(status='passed' if not errors and not parent_overlap and not signer_overlap and not missing else
                'partial_pass' if not errors and not parent_overlap and not signer_overlap else 'failed',
                expected=len(expected),validated=len(data['records']),missing=missing,errors=errors,
                parent_role_overlap=parent_overlap,signer_role_overlap=signer_overlap,
                roles=dict(Counter(r['role'] for r in data['records'])),
                intervals=sum(len(r['intervals']) for r in data['records']),frames=sum(r['frames'] for r in records),
                low_hand_visibility=[r for r in records if r['hand_visible_frame_fraction']<.5],
                records=records,canonical_verification=canonical,training_ready=False,
                note='Structural validation, not visual boundary truth or recognition accuracy. Low hand visibility is diagnostic, not an automatic exclusion.',
                elapsed_seconds=time.perf_counter()-started)
    out=HERE/'validation.json';temp=out.with_suffix('.json.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(out)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','canonical_verification','low_hand_visibility')}))
    print('low_hand_visibility_records',len(result['low_hand_visibility']))
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
