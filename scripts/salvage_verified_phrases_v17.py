"""Recover only complete, gap-free verified multi-sign spans from excluded clips."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from active.v17.approved_phrase_data_v17 import digest, verify_manifest
from active.v17.extract_v17 import AppleVisionDetector, read_video_frames, rotate_frame_clockwise, extract_frames_v17
from active.v17.schema_stage2_features_v17 import landmark_config
from active.v17.schema_v17 import schema_payload, schema_fingerprint
from active.v17.train_unified_streaming_ctc_v17 import validate_phrase_archive, rolling_windows
from active.v17.train_streaming_tcn_ctc_v17 import restore_source_frames
from scripts.extract_stage2_multimodal_v17 import save_archive


def trusted_runs(rows):
    output, run=[],[]
    for row in sorted(rows,key=lambda e:e['start_frame'])+[None]:
        good=row and row['status'] in ('known','oov') and row['complete_event'] and not row['overlapping_event']
        if not good or run and row['start_frame'] != run[-1]['end_frame_exclusive']:
            if len(run)>=2 and any(e['status']=='known' for e in run): output.append(run)
            run=[]
        if good: run.append(row)
    return output


def run():
    old_path=ROOT/'active/v17/approved_phrase_manifest_20260921.json'
    verify_manifest(old_path)
    old=json.loads(old_path.read_text())
    ledger=ROOT/'artifacts/reports/annotation_identity_audit_v17_20260921/event_ledger.json'
    excluded={r['original']:r for r in old['excluded'] if r['source']=='asllrp_other_ctc'}
    groups=defaultdict(list)
    all_events=json.loads(ledger.read_text())
    for event in all_events:
        if event['archive'] in excluded: groups[event['archive']].append(event)
    # Do not count an event again if it already belongs to an admitted full clip.
    old_archives={r['original'] for r in old['admitted']}
    used={(e['parent_utterance'],e['annotation_id']) for e in all_events if e['archive'] in old_archives}
    target_root=ROOT/'data/local/approved_phrases_v17_20260921_v2'
    target_manifest=ROOT/'active/v17/approved_phrase_manifest_20260921_v2.json'
    report=ROOT/'artifacts/reports/verified_phrase_salvage_v17_20260921'
    if target_root.exists() or target_manifest.exists(): raise FileExistsError('salvage version already exists')
    report.mkdir(parents=True,exist_ok=True)
    classes=json.loads((ROOT/'active/v17/citizen100_manifest.json').read_text())['classes']
    codes={r['citizen_asl_lex_code']:r for r in classes}
    labels={r['canonical_label']:r['class_index'] for r in classes}
    config=landmark_config(); detector=AppleVisionDetector(); decisions=[]; additions=[]
    for archive,rows in sorted(groups.items()):
        for seq in trusted_runs(rows):
            keys={(e['parent_utterance'],e['annotation_id']) for e in seq}
            item=dict(original=archive,annotation_ids=[e['annotation_id'] for e in seq],
                      variants=[e['variant'] for e in seq],role=seq[0]['role'],admitted=False)
            decisions.append(item)
            if keys & used:
                item['reason']='annotation already used';continue
            left,right=seq[0]['start_frame'],seq[-1]['end_frame_exclusive']
            targets=[codes[e['asllex_code']]['class_index'] if e['status']=='known' else 100 for e in seq]
            # Keep known repeats; consecutive OTHER identities share the existing one-span policy.
            targets=[t for i,t in enumerate(targets) if t!=100 or i==0 or targets[i-1]!=100]
            steps=len(rolling_windows(np.zeros((right-left,61,5),np.float32),4,8))
            if steps<len(targets)+sum(a==b for a,b in zip(targets,targets[1:])):
                item['reason']='CTC targets cannot fit stride4/window8 observations';continue
            with np.load(ROOT/excluded[archive]['recovered'],allow_pickle=False) as d:
                parent=json.loads(str(d['metadata_json'].item()))
            video=ROOT/parent['video_path']
            assert digest(video)==parent['video_sha256']
            frames,vm=read_video_frames(video,parent['sampled_source_frames'],1280,rotation='auto',input_mirrored=False)
            assert len(frames)==vm['decoded_frame_count']==parent['sampled_source_frames']
            crop=frames[left:right]
            if parent['vision_coarse_rotation_clockwise']:
                crop=[rotate_frame_clockwise(f,parent['vision_coarse_rotation_clockwise']) for f in crop]
            # Equal-size contiguous partitions avoid dropping short tails and keep windows <=32 frames.
            bounds=np.linspace(0,len(crop),(len(crop)+31)//32+1,dtype=int)
            ranges=np.stack((bounds[:-1],bounds[1:]),axis=1)
            results=[extract_frames_v17(crop[a:b],config,detector=detector) for a,b in ranges]
            if any(r is None or r.diagnostics['observed_hand_frame_fraction_before_trim']<.8 for r in results):
                item['reason']='insufficient observed hand coverage';continue
            features=np.asarray([r.features for r in results],dtype=np.float16)
            # Each annotated sign must retain actual hand observations, not just the whole crop.
            restored=restore_source_frames(features,ranges)
            if any(((restored[e['start_frame']-left:e['end_frame_exclusive']-left,:42,3]>0).any(axis=1)).mean()<.8 for e in seq):
                item['reason']='insufficient per-event hand coverage';continue
            identity='asllrp_verified_span:'+seq[0]['parent_utterance']+':'+','.join(item['annotation_ids'])
            metadata=dict(source='asllrp_verified_span',source_item_id=identity,role=parent['role'],
                          signer_id=parent['signer_id'],source_group=parent['source_group'],
                          video_path=parent['video_path'],video_sha256=parent['video_sha256'],
                          target_sequence=[next(k for k,v in labels.items() if v==t) if t!=100 else '__OTHER__' for t in targets],
                          schema=schema_payload(config),schema_fingerprint=schema_fingerprint(config),
                          sampled_source_frames=len(crop),dropped_tail_frames=0,window_count=len(ranges),
                          window_diagnostics=[r.diagnostics for r in results],
                          parent_archive=archive,parent_archive_sha256=digest(ROOT/archive),
                          parent_source_range=[left,right],annotation_ids=item['annotation_ids'],
                          raw_variants=item['variants'],asllex_codes=[e['asllex_code'] for e in seq],
                          temporal_policy='continuous original frames; adjoining complete annotations; no gaps, joining, retiming or label guesses',
                          vision_coarse_rotation_clockwise=parent['vision_coarse_rotation_clockwise'])
            validate_phrase_archive(features,ranges,np.array(targets),metadata,labels)
            path=target_root/'other'/parent['role']/'asllrp_verified_span'/('_'.join(item['annotation_ids'])+'.npz')
            save_archive(path,dict(landmarks=features,window_source_ranges=ranges,target_indices=np.array(targets)),metadata)
            additions.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path),original=archive,
                                  recovered=str(path.relative_to(ROOT)),source=metadata['source'],role=parent['role'],
                                  signer=parent['signer_id'],identity=identity,video_sha256=parent['video_sha256'],
                                  target_indices=targets,parent_source_range=[left,right]))
            used.update(keys);item.update(admitted=True,path=str(path.relative_to(ROOT)),frames=len(crop),steps=steps)
    if not additions:
        (report/'decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
        print('No candidates passed; canonical dataset unchanged.');return
    admitted=[]
    for key,name in [('phrase_root','phrases'),('other_root','other')]:
        for role in ('train','validation'):(target_root/name/role).mkdir(parents=True,exist_ok=True)
        for row in old['admitted']:
            original=ROOT/row['path']; source_root=ROOT/old['roots'][key]
            if not original.is_relative_to(source_root):continue
            output=target_root/name/original.relative_to(source_root)
            output.parent.mkdir(parents=True,exist_ok=True);output.symlink_to(original)
            admitted.append({**row,'path':str(output.relative_to(ROOT))})
    admitted+=additions
    train_signers={r['signer'] for r in admitted if r['role']=='train'}
    val_signers={r['signer'] for r in admitted if r['role']=='validation'}
    assert not train_signers & val_signers
    assert not {r['video_sha256'] for r in admitted if r['role']=='train'} & {r['video_sha256'] for r in admitted if r['role']=='validation'}
    manifest={**old,'version':2,'admitted':admitted,
              'roots':{k:str((target_root/v).relative_to(ROOT)) for k,v in [('phrase_root','phrases'),('other_root','other')]},
              'counts':dict(Counter(r['source']+':'+r['role'] for r in admitted)),
              'evidence_sha256':{**old['evidence_sha256'],str(old_path.relative_to(ROOT)):digest(old_path),
                                 str(Path(__file__).resolve().relative_to(ROOT)):digest(Path(__file__))},
              'salvage_policy':'Gap-free complete established identities only; raw-video re-extraction; original full clips remain excluded.'}
    target_manifest.write_text(json.dumps(manifest,indent=2)+'\n')
    summary=dict(candidates=len(decisions),added=len(additions),total=len(admitted),decisions=decisions,
                 verification=verify_manifest(target_manifest),training_started=False)
    (report/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
