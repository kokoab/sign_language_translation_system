"""Small contract, interval pooling and completed-result checks."""
import runpy
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
m = runpy.run_path(str(HERE / 'run.py'))
values = np.arange(8, dtype=float)[:, None]
np.testing.assert_array_equal(m['pool'](values, np.arange(8)/10, .2, .5), [2, 3, 4, 5])
try:
    m['pool'](values, np.arange(8)/10, .2, .3)
except ValueError:
    pass
else:
    raise AssertionError('too-short interval accepted')
rows = [dict(video=str(i), role='train') for i in range(10)] + [dict(video='held', role='validation')]
fit, cal, val = m['split'](rows)
assert len(fit)==8 and len(cal)==2 and len(val)==1
assert not {r['video'] for r in fit} & {r['video'] for r in cal+val}
# MP4 crop encoding rounds 30000/1001 FPS to 29.97. Interval endpoints
# must still use the original rational source clock, not the new header.
clock=m['original_clock']('data/local/asllrp_contiguous_phrases_v17/spans/train_candidate/asllrp/FAMILY_IMPORTANT/7028838_span00.mp4',34)
assert clock[12] == 0.40040000000000003
m['verify_completed']()
print('Pooling, split isolation and completed-result checks passed')

# Audit source/crop alignment independently of the extraction implementation.
if (HERE/'results.json').exists():
    import cv2
    import hashlib
    import json
    evidence=json.loads((HERE.parent/'reel_decision_probe_v17_20260922/evidence.json').read_text())['results']
    aggregate={}
    timing=[]
    for video in sorted({r['video'] for r in evidence}):
        cache=HERE/'cache'/hashlib.sha256(video.encode()).hexdigest()[:16]
        info=json.loads((cache/'smoke.json').read_text())
        source=cv2.VideoCapture(str(m['ROOT']/video))
        source_fps=source.get(cv2.CAP_PROP_FPS)
        count=0
        while source.read()[0]:
            count+=1
        source.release()
        assert count==info['frames'], (video,count,info['frames'])
        assert abs(source_fps-info['fps'])<.001, (video,source_fps,info['fps'])
        indices=[]
        for fps in [source_fps,info['fps']]:
            deadline=0.;keep=[]
            for i in range(count):
                t=i/fps
                if t+1e-6>=deadline:
                    keep.append(i);deadline=max(deadline+1/20,t)
            indices.append(np.array(keep))
        np.testing.assert_array_equal(*indices)
        for r in [r for r in evidence if r['video']==video]:
            selections=[]
            for fps,keep in zip([source_fps,info['fps']],indices):
                clock=keep/fps
                selections.append(keep[(clock>=r['start']) & (clock<=r['end'])])
            corrected_clock=m['original_clock'](video,count)[indices[0]]
            corrected=indices[0][(corrected_clock>=r['start']) & (corrected_clock<=r['end'])]
            np.testing.assert_array_equal(selections[0],corrected)
            assert len(corrected)==r['proposal']['frames'], (r['id'],r['kind'],len(corrected),r['proposal']['frames'])
        role=next(r['role'] for r in evidence if r['video']==video)
        entry=aggregate.setdefault(role,dict(videos=0,frames=0,face=0,left_hand=0,right_hand=0,pose=0,zero_face_videos=[]))
        entry['videos']+=1;entry['frames']+=count
        for key,value in info['presence'].items():entry[key]+=value
        if info['presence']['face']==0:entry['zero_face_videos'].append(video)
        timing.append(json.loads((cache/'complete.json').read_text())['elapsed_seconds'])
    results=json.loads((HERE/'results.json').read_text())
    for arm in results['arms'].values():
        # A completed experiment may reject no gaps; it must not invent new commits.
        assert arm['validation']['gap']['conditional_commits']<=results['arms']['unchanged']['validation']['gap']['conditional_commits']
    audit=dict(passed=True,source_crop_frame_counts_equal_and_corrected_intervals_match_prior=True,
               detector_coverage=aggregate,extraction_seconds_total=sum(timing),
               extraction_seconds_median=float(np.median(timing)),completed_videos=len(timing),
               expected_videos=len({r['video'] for r in evidence}),
               failures='No skipped sources: any extraction failure aborts; all expected cache entries verified')
    (HERE/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))
