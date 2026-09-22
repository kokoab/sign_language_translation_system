"""Validate geometry, exact held-out membership, and reported conditional veto counts."""
import json,hashlib
from pathlib import Path
import numpy as np
from transfer import letterbox,ROOT,OUT

frame=np.ones((100,200,3),np.uint8)*255
points=np.array([[0.,0.,1.],[200.,100.,1.]],np.float32)
image,mapped=letterbox(frame,points)
assert image.shape==(224,224,3)
np.testing.assert_allclose(mapped[:,:2],[[0,56],[224,168]],atol=1e-5)
assert not image[:56].any() and image[56:168].all()

if (OUT/'transfer_results.json').exists():
    result=json.loads((OUT/'transfer_results.json').read_text())
    evidence=json.loads((ROOT/'artifacts/reports/reel_decision_probe_v17_20260922/evidence.json').read_text())['results']
    expected={(r['id'],r['kind']):r for r in evidence if r['role']=='validation' and r['kind'] in ['core','gap']}
    rows=result['rows'];assert len(rows)==len(expected)==20
    assert len({(r['id'],r['kind']) for r in rows})==20
    for r in rows:
        old=expected[r['id'],r['kind']]
        assert r['video']==old['video'] and r['baseline_commit']==old['conditional_commit']
        assert r['baseline_correct']==bool(old['conditional_commit'] and old['target']==old['verifier']['candidate_gloss'])
        assert r['pass_gate']==(r['checkpoint_top_gloss']!='<blank>')
        assert 0<=r['blank_probability']<=1 and len(r['source_indices'])==16
        assert r['source_indices']==sorted(r['source_indices'])
    gaps=[r for r in rows if r['kind']=='gap'];cores=[r for r in rows if r['kind']=='core'];s=result['summary']
    assert len(gaps)==3 and len(cores)==17
    assert s['gaps_rejected']==sum(not r['pass_gate'] for r in gaps)
    assert s['false_gap_commits']==sum(r['baseline_commit'] and r['pass_gate'] for r in gaps)
    assert s['correct_core_commits']==sum(r['baseline_correct'] and r['pass_gate'] for r in cores)
    assert s['false_gap_commits']<=s['baseline_false_gap_commits']==1
    assert s['correct_core_commits']<=s['baseline_correct_core_commits']==5
    assert result['recipe_sha256']==hashlib.sha256((OUT/'transfer_recipe.json').read_bytes()).hexdigest()
    print('All20paired rows, blank decisions and conditional counts verified')
print('Aspect-preserving geometry check passed')
