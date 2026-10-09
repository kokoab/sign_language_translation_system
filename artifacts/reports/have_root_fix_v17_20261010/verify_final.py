"""Verify saved final evidence without rerunning models or touching protected test data."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def read(name):return json.loads((HERE/name).read_text())
def hyp(row):return [w['gloss'] for w in row['words']]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
reg=read('contact2_regression_pair_val_tune_citizen_val.json')
summary={}
for ds,n in [('pair_val',103),('tune',89),('citizen_val',378)]:
 a,b=reg[ds+':baseline'],reg[ds+':closed_merge_odds']
 assert len(a['rows'])==len(b['rows'])==n
 assert [r['key'] for r in a['rows']]==[r['key'] for r in b['rows']]
 if ds!='tune':assert [hyp(r) for r in a['rows']]==[hyp(r) for r in b['rows']]
 summary[ds]={'baseline':a['summary'],'fixed':b['summary']}
assert summary['pair_val']['fixed']['false_have']==0
assert summary['citizen_val']['fixed']['false_have']==1
assert summary['tune']['baseline']['wer']==34/226
assert summary['tune']['fixed']['wer']==29/226
py=read('python_coreml_replays.json');replays={};score_delta=0
for name,experimental,count in [('take11','contact2',10),('live','contact2_live',9),('0955','contact2_0955',9)]:
 production=read(f'production_{name}.json')
 experiment=read(experimental+'.json')['results']['minimal']
 assert production['hypothesis']==experiment['hypothesis']==py[name]['hypothesis']
 assert production['hypothesis'].count('HAVE')==count
 for a,b,c in zip(production['words'],experiment['words'],py[name]['words']):
  for key in ['start_frame','end_frame','commit_frame']:
   assert a[key]==b[key]==c[key],(name,key)
  assert abs(a['score']-b['score'])<1e-6
  score_delta=max(score_delta,abs(a['score']-c['score']))
 assert score_delta<2e-5
 replays[name]={'frames':production['frames'],'hypothesis':production['hypothesis']}
words=read('production_take11.json')['words']
for lo,hi in [(3.9,4.6),(6.8,7.5),(11.3,12.1),(20.6,21.5),(28.2,29.1)]:
 found=[i for i,w in enumerate(words) if w['gloss']=='HAVE' and lo<=w['start_seconds']<=hi]
 assert len(found)==1,(lo,hi,found)
 assert words[found[0]+1]['gloss']=='BIG'
assert sum(w['gloss']=='ANGRY' for w in words)==1
old_take=read('baseline_scores.json')['results']['baseline']['hypothesis']
old_0955=read('baseline_0955.json')['results']['baseline']['hypothesis']
assert old_take.count('HAVE')==5 and old_0955.count('HAVE')==8
parity=read('contact_parity_summary.json');assert parity['cases']==5 and parity['max_probability_delta']==0
prod=read('production_parity_pair_val_takes2.json')
for ds,n in [('pair_val',6),('takes2',1)]:
 a,b=prod[ds+':experiment']['rows'],prod[ds+':production']['rows']
 assert len(a)==len(b)==n
 assert [hyp(r) for r in a]==[hyp(r) for r in b]
assert '** BUILD SUCCEEDED **' in (HERE/'ios_build.log').read_text()
assert 'Ran 23 tests' in (HERE/'unit_tests.log').read_text() and '\nOK\n' in (HERE/'unit_tests.log').read_text()
old_hashes=json.loads((ROOT/'artifacts/reports/have_omission_v17_20261010/source_hashes.json').read_text())
for path,sha in old_hashes.items():
 if not path.endswith('LiveReelDecoder.swift'):assert digest(path)==sha,path
files=[ROOT/'active/v17/segmental_runtime_v17.py',ROOT/'test/test_segmental_runtime_v17.py',
       Path('/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios/Runner/LiveReel/LiveReelDecoder.swift'),HERE/'config_fixed.json']
deployment=read('installation.json') if (HERE/'installation.json').exists() else {}
if deployment.get('installed'):
 assert read('ios_install.json')['info']['outcome']=='success'
 assert read('ios_installed_app.json')['result']['apps'][0]['url']==deployment['installation_url']
 assert '** BUILD SUCCEEDED **' in (HERE/'ios_signed_build.log').read_text()
out={'development':summary,'replays':replays,'python_swift_max_score_delta':score_delta,
     'contact_fixture':parity,'source_sha256':{str(p):digest(p) for p in files},
     'remaining':'First take11 HAVE is still ANGRY; future live reliability is not guaranteed.',
     'installed':deployment.get('installed',False),'ios_release_build':deployment.get('release_build','succeeded_unsigned'),'deployment':deployment}
(HERE/'final_results.json').write_text(json.dumps(out,indent=2))
print('PASS: 570 paired development clips, three exact-input replays, five omission recoveries, production/experimental parity, source preservation, 23 tests and iOS build.')
