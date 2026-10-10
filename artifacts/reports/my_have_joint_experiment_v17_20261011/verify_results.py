"""Verify final claims from saved experiment results; no models or protected test access."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
read=lambda n:json.loads((P/n).read_text())
for path,digest in read('source_manifest.json').items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
installed=read('native_validation_installed.json')['records'];joint=read('native_validation_hybrid_v2.json')['records']
assert len(installed)==len(joint)==570
for a,b in zip(installed,joint):
 assert (a['dataset'],a['id'],a['reference'])==(b['dataset'],b['id'],b['reference'])
 assert a['hypothesis']==b['hypothesis'],(a['id'],a['hypothesis'],b['hypothesis'])
assert len({r['reference'][0] for r in installed if r['dataset']=='citizen_val'})==100
v1=read('native_validation_control.json')['records'];v2=read('native_validation_ordered_v2.json')['records']
changes=[(a,b) for a,b in zip(v1,v2) if a['hypothesis']!=b['hypothesis']]
assert len(changes)==1 and changes[0][1]['id']=='9dc94ebe93dd51139d48'
assert changes[0][1]['hypothesis']==['PLEASE','HELP','I']
phone={r['id']:r for r in read('hybrid_v2_phone.json')['records']}
assert phone['first_miss']['hypothesis']==['HELLO','MY','NAME']
for name,n in [('take11',10),('take0955',9),('live1006',9)]:
 assert phone[name]['hypothesis']==read('current_'+name+'.json')['hypothesis']
 assert phone[name]['hypothesis'].count('HAVE')==n
for name in ['long_word_run','repeat_opening']:
 assert phone[name]['hypothesis']==read('current_'+name+'.json')['hypothesis']
py=read('hybrid_v2_python_coreml.json')['first_miss'];sw=phone['first_miss']
assert py['hypothesis']==sw['hypothesis']
for a,b in zip(py['words'],sw['words']):
 for field in ['start_frame','end_frame','commit_frame']:assert a[field]==b[field]
 assert abs(a['score']-b['score'])<1e-4
before=read('current_videos.json')['records'];after=read('hybrid_v2_videos.json')['records']
def near9(r):return [w['gloss'] for w in r['words'] if 9.05<w['start_seconds']<9.4]
assert near9(before[0])==['LIKE'] and near9(after[0])==['MY']
assert after[0]['spell_events'][:2]==read('rollback_both_videos.json')['records'][0]['spell_events'][:2]
for a,b in zip(read('ordered_ends_my_like_val.json')['records'],read('hybrid_v2_my_like_val.json')['records']):
 assert a['id']==b['id'] and a['hypothesis']==b['hypothesis']
assert len(read('hybrid_v2_my_like_val.json')['records'])==59
sweep=read('geometry_motion_sweep.json')['sweep']
assert len(sweep)==14400
assert not any(r['first_late'] and r['false_my']==0 for r in sweep)
assert read('export_dimension_check.json')==dict(n=570,mismatches=[])
print('PASS: production hashes unchanged; 570 paired native cases across all 100 classes unchanged; 59 MY/LIKE videos unchanged; two diagnosed MY cases recovered; three HAVE controls retained; HELP state regression fixed; exact Python/CoreML–Swift phone parity; 14,400 direct geometry rules screened.')
