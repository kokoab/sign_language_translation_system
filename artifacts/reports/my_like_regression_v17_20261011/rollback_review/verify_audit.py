"""Verify measured audit claims from saved outputs. No model/test-set access."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
R=P.parents[1]/'have_root_fix_v17_20261010'
def read(n):return json.loads((P/n).read_text())
for clip in ['first_miss','long_word_run','repeat_opening']:
 d=read('settings_'+clip+'.json')
 assert len(d)==9
 assert d['current']['hypothesis']==d['no_have_changes']['hypothesis']
 if clip=='first_miss':assert all(r['hypothesis']==['HELLO','LIKE','NAME'] for r in d.values())
old=read('video_old_engine_current_model.json')['records']
off=read('video_order_off.json')['records'];current=read('video_current.json')['records']
scoped=read('video_order_have.json')['records']
for a,b,c in zip(old,off,scoped):
 assert a['hypothesis']==b['hypothesis']==c['hypothesis']
 assert a['spell_events']==b['spell_events']==c['spell_events']
def at(record,t):return next(w for w in record['words'] if abs(w['start_seconds']-t)<.1)
assert at(old[0],9.2)['gloss']=='MY'
assert at(current[0],9.2)['gloss']=='LIKE'
assert old[1]['hypothesis']==current[1]['hypothesis']
for take in ['take11','0955']:
 a=read('order_have_'+take+'.json');b=json.loads((R/('production_'+take+'.json')).read_text())
 assert a['hypothesis']==b['hypothesis']
base=read('base_comparison.json')
assert len(base)==7
for r in base.values():
 assert sum(x['n'] for x in r['validation'].values())==3959
 assert len(r['phone'])==12
assert sum(x['correct'] for x in base['current']['validation'].values())==3784
assert sum(x['correct'] for x in base['old_phone']['validation'].values())==3743
print('PASS: 27 settings replays, full-engine ordering isolation, two HAVE controls, seven model comparisons.')
