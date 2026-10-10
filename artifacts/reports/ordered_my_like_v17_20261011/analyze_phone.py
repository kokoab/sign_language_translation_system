from pathlib import Path
import json,statistics
O=Path(__file__).resolve().parent
rows=[]
for i,arm in enumerate(['off','on','on','off']):
 p=O/f'phone_{arm}_{i}.json';d=json.loads(p.read_text())
 assert d['arm']==arm and d['pass']==i and d['stage3_ready']
 w=[w for w in d['words'] if min(2.9333333333,w['end_seconds'])-max(2.8,w['start_seconds'])>1e-5]
 ms=sorted(1000*x for x in d['compute_seconds']);ts=d['processed_seconds']
 rs={'arm':arm,'pass':i,'hypothesis':d['hypothesis'],'first_my_core':[x['gloss'] for x in w],
     'first_my_words':w,'processed_frames':len(ts),'median_compute_ms':statistics.median(ms),'p90_compute_ms':ms[int(.9*len(ms))],
     'max_compute_ms':max(ms),'over50ms':sum(x>50 for x in ms),'over100ms':sum(x>100 for x in ms),
     'max_frame_gap_ms':1000*max(b-a for a,b in zip(ts,ts[1:])),
     'ready_mb':d['ready_mb'],'peak_sampled_mb':d['peak_sampled_mb'],'thermal_before':d['thermal_before'],'thermal_after':d['thermal_after'],
     'stage3_ready':d['stage3_ready'],'finish_held':d['finish_held'],'wall_seconds':d['wall_seconds'],
     'rescues':sum(e.get('stage')=='span' for e in d['motion_events'])}
 idx=[i for i,x in enumerate(d['words']) if min(2.9333333333,x['end_seconds'])-max(2.8,x['start_seconds'])>1e-5]
 rs['first_my_then_name']=len(idx)==1 and w[0]['gloss']=='MY' and idx[0]+1<len(d['words']) and d['words'][idx[0]+1]['gloss']=='NAME'
 rs['first_my_commit_delay_ms']=[1000*(x['commit_seconds']-x['end_seconds']) for x in w]
 rs['first_region_motion_events']=[{k:v for k,v in e.items() if k!='rows'} for e in d['motion_events'] if e.get('start',100)<3.2 and e.get('end',0)>2.7]
 rows.append(rs)
result={'scope':'Four real-iPhone paced replays of the same known video; ordinary shared engine with resident English, overlay and latest-frame gate. Not fresh signing or independent accuracy. Scheduling is allowed to differ.','records':rows,'first_my_correct_by_arm':{a:sum(r['first_my_core']==['MY'] for r in rows if r['arm']==a) for a in ['off','on']}}
(O/'phone_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
