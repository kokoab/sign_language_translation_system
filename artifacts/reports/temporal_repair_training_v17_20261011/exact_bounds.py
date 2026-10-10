"""Resolve reviewed source frame indices using integer PTS/time_base, not rounded seconds."""
from pathlib import Path
from fractions import Fraction
import json,subprocess,collections
O=Path(__file__).resolve().parent;R=O.parents[2];P=O.parent/'temporal_repair_contract_v17_20261011'
c=json.loads((P/'contract.json').read_text());groups={};changes=[]
for row in c['records']:
 if row['source']=='o5s5':continue
 p=row['source_video_path']
 if p not in groups:
  d=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=time_base:frame=best_effort_timestamp','-of','json',str(R/p)]));base=Fraction(d['streams'][0]['time_base']);ints=[int(f['best_effort_timestamp']) for f in d['frames']];groups[p]=[(n-ints[0])*base for n in ints]
 pts=groups[p]
 if row['label']=='MY':v=row['visual_review'];a,b=v['core_frame_start'],v['core_frame_end_inclusive']
 else:a,b=row['source_start_frame'],row['source_end_frame_inclusive']
 end=pts[b+1] if b+1<len(pts) else pts[b]+(pts[b]-pts[b-1]);core=[float(pts[a]),float(end)]
 changes.append({'id':row['id'],'old_core_seconds':row['core_seconds'],'exact_core_seconds':core,'frame_start':a,'frame_end_inclusive':b,'start_rational':str(pts[a]),'end_rational':str(end)})
 row['core_seconds']=core
(O/'exact_core_contract.json').write_text(json.dumps(c,indent=2)+'\n');(O/'timestamp_correction.json').write_text(json.dumps({'reason':'ffprobe decimal timestamps round to microseconds; comparisons at 1e-9 can drop first frame or admit exclusive end frame. Source frame annotations unchanged; exact rational PTS resolve the boundary.','records':changes},indent=2)+'\n')
cases={}
for r in c['records']:
 key=r['source_video_sha256'];v=cases.setdefault(key,dict(id=key,path=str(R/r['source_video_path']),limit=0,events=[],source=r['source']));v['events'].append(dict(id=r['id'],core=r['core_seconds']));v['limit']=max(v['limit'],r['core_seconds'][1]+.2)
(O/'exact_cases.json').write_text(json.dumps(sorted(cases.values(),key=lambda r:r['limit']),indent=2)+'\n');print('Exact frame bounds resolved',len(changes),'events; O5S5 annotated time intervals unchanged.')
