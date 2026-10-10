from pathlib import Path
import json,subprocess,os
p=Path(__file__).resolve().parent
(p/'one_video.json').write_text(json.dumps([{'id':'take03','phase':0,'path':str(p/'device_inputs/take03.mp4')}]))
for name in ['candidate_paced','shared_candidate_paced']:
 d=json.loads((p/(name+'.json')).read_text());times=p/(name+'_times.json');times.write_text(json.dumps(d['processed_seconds']))
 output=p/(name+'_schedule_mac.json')
 with (p/(name+'_schedule_mac.log')).open('w') as f:
  subprocess.run([str(p/'phase_mac_probe'),str(p.parent/'my_like_regression_v17_20261011/resources'),str(p/'one_video.json'),str(output)],env=dict(os.environ,FINISH='1',LIVE_TIMES=str(times)),stdout=f,stderr=subprocess.STDOUT,check=True)
 r=json.loads(output.read_text())['records'][0]
 print(name,[(w['gloss'],round(w['start_seconds'],3)) for w in r['words'] if 8.8<w['start_seconds']<9.5],r['spell_events'][:2],flush=True)
