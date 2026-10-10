from pathlib import Path
import json,hashlib,shutil,subprocess
O=Path(__file__).resolve().parent;R=O.parents[2];P=O.parent/'temporal_repair_contract_v17_20261011'
d=O/'native';d.mkdir(exist_ok=True)
for f in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift']:shutil.copy2(P/'native'/f,d/f)
s=(P/'native/main.swift').read_text()
old='''     observations.append(observation);hands.append(try encoder.frameHand(observation,image:image))'''
new='''     // Vision/tracker sees the entire prefix. The stateless crop encoder is needed only inside a requested context.
     let needed=(item["events"] as! [[String:Any]]).contains { event in
      let core=event["core"] as! [Double]
      return seconds >= core[0]-0.2-1e-9 && seconds < core[1]+0.1-1e-9
     }
     if needed { observations.append(observation);hands.append(try encoder.frameHand(observation,image:image)) }'''
assert old in s;s=s.replace(old,new)
(d/'main.swift').write_text(s)
with (O/'native_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(O/'native_probe')]+[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','main.swift']],stdout=f,stderr=subprocess.STDOUT,check=True)
contract=json.loads((P/'contract.json').read_text());groups={}
for r in contract['records']:
 key=r['source_video_sha256'];c=groups.setdefault(key,dict(id=key,path=str(R/r['source_video_path']),limit=0,events=[]));c['events'].append(dict(id=r['id'],core=r['core_seconds']));c['limit']=max(c['limit'],r['core_seconds'][1]+.2)
# Process shorter parents first; group multiple events from one narrative without duplicating prefix work.
cases=sorted(groups.values(),key=lambda c:c['limit']);(O/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
(O/'parity_case.json').write_text(json.dumps(json.loads((P/'pilot_cases.json').read_text())[:1],indent=2)+'\n')
print('Built',len(cases),'parent cases,',sum(len(c['events']) for c in cases),'events; summed source prefix seconds',sum(c['limit'] for c in cases))
