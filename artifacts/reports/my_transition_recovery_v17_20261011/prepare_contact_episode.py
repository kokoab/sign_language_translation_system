from pathlib import Path
import shutil,subprocess,json
D=Path(__file__).resolve().parent;d=D/'contact_native';shutil.copytree(D/'stable_boundary_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('    static var touchLimit=0.06','    static var touchLimit=0.06, maxContact=0.35')
s=s.replace('            return sqrt(dx*dx+dy*dy)<=touchLimit','''            guard sqrt(dx*dx+dy*dy)<=touchLimit else{return false}
            let px=rr.map(\\.px).reduce(0,+)/Double(rr.count),py=rr.map(\\.py).reduce(0,+)/Double(rr.count)
            let all=features.filter{$0.slot==slot}
            guard var lo=all.firstIndex(where:{$0.t==rr.first!.t}),var hi=all.firstIndex(where:{$0.t==rr.last!.t}) else{return false}
            func near(_ r:LiveMyLikeMotion.Row)->Bool {
                r.flatValid && r.straight>=0.97 && r.pinch>=0.8 && hypot(r.px-px,r.py-py)<=touchLimit
            }
            while lo>0 && all[lo].t-all[lo-1].t<=0.175001 && near(all[lo-1]) {lo-=1}
            while hi+1<all.count && all[hi+1].t-all[hi].t<=0.175001 && near(all[hi+1]) {hi+=1}
            return all[hi].t-all[lo].t<=maxContact+1e-6''');p.write_text(s)
p=d/'LiveReelDecoder.swift';s=p.read_text().replace(' && recognizer.labels[$0.1]==label','');p.write_text(s)
validation=D/'contact_validation';validation.mkdir(exist_ok=True);shutil.copy(D/'stable_boundary_validation/main.swift',validation/'main.swift')
for p in [d/'main.swift',validation/'main.swift']:
 s=p.read_text().replace('Experiment.touchLimit=item["touch_limit"] as? Double ?? 0.06','Experiment.touchLimit=item["touch_limit"] as? Double ?? 0.06\n        Experiment.maxContact=item["max_contact"] as? Double ?? 0.35');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
for kind,main in [('video',d/'main.swift'),('validation',validation/'main.swift')]:
 with (D/('contact_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/('contact_'+kind+'_probe'))]+files+[str(main)],stdout=f,stderr=subprocess.STDOUT,check=True)
rr=json.load(open(D/'narrow_final_all_cases.json'));ids=['offset0.01666667__new_phase0','offset0.01666667__new_phase5','offset0.03333333__new_phase3'];(D/'contact_screen_cases.json').write_text(json.dumps([dict(r,id=f'c{limit}__'+r['id'],max_contact=limit,touch_limit=.06) for limit in [.35,.45] for r in rr if r['id'] in ids]))
rr=json.load(open(D/'boundary_negative_cases.json'));(D/'contact_negative_cases.json').write_text(json.dumps([dict(r,id=f'c{limit}__'+r['id'],max_contact=limit,touch_limit=.06) for limit in [.35,.45] for r in rr]))
print('Compiled complete short stationary contact episode, never a cropped tail of a longer held posture')
