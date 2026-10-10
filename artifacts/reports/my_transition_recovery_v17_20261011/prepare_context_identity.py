from pathlib import Path
import shutil,subprocess,json
D=Path(__file__).resolve().parent;d=D/'context_native';shutil.copytree(D/'distinct_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('    static var pairFloor','    static var successorFloor=0.85, successorGap=0.6\n    static var pairFloor');p.write_text(s)
p=d/'LiveReelDecoder.swift';s=p.read_text().replace('    private var repairScores:','    private var lastMyEnd = -Double.infinity\n    private var repairScores:').replace('scores = [:]; repairScores = [:]; origin = 0;','scores = [:]; repairScores = [:]; lastMyEnd = -Double.infinity; origin = 0;')
src=(D/'successor_identity_native/LiveReelDecoder.swift').read_text();a=src.index('        if mode != .letters && Experiment.successorFloor<1');b=src.index('        let horizon = final ? known : known - config.lag',a);insert=src[a:b]
insert=insert.replace('seconds(start)-$0<=0.400001','seconds(start)-$0<=Experiment.successorGap+1e-6')
insert=insert.replace('guard !["MY","LIKE"].contains(label),p[top]>=Experiment.successorFloor,label != old else{continue}', '''guard label != "LIKE",p[top]>=Experiment.successorFloor,label != old else{continue}
                if label=="MY" {
                    guard p[top]>=0.95,seconds(end)-seconds(start)<=0.350001,
                        myEnds.contains(where:{seconds(start)>$0 && seconds(start)-$0<=0.200001}) else{continue}
                }''')
s=s.replace('        let horizon = final ? known : known - config.lag',insert+'        let horizon = final ? known : known - config.lag').replace('                var gloss=originalGloss','                var gloss=originalGloss\n                if gloss=="MY" {lastMyEnd=max(lastMyEnd,seconds(end))}')
p.write_text(s)
validation=D/'context_validation';validation.mkdir(exist_ok=True);shutil.copy(D/'contact_validation/main.swift',validation/'main.swift')
for p in [d/'main.swift',validation/'main.swift']:
 s=p.read_text().replace('Experiment.maxContact=item["max_contact"] as? Double ?? 0.35','Experiment.maxContact=item["max_contact"] as? Double ?? 0.35\n        Experiment.successorFloor=item["successor_floor"] as? Double ?? 0.85\n        Experiment.successorGap=item["successor_gap"] as? Double ?? 0.6');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
for kind,main in [('video',d/'main.swift'),('validation',validation/'main.swift')]:
 with (D/('context_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/('context_'+kind+'_probe'))]+files+[str(main)],stdout=f,stderr=subprocess.STDOUT,check=True)
rr=json.load(open(D/'narrow_final_all_cases.json'));ids=['new_phase2','offset0.01666667__new_phase3','offset0.01666667__new_phase5','offset0.03333333__new_phase3'];(D/'context_screen_cases.json').write_text(json.dumps([dict(r,id=f'f{f}__'+r['id'],successor_floor=f,successor_gap=.6,touch_limit=.06,max_contact=.35) for f in [.75,.85,.95] for r in rr if r['id'] in ids]))
print('Compiled generic immediate-transition identity correction; no successor class whitelist')
