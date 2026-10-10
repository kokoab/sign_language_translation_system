from pathlib import Path
import shutil,subprocess,json
D=Path(__file__).resolve().parent;d=D/'successor_identity_native';shutil.copytree(D/'narrow_boundary_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('    static var pairFloor','    static var successorFloor=0.95\n    static var pairFloor');p.write_text(s)
p=d/'LiveReelDecoder.swift';s=p.read_text().replace('    private var repairScores:','    private var lastMyEnd = -Double.infinity\n    private var repairScores:').replace('scores = [:]; repairScores = [:]; origin = 0;','scores = [:]; repairScores = [:]; lastMyEnd = -Double.infinity; origin = 0;')
needle='        let horizon = final ? known : known - config.lag'
insert='''        if mode != .letters && Experiment.successorFloor<1 {
            let myEnds=segs.filter{$0.gloss=="MY"}.map{seconds($0.end+origin)}+[lastMyEnd]
            for index in segs.indices {
                let g=segs[index],start=g.start+origin,end=g.end+origin
                guard let old=g.gloss,!["MY","LIKE"].contains(old),!old.hasPrefix("FS_"),
                    myEnds.contains(where:{seconds(start)>$0 && seconds(start)-$0<=0.400001}),
                    let adapted=repairScores[SpanKey(start,end)] else{continue}
                let p=LiveSegmental.logSoftmax(adapted).map{exp($0)},top=adapted.indices.max(by:{adapted[$0]<adapted[$1]})!
                let label=recognizer.labels[top]
                guard !["MY","LIKE"].contains(label),p[top]>=Experiment.successorFloor,label != old else{continue}
                segs[index].gloss=label
                if Experiment.audit {Experiment.events.append(["successor_identity":true,"s":seconds(start),"e":seconds(end),"original":old,"adapted":label,"probability":p[top],"available":seconds(T)])}
            }
        }
'''
s=s.replace(needle,insert+needle).replace('                var gloss=originalGloss','                var gloss=originalGloss\n                if gloss=="MY" {lastMyEnd=max(lastMyEnd,seconds(end))}')
p.write_text(s)
validation=D/'successor_identity_validation';validation.mkdir(exist_ok=True);shutil.copy(D/'narrow_boundary_validation/main.swift',validation/'main.swift')
for p in [d/'main.swift',validation/'main.swift']:
 s=p.read_text().replace('Experiment.onsetFloor=Float(item["onset_floor"] as? Double ?? 0.4)','Experiment.onsetFloor=Float(item["onset_floor"] as? Double ?? 0.4)\n        Experiment.successorFloor=item["successor_floor"] as? Double ?? 0.95');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
for kind,main in [('video',d/'main.swift'),('validation',validation/'main.swift')]:
 with (D/('successor_identity_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/('successor_identity_'+kind+'_probe'))]+files+[str(main)],stdout=f,stderr=subprocess.STDOUT,check=True)
rr=json.load(open(D/'narrow_final_all_cases.json'));ids=['new_phase1','new_phase2','offset0.01666667__new_phase3','offset0.01666667__new_phase5'];(D/'successor_identity_screen_cases.json').write_text(json.dumps([dict(r,successor_floor=.95) for r in rr if r['id'] in ids]))
print('Compiled confidence-gated identity-only correction for the immediate successor of MY')
