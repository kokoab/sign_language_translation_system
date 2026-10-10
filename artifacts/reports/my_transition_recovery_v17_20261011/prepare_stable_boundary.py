from pathlib import Path
import shutil,subprocess,json
D=Path(__file__).resolve().parent;d=D/'stable_boundary_native';shutil.copytree(D/'narrow_boundary_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('    static var pairFloor','''    static var touchLimit=0.06
    static func stableTouch(_ features:[LiveMyLikeMotion.Row],start:Double,end:Double)->Bool {
        let rows=features.filter{$0.t>=start-0.05 && $0.t<=end && $0.flatValid && $0.straight>=0.97 && $0.pinch>=0.8 && abs($0.px)<=0.6 && $0.py>=(-0.15) && $0.py<=1.0}
        return (0..<2).contains {slot in
            let rr=rows.filter{$0.slot==slot}
            guard rr.count>=3,rr.last!.t-rr.first!.t>=0.08 else{return false}
            let dx=rr.map(\\.px).max()!-rr.map(\\.px).min()!,dy=rr.map(\\.py).max()!-rr.map(\\.py).min()!
            return sqrt(dx*dx+dy*dy)<=touchLimit
        }
    }
    static var pairFloor''');p.write_text(s)
p=d/'LiveReelDecoder.swift';s=p.read_text().replace('                    Experiment.plausibleMY(completionFeatures,start:seconds(span.s),end:seconds(span.e)),','                    Experiment.stableTouch(completionFeatures,start:seconds(span.s),end:seconds(span.e)),');p.write_text(s)
validation=D/'stable_boundary_validation';validation.mkdir(exist_ok=True);shutil.copy(D/'narrow_boundary_validation/main.swift',validation/'main.swift')
for p in [d/'main.swift',validation/'main.swift']:
 s=p.read_text().replace('Experiment.onsetFloor=Float(item["onset_floor"] as? Double ?? 0.4)','Experiment.onsetFloor=Float(item["onset_floor"] as? Double ?? 0.4)\n        Experiment.touchLimit=item["touch_limit"] as? Double ?? 0.06');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
for kind,main in [('video',d/'main.swift'),('validation',validation/'main.swift')]:
 with (D/('stable_boundary_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/('stable_boundary_'+kind+'_probe'))]+files+[str(main)],stdout=f,stderr=subprocess.STDOUT,check=True)
rr=json.load(open(D/'narrow_final_all_cases.json'));ids=['offset0.01666667__new_phase0','offset0.01666667__new_phase5'];(D/'stable_boundary_screen_cases.json').write_text(json.dumps([dict(r,id=f't{limit}__'+r['id'],touch_limit=limit) for limit in [.04,.06,.08] for r in rr if r['id'] in ids]))
rr=json.load(open(D/'boundary_negative_cases.json'));(D/'stable_boundary_negative_cases.json').write_text(json.dumps([dict(r,id=f't{limit}__'+r['id'],touch_limit=limit) for limit in [.04,.06,.08] for r in rr]))
print('Compiled positive stationary-flat-contact confirmation; missing landmarks cannot authorize inserting MY')
