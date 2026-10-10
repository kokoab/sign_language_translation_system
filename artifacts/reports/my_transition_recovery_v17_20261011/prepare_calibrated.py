from pathlib import Path
import shutil,subprocess,json
D=Path(__file__).resolve().parent;d=D/'calibrated_native';shutil.copytree(D/'rank_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('    static var policy="swap"','    static var strong=0.9\n    static var policy="calibrated"').replace('guard top != my, p.indices.max','guard p.indices.max').replace('        if policy == "swap" || (policy == "hybrid" && top == like) {','        if p[my]<strong {')
s=s.replace('        let target=min(p[my],0.999999),lp=','        guard p[my]>q[my] else{return nil}\n        let target=min(p[my],0.999999),lp=')
p.write_text(s)
validation=D/'calibrated_validation';validation.mkdir(exist_ok=True);shutil.copy(D/'rank_validation/main.swift',validation/'main.swift')
for p in [d/'main.swift',validation/'main.swift']:
 s=p.read_text().replace('Experiment.pairFloor=item["pair_floor"] as? Double ?? 0.0','Experiment.pairFloor=item["pair_floor"] as? Double ?? 0.0\n        Experiment.strong=item["strong"] as? Double ?? 0.9');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
for kind,main in [('video',d/'main.swift'),('validation',validation/'main.swift')]:
 with (D/('calibrated_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/('calibrated_'+kind+'_probe'))]+files+[str(main)],stdout=f,stderr=subprocess.STDOUT,check=True)
for kind in ['new','genuine','validation','phase','splice']:
 base=json.load(open(D/f'rank_swap_{kind}_cases.json'))
 (D/f'calibrated_{kind}_cases.json').write_text(json.dumps([dict(r,policy='calibrated',strong=.9,floor=.5) for r in base]))
base=json.load(open(D/'rank_drink_cases.json'))[:1];(D/'calibrated_drink_cases.json').write_text(json.dumps([dict(r,policy='calibrated',strong=.9,floor=.5) for r in base]))
print('Compiled confidence-calibrated correction')
