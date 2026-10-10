from pathlib import Path
import shutil,subprocess,json
D=Path(__file__).resolve().parent;d=D/'ambiguous_native';shutil.copytree(D/'calibrated_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('static var strong=0.9','static var strong=0.6').replace('        if p[my]<strong {','        if top==my && !q.indices.sorted(by:{q[$0]>q[$1]}).prefix(3).contains(like) {return nil}\n        if p[my]<strong {');p.write_text(s)
validation=D/'ambiguous_validation';validation.mkdir(exist_ok=True);shutil.copy(D/'calibrated_validation/main.swift',validation/'main.swift')
for p in [d/'main.swift',validation/'main.swift']:
 s=p.read_text().replace('as? Double ?? 0.9','as? Double ?? 0.6');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
for kind,main in [('video',d/'main.swift'),('validation',validation/'main.swift')]:
 with (D/('ambiguous_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/('ambiguous_'+kind+'_probe'))]+files+[str(main)],stdout=f,stderr=subprocess.STDOUT,check=True)
for kind in ['new','genuine','validation','phase','splice','drink']:
 base=json.load(open(D/f'calibrated_{kind}_cases.json'));(D/f'ambiguous_{kind}_cases.json').write_text(json.dumps([dict(r,strong=.6,floor=.5) for r in base]))
print('Compiled ambiguity-scoped confidence experiment')
