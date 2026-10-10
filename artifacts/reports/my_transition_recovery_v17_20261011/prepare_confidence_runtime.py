from pathlib import Path
import shutil,subprocess,json
D=Path(__file__).resolve().parent;d=D/'confidence_native';shutil.copytree(D/'optimized_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('    static var pairFloor','    static var agreementFloor=2.0\n    static var pairFloor').replace('p[my]>=0.9','p[my]>=agreementFloor');p.write_text(s)
validation=D/'confidence_validation';validation.mkdir(exist_ok=True);shutil.copy(D/'preserved_validation/main.swift',validation/'main.swift')
for p in [d/'main.swift',validation/'main.swift']:
 s=p.read_text().replace('Experiment.pairFloor=item["pair_floor"] as? Double ?? 0.0','Experiment.pairFloor=item["pair_floor"] as? Double ?? 0.0\n        Experiment.agreementFloor=item["agreement_floor"] as? Double ?? 2.0');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
for kind,main in [('video',d/'main.swift'),('validation',validation/'main.swift')]:
 with (D/('confidence_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/('confidence_'+kind+'_probe'))]+files+[str(main)],stdout=f,stderr=subprocess.STDOUT,check=True)
rr=json.load(open(D/'jitter_all_cases.json'));(D/'confidence_all_cases.json').write_text(json.dumps([dict(r,agreement_floor=2,floor=.5) for r in rr]))
print('Compiled confidence-model runner; no boosting unambiguous already-MY spans')
