from pathlib import Path
import shutil,subprocess
D=Path(__file__).resolve().parent;d=D/'optimized_native';shutil.copytree(D/'preserved_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('_ observations:[LiveObservation]','_ features:[LiveMyLikeMotion.Row]').replace('LiveMyLikeMotion.features(observations).filter','features.filter').replace('plausibleMY(observations,start:start,end:end)','plausibleMY(features,start:start,end:end)');p.write_text(s)
p=d/'LiveReelDecoder.swift';s=p.read_text().replace('        var logits = batch.isEmpty','        let repairFeatures=LiveMyLikeMotion.features(obs)\n        var logits = batch.isEmpty').replace('Experiment.eligible(obs,start:','Experiment.eligible(repairFeatures,start:')
s=s.replace('        var segs = LiveSegmental.decode(sub, spans: localSpans, records: local, labels: labels, config: config)','        let completionFeatures=LiveMyLikeMotion.features(obs)\n        var segs = LiveSegmental.decode(sub, spans: localSpans, records: local, labels: labels, config: config)').replace('Experiment.plausibleMY(obs,start:','Experiment.plausibleMY(completionFeatures,start:').replace('            let motionRows=LiveMyLikeMotion.features(obs)','            let motionRows=completionFeatures').replace('                    let rows=LiveMyLikeMotion.features(obs)','                    let rows=completionFeatures');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
for kind,main in [('video',d/'main.swift'),('validation',D/'preserved_validation/main.swift')]:
 with (D/('optimized_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/('optimized_'+kind+'_probe'))]+files+[str(main)],stdout=f,stderr=subprocess.STDOUT,check=True)
print('Compiled equivalent geometry calculation once per scoring/decoding step')
