from pathlib import Path
import json, shutil, subprocess, hashlib
D=Path(__file__).resolve().parent;G=D.parent/'ordered_my_like_v17_20261011/gap175';N=D.parent/'sept30_fix_port_v17_20261011'
d=D/'native';shutil.copytree(G/'native',d,dirs_exist_ok=True)
p=d/'LiveReelDecoder.swift';s=p.read_text();s=s.replace('let lo = obs[s - off].seconds - 0.1, hi = obs[e - off].seconds + 0.1','let lo = obs[s - off].seconds - Experiment.pre, hi = obs[e - off].seconds + Experiment.post').replace('s - off - 3), to = min(obs.count, e - off + 4)','s - off - 5), to = min(obs.count, e - off + 5)')
s=s.replace('        for (key, row) in zip(keys, logits) { shared.values[key] = .some(row) }','''        if Experiment.audit {
            for (key, row) in zip(keys, logits) {
                let word = Array(row.prefix(recognizer.labels.count))
                let lp = LiveSegmental.logSoftmax(word)
                let top = lp.indices.sorted { lp[$0] > lp[$1] }.prefix(5)
                if top.contains(where: { ["MY","LIKE"].contains(recognizer.labels[$0]) }) {
                    Experiment.events.append(["s":seconds(key.s),"e":seconds(key.e),"available":obs.last!.seconds,"top":top.map { recognizer.labels[$0] },"p":top.map { exp(lp[$0]) }])
                }
            }
        }
        for (key, row) in zip(keys, logits) { shared.values[key] = .some(row) }''');p.write_text(s)
p=d/'Experiment.swift';p.write_text('''import Foundation
// Report-local ablation controls. No production source imports this file.
enum Experiment {
    static var pre = 0.1, post = 0.1
    static var audit = true
    static var events: [[String:Any]] = []
}
''')
p=d/'main.swift';s=p.read_text().replace('        engine.reset()','''        engine.reset()
        Experiment.pre = item["pre"] as? Double ?? 0.1
        Experiment.post = item["post"] as? Double ?? 0.1
        Experiment.events = []''').replace('"motion_events": LiveMyLikeMotion.events,','"score_events": Experiment.events, "motion_events": LiveMyLikeMotion.events,');s=s.replace('        fflush(stdout)','        fflush(stdout)\n        try writeJSON(["records": records], out)');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
with (D/'build.log').open('w') as f: subprocess.run(['xcrun','swiftc','-O','-o',str(D/'video_probe')]+files+[str(d/'main.swift')],stdout=f,stderr=subprocess.STDOUT,check=True)
arms=[('baseline',.1,.1),('no_future',.1,0),('core_only',0,0),('no_past',0,.1),('past150',.15,0),('past200',.2,0),('future50',.1,.05)]
base=json.loads((N/'new_video_cases.json').read_text())
cases=[dict(r,id=name+'__'+r['id'],pre=pre,post=post) for r in base if r['phase'] in [0,4] for name,pre,post in arms]
(D/'context_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
(D/'arms.json').write_text(json.dumps(arms,indent=2)+'\n')
pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in d.glob('*.swift')}
(D/'source_pins.json').write_text(json.dumps(pins,indent=2)+'\n')
print('Built',len(cases),'screening cases per model')
