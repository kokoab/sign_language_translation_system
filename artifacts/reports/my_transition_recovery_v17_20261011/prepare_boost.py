from pathlib import Path
import shutil,subprocess,json
D=Path(__file__).resolve().parent;d=D/'boost_native';shutil.copytree(D/'rescore_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('    static var expert:', '''    static var kind = "swap", eligibility = "like"
    static var floor = 0.0
    static var expert:''');p.write_text(s)
p=d/'main.swift';s=p.read_text().replace('        Experiment.events = []','''        Experiment.events = []
        Experiment.kind = item["kind"] as? String ?? "swap"
        Experiment.eligibility = item["eligibility"] as? String ?? "like"
        Experiment.floor = item["floor"] as? Double ?? 0.0''');p.write_text(s)
p=d/'LiveReelDecoder.swift';s=p.read_text().replace('guard order.first==like && order.prefix(3).contains(my) else { continue }','guard (Experiment.eligibility == "like" ? order.first==like : order.prefix(3).contains(like) || order.prefix(3).contains(my)) && order.first != my else { continue }',1)
s=s.replace('                    if accept {logits[i].swapAt(my,like)}','''                    let chosen = Experiment.policy == "current" ? current[j] : old[j]
                    let lp = LiveSegmental.logSoftmax(Array(chosen.prefix(wordCount)))
                    if accept && exp(lp[my]) >= Experiment.floor {
                        if Experiment.kind == "swap" { logits[i].swapAt(my,like) }
                        else if Experiment.kind == "row" { logits[i]=Array(chosen.prefix(wordCount))+Array(logits[i].dropFirst(wordCount)) }
                        else {
                            var original = LiveSegmental.logSoftmax(Array(logits[i].prefix(wordCount))).map(Float.init)
                            original[my]=Float(lp[my]);original[like]=Float(lp[like])
                            logits[i]=original+Array(logits[i].dropFirst(wordCount))
                        }
                    }''');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
with (D/'boost_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/'boost_probe')]+files+[str(d/'main.swift')],stdout=f,stderr=subprocess.STDOUT,check=True)
N=D.parent/'sept30_fix_port_v17_20261011';P=D.parent/'temporal_repair_training_v17_20261011'
arms=[dict(name=f'{pol}_{kind}_{elig}_{pre}',policy=pol,kind=kind,eligibility=elig,alt_pre=pre,floor=.5) for pol,kind,elig,pre in [('old','pair','like',.1),('old','pair','like',.2),('old','pair','top3',.2),('old','row','top3',.2),('current','pair','top3',.1),('current','pair','top3',.2)]]
(D/'boost_arms.json').write_text(json.dumps(arms,indent=2))
base=json.loads((N/'new_video_cases.json').read_text())
cases=[dict(r,**{k:v for k,v in arm.items() if k!='name'},id=arm['name']+'__'+r['id']) for r in base if r['phase'] in [0,4] for arm in arms]
(D/'boost_screen_cases.json').write_text(json.dumps(cases,indent=2))
print('Built six probability rescoring arms')
