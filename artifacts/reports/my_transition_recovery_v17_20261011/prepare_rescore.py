from pathlib import Path
import shutil,subprocess,json
D=Path(__file__).resolve().parent;d=D/'rescore_native';shutil.copytree(D/'native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('    static var audit', '''    static var policy = "none"
    static var altPre = 0.1
    static var expert: LiveSpanRecognizer?
    static var audit''');p.write_text(s)
p=d/'main.swift';s=p.read_text().replace('        Experiment.events = []','''        Experiment.events = []
        Experiment.policy = item["policy"] as? String ?? "none"
        Experiment.altPre = item["alt_pre"] as? Double ?? 0.1''');p.write_text(s)
p=d/'LiveReelDecoder.swift';s=p.read_text();needle='        let wordCount = recognizer.labels.count';replace='''        let wordCount = recognizer.labels.count
        if mode != .letters, Experiment.policy != "none",
           let my=recognizer.labels.firstIndex(of:"MY"), let like=recognizer.labels.firstIndex(of:"LIKE") {
            var positions:[Int]=[], alternatives:[LiveSpanInput]=[]
            for (i,key) in keys.enumerated() {
                let row=logits[i];let order=Array(row.prefix(wordCount).indices.sorted { row[$0]>row[$1] })
                guard order.first==like && order.prefix(3).contains(my) else { continue }
                let lo=seconds(key.s)-Experiment.altPre, hi=seconds(key.e)
                let ids=(max(0,key.s-off-5)..<min(obs.count,key.e-off+1)).filter { obs[$0].seconds>=lo-1e-9 && obs[$0].seconds<=hi+1e-9 }
                guard ids.count>=4, let x=LiveSpanRecognizer.inputs(ids.map {obs[$0]},ids.map {hands[$0]}) else {continue}
                positions.append(i);alternatives.append(x)
            }
            if !alternatives.isEmpty {
                if Experiment.policy != "current" && Experiment.expert == nil {
                    Experiment.expert=try LiveSpanRecognizer(name:"SpanRecognizerV17LocalALettersB8FP16",labels:recognizer.labels,units:.all)
                }
                let current = Experiment.policy == "old" ? [] : try recognizer.logits(alternatives)
                let old = Experiment.policy == "current" ? [] : try Experiment.expert!.logits(alternatives)
                for (j,i) in positions.enumerated() {
                    func wins(_ row:[Float])->Bool { Array(row.prefix(wordCount)).indices.max(by:{row[$0]<row[$1]})==my }
                    let accept = Experiment.policy == "current" ? wins(current[j]) : Experiment.policy == "old" ? wins(old[j]) : wins(current[j]) && wins(old[j])
                    if accept {logits[i].swapAt(my,like)}
                    Experiment.events.append(["rescore":true,"s":seconds(keys[i].s),"e":seconds(keys[i].e),"accepted":accept,"policy":Experiment.policy])
                }
            }
        }'''
assert s.count(needle)==1;s=s.replace(needle,replace);p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
with (D/'rescore_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/'rescore_probe')]+files+[str(d/'main.swift')],stdout=f,stderr=subprocess.STDOUT,check=True)
N=D.parent/'sept30_fix_port_v17_20261011';P=D.parent/'temporal_repair_training_v17_20261011'
arms=[(f'{policy}_{int(pre*1000)}',policy,pre) for policy in ['current','old','consensus'] for pre in [.1,.2]]
base=json.loads((N/'new_video_cases.json').read_text());genuine=json.loads((P/'genuine_control_cases.json').read_text())
for kind,rr in [('screen',[r for r in base if r['phase'] in [0,4]]),('genuine',genuine)]:
 cases=[dict(r,id=name+'__'+r['id'],policy=policy,alt_pre=pre) for r in rr for name,policy,pre in arms]
 (D/('rescore_'+kind+'_cases.json')).write_text(json.dumps(cases,indent=2)+'\n')
print('Built six targeted rescoring arms')
