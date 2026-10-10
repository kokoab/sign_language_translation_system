from pathlib import Path
import shutil,json,subprocess
D=Path(__file__).resolve().parent;d=D/'confirm_native';shutil.copytree(D/'mix_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('    static var mix:Float=1','    static var support=0.7, nextGap=0.4\n    static var mix:Float=1');p.write_text(s)
p=d/'main.swift';s=p.read_text().replace('        Experiment.events = []','''        Experiment.events = []
        Experiment.support=item["support"] as? Double ?? 0.7
        Experiment.nextGap=item["next_gap"] as? Double ?? 0.4''');p.write_text(s)
p=d/'LiveReelModels.swift';s=p.read_text().replace('    private let model: MLModel\n    private let wordOutput:', '    var lastAdapted: [[Float]] = []\n    private let model: MLModel\n    private let wordOutput:')
s=s.replace('        var out: [[Float]] = []\n        var i = 0','        lastAdapted=[]\n        var out: [[Float]] = []\n        var i = 0')
s=s.replace('                    chosen=zip(base,chosen).map { a,b in a+Experiment.mix*(b-a) }','                    lastAdapted.append(chosen); chosen=base');p.write_text(s)
p=d/'LiveReelDecoder.swift';s=p.read_text().replace('    var values: [SpanKey: [Float]?] = [:]','    var values: [SpanKey: [Float]?] = [:]\n    var repairValues: [SpanKey: [Float]] = [:]')
s=s.replace('    private var scores: [SpanKey: [Float]?] = [:]','    private var scores: [SpanKey: [Float]?] = [:]\n    private var repairScores: [SpanKey: [Float]] = [:]')
s=s.replace('        scores = [:]; origin = 0;', '        scores = [:]; repairScores = [:]; origin = 0;')
s=s.replace('        _ = tick("recognizer", t)','''        if !batch.isEmpty { for (key,row) in zip(keys,recognizer.lastAdapted) { shared.repairValues[key]=row } }
        for key in spans { if let row=shared.repairValues[key] { repairScores[key]=row } }
        _ = tick("recognizer", t)''')
s=s.replace('        scores = scores.filter { $0.key.s >= origin }','        scores = scores.filter { $0.key.s >= origin }\n        repairScores = repairScores.filter { $0.key.s >= origin }')
needle='        // Ordered completion guard uses raw timestamped landmarks'
# Insert immediately after the original baseline decode, before the existing motion rescue.
needle='        var segs = LiveSegmental.decode(sub, spans: localSpans, records: local, labels: labels, config: config)'
insert='''
        if mode != .letters, let my=recognizer.labels.firstIndex(of:"MY") {
            let wordCount=recognizer.labels.count
            // A second independently recognized gesture confirms the preceding contact ended.
            // No successor label is whitelisted; LIKE/MY are excluded because they share the ambiguous onset.
            let supports=spans.filter { span in
                guard let row=local[SpanKey(span.s-origin,span.e-origin)] else {return false}
                let lp=LiveSegmental.logSoftmax(row);guard let top=lp.indices.max(by:{lp[$0]<lp[$1]}),top<wordCount else{return false}
                return !["MY","LIKE"].contains(recognizer.labels[top]) && exp(lp[top])>=Experiment.support
            }
            var refined=false
            for span in spans {
                let key=SpanKey(span.s-origin,span.e-origin)
                guard var row=local[key],let adapted=repairScores[span] else {continue}
                let ap=LiveSegmental.logSoftmax(adapted).map {exp($0)}
                guard ap.indices.max(by:{ap[$0]<ap[$1]})==my && ap[my]>=Experiment.floor else {continue}
                guard let next=supports.first(where:{$0.s>span.e && seconds($0.s)-seconds(span.e)<=Experiment.nextGap}) else {continue}
                let word=Array(row.prefix(wordCount));let lp=LiveSegmental.logSoftmax(word);let q=exp(lp[my])
                guard ap[my]>q else {continue}
                let target=min(ap[my],0.999999);let adjustment=log((1-target)/max(1-q,1e-9))
                // Keep total word mass and every letter score fixed; only MY's share of word mass grows.
                let wordMass=Double(word[my])-lp[my]
                for index in 0..<wordCount { row[index]=index==my ? Float(wordMass+log(target)) : row[index]+Float(adjustment) }
                local[key]=row;refined=true
                Experiment.events.append(["confirmation":true,"start":seconds(span.s),"end":seconds(span.e),"next_start":seconds(next.s),"available":seconds(T)])
            }
            if refined {segs=LiveSegmental.decode(sub,spans:localSpans,records:local,labels:labels,config:config)}
        }
'''
assert needle in s;s=s.replace(needle,needle+insert)
s=s.replace('        shared.values = [:]','        shared.values = [:]; shared.repairValues = [:]').replace('        shared.values = shared.values.filter { $0.key.s >= oldest }','        shared.values = shared.values.filter { $0.key.s >= oldest }\n        shared.repairValues = shared.repairValues.filter { $0.key.s >= oldest }');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
with (D/'confirm_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/'confirm_probe')]+files+[str(d/'main.swift')],stdout=f,stderr=subprocess.STDOUT,check=True)
base=json.loads((D/'new_video_cases.json').read_text());arms=[dict(name=f's{s}_g{g}',support=s,next_gap=g,floor=.6) for s in [.5,.7,.85] for g in [.25,.4,.6]]
(D/'confirm_arms.json').write_text(json.dumps(arms,indent=2));(D/'confirm_screen_cases.json').write_text(json.dumps([dict(r,**{k:v for k,v in a.items() if k!='name'},id=a['name']+'__'+r['id']) for r in base if r['phase'] in [0,4] for a in arms]))
print('Built nine successor-confirmed MY refinement arms')
