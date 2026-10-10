from pathlib import Path
import shutil,subprocess,json
D=Path(__file__).resolve().parent;d=D/'boundary_native';shutil.copytree(D/'optimized_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('    static var pairFloor','    static var onsetFloor:Float=0.2\n    static var pairFloor');p.write_text(s)
p=d/'LiveReelDecoder.swift';s=p.read_text().replace('                       config: LiveDecoderConfig) -> [Segment] {','                       config: LiveDecoderConfig, boundaryFloors:[SpanKey:Float]=[:]) -> [Segment] {').replace('config.wB * Double(logb[span.s]) - config.c','config.wB * Double(max(logb[span.s],log(boundaryFloors[span] ?? 1e-6))) - config.c')
needle='        let horizon = final ? known : known - config.lag'
insert='''        if mode != .letters, let my=recognizer.labels.firstIndex(of:"MY"),Experiment.onsetFloor>0 {
            let wordCount=recognizer.labels.count
            let supports=spans.compactMap { span -> (SpanKey,Int)? in
                guard let row=local[SpanKey(span.s-origin,span.e-origin)] else{return nil}
                let lp=LiveSegmental.logSoftmax(row),top=row.indices.max(by:{row[$0]<row[$1]})!
                guard top<wordCount,!["MY","LIKE"].contains(recognizer.labels[top]),exp(lp[top])>=0.8 else{return nil}
                return (span,top)
            }
            var floors:[SpanKey:Float]=[:]
            for span in spans where seconds(span.e)-seconds(span.s)<=0.350001 {
                let key=SpanKey(span.s-origin,span.e-origin)
                guard let row=local[key],let adapted=repairScores[span] else{continue}
                let lp=LiveSegmental.logSoftmax(row)
                guard row.indices.max(by:{row[$0]<row[$1]})==my,exp(lp[my])>=0.8,
                    adapted.indices.max(by:{adapted[$0]<adapted[$1]})==my,
                    Experiment.plausibleMY(completionFeatures,start:seconds(span.s),end:seconds(span.e)),
                    let next=supports.first(where:{$0.0.s>span.e && seconds($0.0.s)-seconds(span.e)<=0.400001}) else{continue}
                floors[key]=Experiment.onsetFloor
                if Experiment.audit && exp(sub[key.s][B])<Experiment.onsetFloor {
                    Experiment.events.append(["boundary_confirmation":true,"s":seconds(span.s),"e":seconds(span.e),"original_onset":exp(sub[key.s][B]),"floor":Experiment.onsetFloor,"next":recognizer.labels[next.1],"next_start":seconds(next.0.s),"available":seconds(T)])
                }
            }
            if !floors.isEmpty {segs=LiveSegmental.decode(sub,spans:localSpans,records:local,labels:labels,config:config,boundaryFloors:floors)}
        }
'''
assert needle in s;s=s.replace(needle,insert+needle);p.write_text(s)
validation=D/'boundary_validation';validation.mkdir(exist_ok=True);shutil.copy(D/'preserved_validation/main.swift',validation/'main.swift')
for p in [d/'main.swift',validation/'main.swift']:
 s=p.read_text().replace('Experiment.pairFloor=item["pair_floor"] as? Double ?? 0.0','Experiment.pairFloor=item["pair_floor"] as? Double ?? 0.0\n        Experiment.onsetFloor=Float(item["onset_floor"] as? Double ?? 0.2)');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
for kind,main in [('video',d/'main.swift'),('validation',validation/'main.swift')]:
 with (D/('boundary_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/('boundary_'+kind+'_probe'))]+files+[str(main)],stdout=f,stderr=subprocess.STDOUT,check=True)
rr=json.load(open(D/'jitter_all_cases.json'));ids=['new_phase1','new_phase4','offset0.01666667__new_phase0','offset0.01666667__new_phase3']
(D/'boundary_screen_cases.json').write_text(json.dumps([dict(r,id=f'b{floor}__'+r['id'],floor=.4,onset_floor=floor) for floor in [.1,.2,.4] for r in rr if r['id'] in ids]))
print('Compiled confirmed short-MY boundary ablation, three onset floors')
