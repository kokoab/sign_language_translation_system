from pathlib import Path
import shutil,subprocess,json
D=Path(__file__).resolve().parent;d=D/'narrow_boundary_native';shutil.copytree(D/'confidence_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('    static var pairFloor','    static var onsetFloor:Float=0.4\n    static var pairFloor');p.write_text(s)
p=d/'LiveReelDecoder.swift';s=p.read_text().replace('                       config: LiveDecoderConfig) -> [Segment] {','                       config: LiveDecoderConfig, boundaryFloors:[SpanKey:Float]=[:]) -> [Segment] {').replace('config.wB * Double(logb[span.s]) - config.c','config.wB * Double(max(logb[span.s],log(boundaryFloors[span] ?? 1e-6))) - config.c')
source=(D/'boundary_native/LiveReelDecoder.swift').read_text();a=source.index('        if mode != .letters, let my=recognizer.labels.firstIndex(of:"MY"),Experiment.onsetFloor>0');b=source.index('        let horizon = final ? known : known - config.lag',a);insert=source[a:b]
insert=insert.replace('let next=supports.first(where:{$0.0.s>span.e && seconds($0.0.s)-seconds(span.e)<=0.400001}) else{continue}', '''let covering=segs.first(where:{g in
                        guard let gloss=g.gloss,!["MY","LIKE"].contains(gloss),!gloss.hasPrefix("FS_") else{return false}
                        return g.start+origin<=span.e && g.end+origin>=span.e && span.s>=g.start+origin-2
                    }),let label=covering.gloss,
                    let next=supports.first(where:{$0.0.s>span.e && seconds($0.0.s)-seconds(span.e)<=0.400001 && recognizer.labels[$0.1]==label}) else{continue}
                guard !segs.contains(where:{$0.gloss=="MY" && $0.start+origin<=span.e && $0.end+origin>=span.s}) else{continue}''')
s=s.replace('        let horizon = final ? known : known - config.lag',insert+'        let horizon = final ? known : known - config.lag');p.write_text(s)
validation=D/'narrow_boundary_validation';validation.mkdir(exist_ok=True);shutil.copy(D/'confidence_validation/main.swift',validation/'main.swift')
for p in [d/'main.swift',validation/'main.swift']:
 s=p.read_text().replace('Experiment.agreementFloor=item["agreement_floor"] as? Double ?? 2.0','Experiment.agreementFloor=item["agreement_floor"] as? Double ?? 2.0\n        Experiment.onsetFloor=Float(item["onset_floor"] as? Double ?? 0.4)');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
for kind,main in [('video',d/'main.swift'),('validation',validation/'main.swift')]:
 with (D/('narrow_boundary_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/('narrow_boundary_'+kind+'_probe'))]+files+[str(main)],stdout=f,stderr=subprocess.STDOUT,check=True)
rr=json.load(open(D/'jitter_all_cases.json'));ids=['new_phase1','new_phase4','offset0.01666667__new_phase0','offset0.01666667__new_phase3']
(D/'narrow_boundary_screen_cases.json').write_text(json.dumps([dict(r,id=f'b{floor}__'+r['id'],floor=.5,onset_floor=floor,agreement_floor=2) for floor in [.4,.6,.8] for r in rr if r['id'] in ids]))
print('Compiled swallowed-MY-only boundary confirmation')
