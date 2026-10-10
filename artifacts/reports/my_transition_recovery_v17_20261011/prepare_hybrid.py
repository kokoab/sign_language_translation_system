from pathlib import Path
import shutil,json,subprocess
D=Path(__file__).resolve().parent;d=D/'hybrid_native';shutil.copytree(D/'confirm_native',d,dirs_exist_ok=True)
p=d/'LiveReelModels.swift';s=p.read_text();old='                    lastAdapted.append(chosen); chosen=base';new='''                    let adapted=chosen;lastAdapted.append(adapted); chosen=base
                    if let my=labels.firstIndex(of:"MY"),repairEligible?[i+slot] == true {
                        let p=LiveSegmental.logSoftmax(adapted).map {exp($0)}
                        let q=LiveSegmental.logSoftmax(base).map {exp($0)}
                        if p.indices.max(by:{p[$0]<p[$1]})==my && p[my]>=Experiment.floor && p[my]>q[my] {
                            let target=min(p[my],0.999999)
                            chosen=q.enumerated().map { index,value in Float(log(max(1e-9,index==my ? target : value*(1-target)/max(1e-9,1-q[my])))) }
                        }
                    }''';assert old in s;s=s.replace(old,new);p.write_text(s)
p=d/'LiveReelDecoder.swift';s=p.read_text();needle='''            for span in spans {
                let key=SpanKey(span.s-origin,span.e-origin)
                guard var row=local[key],let adapted=repairScores[span] else {continue}''';new='''            for span in spans {
                guard seconds(span.e)-seconds(span.s)>Experiment.maxDuration+1e-6 else {continue}
                let key=SpanKey(span.s-origin,span.e-origin)
                guard var row=local[key],let adapted=repairScores[span] else {continue}''';assert needle in s;s=s.replace(needle,new);p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
for kind,main in [('video',d/'main.swift'),('validation',D/'confirm_validation/main.swift')]:
 with (D/('hybrid_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/('hybrid_'+kind+'_probe'))]+files+[str(main)],stdout=f,stderr=subprocess.STDOUT,check=True)
for kind in ['new','genuine','validation','splice','phase']:
 rr=json.loads((D/('confirm_'+kind+'_cases.json')).read_text());(D/('hybrid_'+kind+'_cases.json')).write_text(json.dumps([dict(r,max_duration=.35,flat=False,support=.5,next_gap=.25,floor=.6) for r in rr]))
print('Built short-contact plus successor-confirmed longer-contact candidate')
