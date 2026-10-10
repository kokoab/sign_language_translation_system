from pathlib import Path
import shutil,subprocess,json
D=Path(__file__).resolve().parent;d=D/'decision_native';shutil.copytree(D/'confirm_native',d,dirs_exist_ok=True)
p=d/'LiveReelDecoder.swift';s=p.read_text();a=s.index('        if mode != .letters, let my=recognizer.labels.firstIndex(of:"MY") {',s.index('        var segs = LiveSegmental.decode'));b=s.index('        if LiveMyLikeMotion.mode == "on"',a);s=s[:a]+s[b:]
needle='        let horizon = final ? known : known - config.lag';insert='''
        if mode != .letters, let my=recognizer.labels.firstIndex(of:"MY") {
            let motionRows=LiveMyLikeMotion.features(obs)
            for index in segs.indices {
                let g=segs[index],start=g.start+origin,end=g.end+origin
                guard g.gloss=="LIKE",end<=horizon,(final || seconds(T)-seconds(end)>=0.5),
                    let adapted=repairScores[SpanKey(start,end)] else{continue}
                let p=LiveSegmental.logSoftmax(adapted).map{exp($0)}
                guard p.indices.max(by:{p[$0]<p[$1]})==my,p[my]>=Experiment.floor else{continue}
                let nextStart=segs.filter{$0.start+origin>end && $0.gloss != nil && !$0.gloss!.hasPrefix("FS_") && !["MY","LIKE"].contains($0.gloss!)}.map{seconds($0.start+origin)}.min()
                let (decision,reason)=LiveMyLikeMotion.assess(motionRows,start:seconds(start),end:seconds(end),available:seconds(T),nextStart:nextStart)
                guard decision != "LIKE" else{continue}
                segs[index].gloss="MY"
                if Experiment.audit {Experiment.events.append(["identity_only":true,"s":seconds(start),"e":seconds(end),"adapted_my":p[my],"available":seconds(T),"motion":decision,"reason":reason])}
            }
        }
''';s=s.replace(needle,needle+insert);p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
for kind,main in [('video',d/'main.swift'),('validation',D/'confirm_validation/main.swift')]:
 with (D/('decision_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/('decision_'+kind+'_probe'))]+files+[str(main)],stdout=f,stderr=subprocess.STDOUT,check=True)
rr=json.load(open(D/'jitter_all_cases.json'));(D/'decision_all_cases.json').write_text(json.dumps([dict(r,floor=.3) for r in rr]));rr=json.load(open(D/'preserved_genuine_cases.json'));(D/'decision_genuine_cases.json').write_text(json.dumps([dict(r,floor=.3) for r in rr]))
print('Compiled completed-segment identity correction with original scores and boundaries preserved')
