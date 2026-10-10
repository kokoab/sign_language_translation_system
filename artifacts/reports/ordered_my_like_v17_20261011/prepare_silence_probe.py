from pathlib import Path
import json,shutil,subprocess
O=Path(__file__).resolve().parent;D=O/'silence_probe';D.mkdir(exist_ok=True)
for p in (O/'native').glob('*.swift'):shutil.copy2(p,D/p.name)
s=(D/'LiveReelDecoder.swift').read_text();needle='        if LiveMyLikeMotion.mode == "on", mode != .letters,'
insert='''        if mode != .letters && seconds(T)>=2.5 && seconds(T)<=4.5 {
            let rr=LiveMyLikeMotion.features(obs)
            var audit:[[String:Any]]=[]
            for span in spans where seconds(span.s)<3.3 && seconds(span.e)>2.7 {
                guard let row=local[SpanKey(span.s-origin,span.e-origin)] else {continue}
                let order=Array(row.prefix(recognizer.labels.count).indices.sorted{row[$0]>row[$1]}.prefix(4))
                let nextStart=segs.filter{$0.start+origin>span.e && $0.gloss != nil && !$0.gloss!.hasPrefix("FS_") && !["MY","LIKE"].contains($0.gloss!)}.map{seconds($0.start+origin)}.min()
                let (cue,reason)=LiveMyLikeMotion.assess(rr,start:seconds(span.s),end:seconds(span.e),available:seconds(T),nextStart:nextStart)
                audit.append(["start":seconds(span.s),"end":seconds(span.e),"top":order.map{recognizer.labels[$0]},"logits":order.map{row[$0]},"cue":cue,"reason":reason,"next":nextStart as Any? ?? NSNull()])
            }
            LiveMyLikeMotion.events.append(["stage":"audit","available":seconds(T),"candidates":audit,"decoded":segs.map{["gloss":$0.gloss as Any? ?? NSNull(),"start":seconds($0.start+origin),"end":seconds($0.end+origin)]}])
        }
'''
assert needle in s;s=s.replace(needle,insert+needle);(D/'LiveReelDecoder.swift').write_text(s)
cases=json.loads((O.parent/'temporal_repair_training_v17_20261011/evaluation_cases.json').read_text());c=next(c for c in cases if c['id']=='oct11_offset0.00000000_phase0');c['limit']=5
(O/'silence_case.json').write_text(json.dumps([c]))
with (O/'silence_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(O/'silence_probe_bin')]+[str(D/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','main.swift']],stdout=f,stderr=subprocess.STDOUT,check=True)
