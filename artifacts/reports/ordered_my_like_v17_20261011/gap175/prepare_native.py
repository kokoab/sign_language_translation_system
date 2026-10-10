from pathlib import Path
import shutil,subprocess,json,hashlib
O=Path(__file__).resolve().parent;P=O.parent.parent/'temporal_repair_training_v17_20261011';d=O/'native';d.mkdir(exist_ok=True)
for name in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','main.swift']:shutil.copy2(P/'eval_control'/name,d/name)
s=(d/'LiveReelDecoder.swift').read_text();needle='&& !gloss.hasPrefix("FS_")';assert s.count(needle)==1;s=s.replace(needle,'&& !(LiveMyLikeMotion.mode != "off" && gloss == "LIKE") '+needle)
needle='            let start = g.start + origin\n            if let shown';assert needle in s
s=s.replace(needle,'''            let start = g.start + origin
            if !final && LiveMyLikeMotion.mode != "off" && g.gloss == "LIKE" && seconds(T)-seconds(end)<0.5 {
                pending=start;break
            }
            if let shown''')
needle='            if let gloss = g.gloss {\n                let sameGap';assert s.count(needle)==1
s=s.replace(needle,'''            if let originalGloss = g.gloss {
                var gloss=originalGloss
                if mode != .letters && ["MY","LIKE"].contains(originalGloss) {
                    let nextStart=segs.filter{$0.start+origin>end && $0.gloss != nil && !$0.gloss!.hasPrefix("FS_") && !["MY","LIKE"].contains($0.gloss!)}.map{seconds($0.start+origin)}.min()
                    let rows=LiveMyLikeMotion.features(obs)
                    let (decision,reason)=LiveMyLikeMotion.assess(rows,start:seconds(start),end:seconds(end),available:seconds(T),nextStart:nextStart)
                    var rank=999
                    if let target=recognizer.labels.firstIndex(of:decision),case .some(.some(let raw))=scores[SpanKey(start,end)] {
                        let order=Array(raw.prefix(recognizer.labels.count).indices.sorted{raw[$0]>raw[$1]})
                        rank=(order.firstIndex(of:target) ?? 998)+1
                    }
                    // Word identities are resolved through candidate scores before decoding, never rewritten here.
                    LiveMyLikeMotion.events.append(["original":originalGloss,"chosen":gloss,"decision":decision,"reason":reason,"rank":rank,"rows":rows.map{["t":$0.t,"slot":$0.slot == 0 ? "l" : "r","pinch":$0.pinch,"straight":$0.straight,"flat_valid":$0.flatValid,"size":$0.size,"px":$0.px,"py":$0.py] as [String:Any]},"start":seconds(start),"end":seconds(end),"available":seconds(T),"next_start":nextStart as Any? ?? NSNull()])
                }
                let sameGap''')
s=s.replace('let segs = LiveSegmental.decode(sub, spans: localSpans, records: local, labels: labels, config: config)', 'var segs = LiveSegmental.decode(sub, spans: localSpans, records: local, labels: labels, config: config)\n        if LiveMyLikeMotion.mode == "on", mode != .letters, let my=recognizer.labels.firstIndex(of:"MY"), let like=recognizer.labels.firstIndex(of:"LIKE") {\n            let motionRows=LiveMyLikeMotion.features(obs)\n            var changed=false\n            for span in spans {\n                let key=SpanKey(span.s-origin,span.e-origin)\n                guard var row=local[key] else { continue }\n                let order=Array(row.prefix(recognizer.labels.count).indices.sorted{row[$0]>row[$1]})\n                guard order.first==like && order.prefix(3).contains(my) else { continue }\n                let nextStart=segs.filter{$0.start+origin>span.e && $0.gloss != nil && !$0.gloss!.hasPrefix("FS_") && !["MY","LIKE"].contains($0.gloss!)}.map{seconds($0.start+origin)}.min()\n                let (decision,reason)=LiveMyLikeMotion.assess(motionRows,start:seconds(span.s),end:seconds(span.e),available:seconds(T),nextStart:nextStart)\n                if decision == "MY" {\n                    row.swapAt(my,like);local[key]=row;changed=true\n                    let eventKey="\\(span.s):\\(span.e)"\n                    if !LiveMyLikeMotion.logged.contains(eventKey) {\n                        LiveMyLikeMotion.logged.insert(eventKey)\n                        LiveMyLikeMotion.events.append(["stage":"span","original":"LIKE","chosen":"MY","start":seconds(span.s),"end":seconds(span.e),"available":seconds(T),"reason":reason])\n                    }\n                }\n            }\n            if changed { segs=LiveSegmental.decode(sub,spans:localSpans,records:local,labels:labels,config:config) }\n        }')
(d/'LiveReelDecoder.swift').write_text(s);shutil.copy2(O/'LiveMyLikeMotion.swift',d/'LiveMyLikeMotion.swift')
s=(d/'main.swift').read_text().replace('        engine.reset()','        engine.reset()\n        LiveMyLikeMotion.events=[]; LiveMyLikeMotion.logged=[]').replace('"words": words.map(\\.json),','"motion_events": LiveMyLikeMotion.events, "words": words.map(\\.json),')
s=s.replace('func limited(_ buffer:', 'let replayResizeContext=CIContext()\nfunc limited(_ buffer:').replace('CIContext().render(', 'replayResizeContext.render(')
(d/'main.swift').write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift']]
with (O/'build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(O/'video_probe')]+files+[str(d/'main.swift')],stdout=f,stderr=subprocess.STDOUT,check=True)
v=(O.parent.parent/'my_have_joint_experiment_v17_20261011/validation/main.swift').read_text().replace('    let runtime = sharedRuntime!; runtime.reset()','    let runtime = sharedRuntime!; runtime.reset(); LiveMyLikeMotion.events=[]; LiveMyLikeMotion.logged=[]').replace('["frames": frames.count,','["motion_events": LiveMyLikeMotion.events,"frames": frames.count,');(O/'validation').mkdir(exist_ok=True);(O/'validation/main.swift').write_text(v)
with (O/'validation_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(O/'validation_probe')]+files+[str(O/'validation/main.swift')],stdout=f,stderr=subprocess.STDOUT,check=True)
print('Built ordered motion native video and cached-trace probes.')
