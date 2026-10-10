from pathlib import Path
import shutil,subprocess,json,hashlib
O=Path(__file__).resolve().parent;P=O.parent/'temporal_repair_training_v17_20261011';d=O/'native';d.mkdir(exist_ok=True)
for name in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','main.swift']:shutil.copy2(P/'eval_control'/name,d/name)
s=(d/'LiveReelDecoder.swift').read_text();needle='&& !gloss.hasPrefix("FS_")';assert s.count(needle)==1;s=s.replace(needle,'&& !(LiveMyLikeMotion.mode != "off" && gloss == "LIKE") '+needle)
needle='            let start = g.start + origin\n            if let shown';assert needle in s
s=s.replace(needle,'''            let start = g.start + origin
            if !final && LiveMyLikeMotion.mode != "off" && g.gloss == "LIKE" && seconds(T)-seconds(end)<0.35 {
                pending=start;break
            }
            if let shown''')
needle='            if let gloss = g.gloss {\n                let sameGap';assert s.count(needle)==1
s=s.replace(needle,'''            if let originalGloss = g.gloss {
                var gloss=originalGloss
                if mode != .letters && ["MY","LIKE"].contains(originalGloss) {
                    let nextStart=segs.filter{$0.start+origin>end && $0.gloss != nil}.map{seconds($0.start+origin)}.min()
                    let rows=LiveMyLikeMotion.features(obs)
                    let (decision,reason)=LiveMyLikeMotion.assess(rows,start:seconds(start),end:seconds(end),available:seconds(T),nextStart:nextStart)
                    var rank=999
                    if let target=recognizer.labels.firstIndex(of:decision),case .some(.some(let raw))=scores[SpanKey(start,end)] {
                        let order=Array(raw.prefix(recognizer.labels.count).indices.sorted{raw[$0]>raw[$1]})
                        rank=(order.firstIndex(of:target) ?? 998)+1
                    }
                    if LiveMyLikeMotion.mode == "on" && decision != "unknown" && rank<=3 { gloss=decision }
                    LiveMyLikeMotion.events.append(["original":originalGloss,"chosen":gloss,"decision":decision,"reason":reason,"rank":rank,"start":seconds(start),"end":seconds(end),"available":seconds(T),"next_start":nextStart as Any? ?? NSNull()])
                }
                let sameGap''')
(d/'LiveReelDecoder.swift').write_text(s);shutil.copy2(O/'LiveMyLikeMotion.swift',d/'LiveMyLikeMotion.swift')
s=(d/'main.swift').read_text().replace('        engine.reset()','        engine.reset()\n        LiveMyLikeMotion.events=[]').replace('"words": words.map(\\.json),','"motion_events": LiveMyLikeMotion.events, "words": words.map(\\.json),')
s=s.replace('func limited(_ buffer:', 'let replayResizeContext=CIContext()\nfunc limited(_ buffer:').replace('CIContext().render(', 'replayResizeContext.render(')
(d/'main.swift').write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift']]
with (O/'build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(O/'video_probe')]+files+[str(d/'main.swift')],stdout=f,stderr=subprocess.STDOUT,check=True)
v=(O.parent/'my_have_joint_experiment_v17_20261011/validation/main.swift').read_text().replace('    let runtime = sharedRuntime!; runtime.reset()','    let runtime = sharedRuntime!; runtime.reset(); LiveMyLikeMotion.events=[]').replace('["frames": frames.count,','["motion_events": LiveMyLikeMotion.events,"frames": frames.count,');(O/'validation').mkdir(exist_ok=True);(O/'validation/main.swift').write_text(v)
with (O/'validation_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(O/'validation_probe')]+files+[str(O/'validation/main.swift')],stdout=f,stderr=subprocess.STDOUT,check=True)
print('Built ordered motion native video and cached-trace probes.')
