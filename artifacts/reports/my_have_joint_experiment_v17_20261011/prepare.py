from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
R=P.parents[2]
S=Path('/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios/Runner/LiveReel')
OLD=R/'artifacts/reports/my_like_regression_v17_20261011/rollback_review'
P.joinpath('PLAN.md').write_text('''# MY/HAVE joint repair experiment — 2026-10-11
User priority: restore MY; accept situational HAVE→BIG omissions if a joint repair fails.
All work is report-local; no installation, training or protected test use.
Arms: installed control; pre-contact algorithm with no HAVE cues, geometry only, roll only,
both existing cues; current contact closure with identity locking separated from ordered
publication. Compare exact contiguous phone word traces, full-engine actual-PTS videos,
three HAVE recordings, then development validation for finalists. Existing historical model
comparisons remain evidence; intentional LIKE location is unknown, so output counts are not
accuracy labels. Do not tune geometry to mislabeled phone predictions.
''')
src=(S/'LiveReelDecoder.swift').read_text()
start=src.index('                // Never publish BIG')
end=src.index('                let st = g.start + origin',start)
guard=src[start:end]
src=src[:start]+'''                if ProcessInfo.processInfo.environment["ARM", default: "current"] == "current" {
'''+guard+'''                }
'''+src[end:]
needle='c.haveContactClosure = true; return c'
src=src.replace(needle,'''c.haveContactClosure = true
        let arm = ProcessInfo.processInfo.environment["ARM", default: "current"]
        if arm.hasPrefix("rollback") {
            c.haveContactClosure = false
            c.haveAngryCheck = arm == "rollback_geometry" || arm == "rollback_both"
            c.tiredRescueDroop = (arm == "rollback_roll" || arm == "rollback_both") ? 1.0 : nil
        }
        return c''')
src=src.replace('    private var shownEarly: [(Int, LiveWord)] = []','    private var shownEarly: [(Int, LiveWord)] = []\n    private var pendingOutput: [LiveWord] = []')
src=src.replace('early = [:]; shownEarly = []','early = [:]; shownEarly = []; pendingOutput = []')
needle='''        shownEarly = shownEarly.filter { $0.0 >= self.origin - 2 }
        trim()
        return committed'''
replacement='''        if ProcessInfo.processInfo.environment["ARM"] == "ordered" {
            // Lock identity at the original time; only publication waits for earlier segments.
            pendingOutput += committed
            let barrier = final ? Int.max : segs.filter { g in
                g.end + origin >= self.origin &&
                !shownEarly.contains(where: { abs(g.start + origin - $0.0) <= 2 })
            }.map { $0.start + origin }.min() ?? Int.max
            committed = pendingOutput.filter { $0.startFrame < barrier }.sorted { $0.startFrame < $1.startFrame }
            pendingOutput.removeAll { $0.startFrame < barrier }
        }
        shownEarly = shownEarly.filter { $0.0 >= self.origin - 2 }
        trim()
        return committed'''
assert needle in src
src=src.replace(needle,replacement)
P.joinpath('LiveReelDecoder.swift').write_text(src)
for name,source in [('video',OLD/'video_order/main.swift'),('trace',OLD/'order_trace/main.swift')]:
 d=P/name;d.mkdir(exist_ok=True);(d/'main.swift').write_text(source.read_text())
(P/'LiveReelModels.swift').write_text((OLD/'video_order/LiveReelModels.swift').read_text())
files=[S/n for n in ['LiveReelDecoder.swift','LiveReelEngine.swift','LiveReelModels.swift','LiveReelCore.swift','LiveReelStage3.swift']]+[R/'active/v17/segmental_runtime_v17.py']
(P/'source_manifest.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
log=R/'docs/ground_truth/live-streaming/log.md'
s=log.read_text();marker='# live-streaming — log\n'
s=s.replace(marker,marker+'''\n## 2026-10-11 — MY/HAVE joint experiments authorized; MY takes priority

User explicitly accepts rollback before HAVE omission repair if a joint fix fails and asks
for geometry/curl/roll-only alternatives as experiments. Report-local plan and native
harness variants prepared in my_have_joint_experiment_v17_20261011. Compare model-fixed
whole-engine NAME behavior as well as exact word traces; no training or installation.
Current sources hashed before work. New candidate separates identity locking from output
publication; rollback arms remove contact closure/global ordering with cue ablations.
Results pending; intentional LIKE location remains unknown.

''',1);log.write_text(s)
