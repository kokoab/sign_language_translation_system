from pathlib import Path
import shutil,json,subprocess
D=Path(__file__).resolve().parent;d=D/'mix_native';shutil.copytree(D/'bounded_native',d,dirs_exist_ok=True)
s=(D/'export_bounded.py').read_text().replace('SpanRecognizerV17BoundedRepairB8FP16','SpanRecognizerV17PairedDualB8FP16').replace("O/'bounded_export.json'","O/'paired_dual_export.json'");(D/'export_dual_paired.py').write_text(s)
p=d/'Experiment.swift';s=p.read_text().replace('    static var maxDuration','    static var mix:Float=1\n    static var maxDuration');p.write_text(s)
p=d/'main.swift';s=p.read_text().replace('        Experiment.events = []','        Experiment.events = []\n        Experiment.mix=Float(item["mix"] as? Double ?? 1)');p.write_text(s)
p=d/'LiveReelModels.swift';s=p.read_text();start=s.index('                if let original, let my=');end=s.index('\n                out.append(chosen',start);s=s[:start]+'''                if let original {
                    let base=Array(original[slot*100..<slot*100+100])
                    chosen=zip(base,chosen).map { a,b in a+Experiment.mix*(b-a) }
                }'''+s[end:];p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
with (D/'mix_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/'mix_probe')]+files+[str(d/'main.swift')],stdout=f,stderr=subprocess.STDOUT,check=True)
base=json.loads((D/'new_video_cases.json').read_text());arms=[0,.35,.5,.65,.8,.9,1]
(D/'mix_screen_cases.json').write_text(json.dumps([dict(r,mix=a,id=f'mix{a}__'+r['id']) for r in base if r['phase'] in [0,4] for a in arms]))
print('Built seven learned-update-strength arms (including no-op and full references)')
