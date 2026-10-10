from pathlib import Path
import json,shutil,subprocess
D=Path(__file__).resolve().parent;d=D/'completion_native';shutil.copytree(D/'native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('    static var audit', '''    static var gap=0.4, duration=0.6, tail=0.0
    static var audit''');p.write_text(s)
p=d/'main.swift';s=p.read_text().replace('        Experiment.events = []','''        Experiment.events = []
        Experiment.gap=item["gap"] as? Double ?? 0.4
        Experiment.duration=item["duration"] as? Double ?? 0.6
        Experiment.tail=item["tail"] as? Double ?? 0.0''');p.write_text(s)
p=d/'LiveMyLikeMotion.swift';s=p.read_text().replace('let limit=min(end+0.5,nextStart.map{$0-1e-5} ?? Double.infinity)','''let adjacent = nextStart.map { $0>end && $0-end<=Experiment.gap && end-start<=Experiment.duration } ?? false
        let limit=adjacent ? end+Experiment.tail : min(end+0.5,nextStart.map{$0-1e-5} ?? Double.infinity)''').replace('guard limit>=end+0.08 && available>=limit-1e-6','guard (adjacent || limit>=end+0.08) && available>=limit-1e-6');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
with (D/'completion_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/'completion_probe')]+files+[str(d/'main.swift')],stdout=f,stderr=subprocess.STDOUT,check=True)
N=D.parent/'sept30_fix_port_v17_20261011';base=json.loads((N/'new_video_cases.json').read_text())
arms=[dict(name=f'gap{gap}_duration{duration}_tail{tail}',gap=gap,duration=duration,tail=tail) for gap in [.25,.4,.6] for duration in [.35,.6] for tail in [0,.05]]
(D/'completion_arms.json').write_text(json.dumps(arms,indent=2))
(D/'completion_screen_cases.json').write_text(json.dumps([dict(r,**{k:v for k,v in a.items() if k!='name'},id=a['name']+'__'+r['id']) for r in base if r['phase'] in [0,4] for a in arms],indent=2))
print('Built 12 adjacent-completion arms')
