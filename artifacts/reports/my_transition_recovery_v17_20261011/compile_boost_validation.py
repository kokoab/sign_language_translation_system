from pathlib import Path
import subprocess
D=Path(__file__).resolve().parent;G=D.parent/'ordered_my_like_v17_20261011/gap175'
d=D/'boost_native';v=D/'boost_validation';v.mkdir(exist_ok=True)
s=(G/'validation/main.swift').read_text();s=s.replace('    let result = try autoreleasepool','''    Experiment.policy = item["policy"] as? String ?? "none"
    Experiment.altPre = item["alt_pre"] as? Double ?? 0.1
    Experiment.pre = item["pre"] as? Double ?? 0.1
    Experiment.post = item["post"] as? Double ?? 0.1
    Experiment.events = []
    Experiment.kind=item["kind"] as? String ?? "swap"
    Experiment.eligibility=item["eligibility"] as? String ?? "like"
    Experiment.floor=item["floor"] as? Double ?? 0.0
    let result = try autoreleasepool''');s=s.replace('"motion_events": LiveMyLikeMotion.events,','"score_events": Experiment.events, "motion_events": LiveMyLikeMotion.events,');(v/'main.swift').write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
with (D/'boost_validation_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/'boost_validation_probe')]+files+[str(v/'main.swift')],stdout=f,stderr=subprocess.STDOUT,check=True)
print('Built validation harness')
