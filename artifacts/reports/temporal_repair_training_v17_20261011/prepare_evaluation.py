from pathlib import Path
import json,shutil,subprocess
O=Path(__file__).resolve().parent;P=O.parent/'clean_lineage_recovery_v17_20261011';R=O.parents[2]
resources=O/'resources';resources.mkdir(exist_ok=True)
for p in (P/'resources').iterdir():
 q=resources/p.name
 if not q.exists():q.symlink_to(p.resolve(),target_is_directory=p.is_dir())
for variant in ['control','continuous_contact']:
 d=O/('eval_'+variant);d.mkdir(exist_ok=True)
 for name in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift']:
  src=O/'native_cached/LiveReelCore.swift' if name=='LiveReelCore.swift' else P/variant/name;shutil.copy2(src,d/name)
 s=(P/variant/'main.swift').read_text().replace('    for item in list {','    var previousPath=""\n    for item in list {\n        let path=item["path"] as! String\n        if path != previousPath { LiveVision.clearReplayDetections();previousPath=path }',1).replace('var deadline = 0.0','var deadline = item["capture_offset"] as? Double ?? 0.0',1)
 (d/'main.swift').write_text(s)
 with (O/(variant+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(O/(variant+'_probe'))]+[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','main.swift']],stdout=f,stderr=subprocess.STDOUT,check=True)
base=json.loads((O.parent/'lineage_runtime_investigation_v17_20261011/cases.json').read_text());cases=[]
for prefix in ['sept30','oct11','have_take11']:
 original=next(r for r in base if r['id'].startswith(prefix))
 for offset in [0,1/60,1/30]:
  for phase in range(8):
   row=dict(original,id=f'{prefix}_offset{offset:.8f}_phase{phase}',phase=phase,capture_offset=offset);cases.append(row)
(O/'evaluation_cases.json').write_text(json.dumps(cases,indent=2)+'\n');print('Prepared72phase/capture cases per model/runtime arm.')
