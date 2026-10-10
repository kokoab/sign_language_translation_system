from pathlib import Path
import json,hashlib,shutil
G=Path(__file__).resolve().parent;O=G.parent;R=Path('/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios');B=O/'phone_backup'
assert (R/'Runner/LiveReel/LiveReelDecoder.swift').read_bytes()==(G/'phone_candidate_LiveReelDecoder.swift').read_bytes()
assert (R/'RunnerTests/RunnerTests.swift').read_text()==(B/'RunnerTests.swift').read_text()+'\n'+(G/'PhoneTest.swift').read_text()
for rel in ['Runner/LiveReel/LiveReelDecoder.swift','RunnerTests/RunnerTests.swift']:shutil.copy2(B/Path(rel).name,R/rel)
checks={rel:hashlib.sha256((R/rel).read_bytes()).hexdigest()==h for rel,h in json.loads((O/'phone_source_before.json').read_text()).items()};assert all(checks.values());(G/'phone_source_restored.json').write_text(json.dumps(checks,indent=2)+'\n');print('Six app/test sources restored and verified.')
