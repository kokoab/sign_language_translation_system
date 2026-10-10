from pathlib import Path
import shutil,subprocess,json
O=Path(__file__).resolve().parent;d=O/'native_context';d.mkdir(exist_ok=True)
for p in (O/'native_oriented').glob('*.swift'):shutil.copy2(p,d/p.name)
p=d/'main.swift';s=p.read_text();s=s.replace('func limited(_ buffer:', 'let replayResizeContext=CIContext()\nfunc limited(_ buffer:',1);s=s.replace('    CIContext().render(', '    replayResizeContext.render(',1);p.write_text(s)
with (O/'context_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(O/'context_probe')]+[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','main.swift']],stdout=f,stderr=subprocess.STDOUT,check=True)
case=next(c for c in json.loads((O/'cases.json').read_text()) if 89<c['limit']<90);(O/'context_long_case.json').write_text(json.dumps([case])+'\n')
