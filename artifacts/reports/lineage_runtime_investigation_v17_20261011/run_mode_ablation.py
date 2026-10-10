from pathlib import Path
import shutil,subprocess,os,json
O=Path(__file__).resolve().parent;d=O/'mode_ablation';d.mkdir(exist_ok=True)
for p in (O/'current').glob('*.swift'):shutil.copy2(p,d/p.name)
p=d/'LiveReelEngine.swift';s=p.read_text()
s=s.replace('if let name = out.last(where:', 'if ProcessInfo.processInfo.environment["NO_NAME"] != "1", let name = out.last(where:')
needle='words: LiveSegmentalRuntime(boundary: wordBoundary, recognizer: recognizer, encoder: encoder, mode: .both),'
assert needle in s
s=s.replace(needle,'words: LiveSegmentalRuntime(boundary: wordBoundary, recognizer: recognizer, encoder: encoder, config: ProcessInfo.processInfo.environment["UNIFORM_WORD"] == "1" ? .v3Words : .v3, mode: .both, letterGate: ProcessInfo.processInfo.environment["UNIFORM_WORD"] == "1" ? Self.wordLetterGate : nil),')
p.write_text(s)
cmd=['xcrun','swiftc','-O','-o',str(O/'mode_probe')]+[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','main.swift']]
with (O/'mode_build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
cases=[c for c in json.load(open(O/'cases.json')) if c['id'].startswith('oct11')]
(O/'mode_cases.json').write_text(json.dumps(cases))
for mode,flag in [('no_name','NO_NAME'),('uniform_word','UNIFORM_WORD')]:
 with (O/(mode+'.log')).open('w') as f:subprocess.run([str(O/'mode_probe'),str(O.parent/'my_like_regression_v17_20261011/resources'),str(O/'mode_cases.json'),str(O/(mode+'.json'))],env=dict(os.environ,FINISH='0',**{flag:'1'}),stdout=f,stderr=subprocess.STDOUT,check=True)
 print(mode,'complete',flush=True)
