"""Report-only, class-independent three-view word probability averaging.

Keep the candidate's normal view and letter logits, add an earlier onset and a
shorter release view. No MY/HAVE lookup or per-class threshold. This is an
inference robustness diagnostic, not a new training recipe or shipping default.
"""
from pathlib import Path
import json,shutil,subprocess,os
O=Path(__file__).resolve().parent
d=O/'consensus';d.mkdir(exist_ok=True)
for p in (O/'continuous_contact').glob('*.swift'):shutil.copy2(p,d/p.name)
p=d/'LiveReelDecoder.swift';s=p.read_text()
needle='        let wordCount = recognizer.labels.count\n'
assert s.count(needle)==1
s=s.replace(needle,needle+'''
        if mode != .letters {
            var extra: [LiveSpanInput] = [], owners: [Int] = []
            for (j, key) in keys.enumerated() {
                for view in [SpanKey(max(off, key.s - 2), key.e),
                             SpanKey(key.s, min(key.e, max(key.s + 3, key.e - 2)))] {
                    if view.s == key.s && view.e == key.e { continue }
                    let ids = spanObs(view.s, view.e)
                    if ids.count >= 4, let input = LiveSpanRecognizer.inputs(ids.map { obs[$0] }, ids.map { hands[$0] }) {
                        extra.append(input); owners.append(j)
                    }
                }
            }
            let extraLogits = extra.isEmpty ? [] : try recognizer.logits(extra)
            var probabilities = logits.map { LiveSegmental.logSoftmax(Array($0[..<wordCount])).map { exp($0) } }
            var counts = [Double](repeating: 1, count: logits.count)
            for (j, row) in zip(owners, extraLogits) {
                let lp = LiveSegmental.logSoftmax(Array(row[..<wordCount]))
                for c in 0..<wordCount { probabilities[j][c] += exp(lp[c]) }
                counts[j] += 1
            }
            for j in logits.indices {
                for c in 0..<wordCount { logits[j][c] = Float(log(max(1e-12, probabilities[j][c] / counts[j]))) }
            }
        }
''')
p.write_text(s)
with (O/'consensus_build.log').open('w') as f:
 subprocess.run(['xcrun','swiftc','-O','-o',str(O/'consensus_probe')]+[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','main.swift']],stdout=f,stderr=subprocess.STDOUT,check=True)
with (O/'consensus_replay.log').open('w') as f:
 subprocess.run([str(O/'consensus_probe'),str(O/'resources'),str(O.parent/'lineage_runtime_investigation_v17_20261011/cases.json'),str(O/'consensus_replay.json')],env=dict(os.environ,FINISH='0',LETTER_THRESHOLD='0.7'),stdout=f,stderr=subprocess.STDOUT,check=True)
print('Consensus diagnostic complete')
