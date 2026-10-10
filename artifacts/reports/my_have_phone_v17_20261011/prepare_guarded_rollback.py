from pathlib import Path
p=Path(__file__).resolve().parent;q=p.parent/'my_have_joint_experiment_v17_20261011'
s=(q/'fallback_before_contact/LiveReelDecoder.swift').read_text()
s=s.replace('    private var shownEarly: [(Int, LiveWord)] = []','    private var shownEarly: [(Int, LiveWord)] = []\n    private var pendingOutput: [LiveWord] = []')
s=s.replace('early = [:]; shownEarly = []','early = [:]; shownEarly = []; pendingOutput = []')
assert s.count('                    last = w')==2
s=s.replace('                    last = w','                    if last == nil || w.startFrame >= last!.startFrame { last = w }')
s=s.replace('shown.endSeconds = seconds(end)','shown.endSeconds = seconds(shown.endFrame)')
assert s.count('                    last.endFrame = end')==1
s=s.replace('                    last.endFrame = end','                    last.endFrame = max(last.endFrame, end)\n                    last.endSeconds = seconds(last.endFrame)')
v2=(q/'v2/LiveReelDecoder.swift').read_text();start=v2.index('        do {\n            // Lock identity');end=v2.index('        shownEarly = shownEarly.filter',start)
s=s.replace('        shownEarly = shownEarly.filter',v2[start:end]+'        shownEarly = shownEarly.filter',1)
assert 'myFallback' not in s and 'haveContact' not in s and 'HYBRID' not in s
(p/'guarded_rollback_LiveReelDecoder.swift').write_text(s)
# Keep the runtime checks compatible with the existing native batch-trace harness.
(p/'guarded_rollback').mkdir(exist_ok=True)
(p/'guarded_rollback/main.swift').write_text((q/'validation/main.swift').read_text())
print('Prepared bounded rollback: prior HAVE policy + chronological publication/last-word/end-time invariants; no ensemble, capture or feature-clock change.')
