from pathlib import Path
import shutil,subprocess
O=Path(__file__).resolve().parent;d=O/'traced';d.mkdir(exist_ok=True)
for p in (O/'continuous_contact').glob('*.swift'):shutil.copy2(p,d/p.name)
p=d/'LiveReelEngine.swift';s=p.read_text()
s=s.replace('"ff": o.faceForFeatures, "held": held]','"ff": o.faceForFeatures, "held": held, "mode": spellMode.rawValue,\n                                  "auxiliary_phase": (vision.processed - 1) % LiveVision.auxiliaryInterval,\n                                  "word_count": wordDecoder.count, "letter_count": runtime.letters.count]')
needle='    func takeTrace() -> [[String: Any]]?'
pos=s.index(needle)
s=s[:pos]+'''    private func traceEvent(_ event: String, seconds: Double? = nil) {
        traceLock.lock(); defer { traceLock.unlock() }
        guard traceRows != nil else { return }
        var row: [String: Any] = ["start": true, "event": event, "mode": spellMode.rawValue,
                                "processed": vision.processed]
        if let seconds { row["s"] = seconds }
        traceRows!.append(row)
    }
'''+s[pos:]
s=s.replace('    func reset() {\n        resetFinishGesture()', '    func reset() {\n        traceEvent("reset")\n        resetFinishGesture()')
s=s.replace('for w in try runtime.observe(observation, image: image) { shown += speller.push(w) }','for w in try runtime.observe(observation, image: image) { shown += speller.push(w) }\n            traceFrame(observation, runtime.words.hands.last, held: false)')
needle='''                for w in try runtime.letters.observe(observation, image: image)
                    where w.startSeconds >= max(spellSince, suppressUntil) {'''
assert needle in s
s=s.replace(needle,'''                let letterOut = try runtime.letters.observe(observation, image: image)
                traceFrame(observation, runtime.letters.hands.last, held: false)
                for w in letterOut where w.startSeconds >= max(spellSince, suppressUntil) {''')
s=s.replace('        let spent = Date().timeIntervalSince(started)','        if let event { traceEvent(event, seconds: seconds) }\n        let spent = Date().timeIntervalSince(started)')
p.write_text(s)
p=d/'main.swift';s=p.read_text();s=s.replace('    var records: [[String: Any]] = []','    engine.startTrace()\n    var records: [[String: Any]] = []')
s=s.replace('    let sorted = compute.sorted()','    try writeJSON(engine.takeTrace() ?? [], out + ".trace.json")\n    let sorted = compute.sorted()');p.write_text(s)
cmd=['xcrun','swiftc','-O','-o',str(O/'traced_probe')]+[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','main.swift']]
with (O/'traced_build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
print('traced engine built')
