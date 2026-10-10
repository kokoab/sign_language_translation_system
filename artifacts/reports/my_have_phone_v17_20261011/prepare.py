from pathlib import Path
import json,shutil,hashlib
out=Path(__file__).resolve().parent
root=Path('/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios')
prior=out.parent/'my_have_joint_experiment_v17_20261011'
files=['Runner/LiveReel/LiveReelDecoder.swift','Runner/LiveReel/LiveReelEngine.swift','RunnerTests/RunnerTests.swift','Runner/AppDelegate.swift']
hashes={}
for rel in files:
 p=root/rel;q=out/'source_backup'/p.name
 assert not q.exists(),q
 shutil.copy2(p,q);hashes[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'source_hashes_before.json').write_text(json.dumps(hashes,indent=2))
cases=json.loads((prior/'phone_and_help_cases.json').read_text())
for item in cases:
 p=Path(item['path']);q=out/'device_inputs'/(item['id']+'.json');shutil.copy2(p,q);item['path']=q.name
(out/'device_inputs/cases.json').write_text(json.dumps(cases))
# Keep movie inputs out of build resources; use the test-only Documents folder.
shutil.copy2('/Volumes/secret/SLT/SLT/data/local/phone_session_pull_v17_20261011/20261011_102647_03.mp4',out/'device_inputs/take03.mp4')
# Eager single fallback shared by all word decoders; propagate failure at startup, not silently mid-frame.
s=(prior/'v2/LiveReelDecoder.swift').read_text()
s=s.replace('private lazy var myFallback = try? LiveSpanRecognizer(name: "SpanRecognizerV17LocalALettersB8FP16", labels: recognizer.labels, letterThreshold: 0.5, units: .all)','let myFallback: LiveSpanRecognizer?')
s=s.replace('mode: LiveDecoderMode, letterGate: Double? = nil) {','mode: LiveDecoderMode, letterGate: Double? = nil, myFallback: LiveSpanRecognizer? = nil) {')
s=s.replace('self.letterGate = letterGate','self.myFallback = myFallback\n        self.letterGate = letterGate',1)
s=s.replace('ProcessInfo.processInfo.environment["HYBRID"] != "0", mode != .letters','mode != .letters')
s=s.replace('                print("HYBRID_RESCUE", keys[j].s, keys[j].e)\n','')
(out/'candidate_LiveReelDecoder.swift').write_text(s)
s=(out/'source_backup/LiveReelEngine.swift').read_text()
s=s.replace('        let encoder = try LiveHandEncoder(units: encoderUnits)','        let myFallback = try LiveSpanRecognizer(name: "SpanRecognizerV17LocalALettersB8FP16", labels: labels, units: recognizerUnits)\n        let encoder = try LiveHandEncoder(units: encoderUnits)',1)
s=s.replace('encoder: encoder, mode: .both)', 'encoder: encoder, mode: .both, myFallback: myFallback)',1)
s=s.replace('config: .v3Words, mode: .words)', 'config: .v3Words, mode: .words, myFallback: myFallback)',1)
s=s.replace('mode: .both, letterGate: Self.wordLetterGate)', 'mode: .both, letterGate: Self.wordLetterGate, myFallback: myFallback)',1)
s=s.replace('        _ = try recognizer.logits([zero])','        _ = try recognizer.logits([zero])\n        _ = try runtime.words.myFallback?.logits([zero])',1)
(out/'candidate_LiveReelEngine.swift').write_text(s)
# Both benchmark variants suppress the ordinary automatic engine, preventing a duplicate engine during XCTest.
s=(out/'source_backup/AppDelegate.swift').read_text().replace('    LiveReelShared.warm()','    // Diagnostic build: XCTest owns the engine and its cold/warm measurements.')
(root/'Runner/AppDelegate.swift').write_text(s)
print('Backed up four sources; prepared shared prewarmed candidate and seven trace inputs')
