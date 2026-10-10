from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'PhaseTest.swift').read_text().replace('testMYHaveAuxiliaryPhaseSweep','testMYLikePhoneControls')
s=s.replace('    for pass in 0..<8 {','    let videos = try JSONSerialization.jsonObject(with: Data(contentsOf: docs.appendingPathComponent("my_have_validation_20261011/videos.json"))) as! [[String: Any]]\n    for (pass, item) in videos.enumerated() {')
s=s.replace('engine.vision.setDiagnosticPhase(pass)','engine.vision.setDiagnosticPhase(0)')
s=s.replace('dir.appendingPathComponent("take03.mp4")','docs.appendingPathComponent("my_have_validation_20261011/" + (item["path"] as! String))')
s=s.replace('if seconds > 16 {reader.cancelReading(); break};','')
s=s.replace('XCTAssertEqual(reader.status,.cancelled)','XCTAssertEqual(reader.status,.completed)')
s=s.replace('"id":"take03"','"id":item["id"]!,"reference":item["reference"]!')
s=s.replace('"my_have_phase_" + variant','"my_have_controls_" + variant')
(p/'ControlsTest.swift').write_text(s)
core=(p/'source_backup/LiveReelCore.swift').read_text().replace('/// Forget hand-slot history and the signer (after a reset); the auxiliary cadence keeps running.','/// Restart feature sampling with hand-slot and signer state so each new stream starts on an auxiliary frame.').replace('func resetTracking() { previousWrists = [:]; signer = nil }','func resetTracking() { previousWrists = [:]; signer = nil; processed = 0 }')
(p/'candidate_LiveReelCore.swift').write_text(core)
print('Prepared59controls fromidenticalphase0 andreport-local full feature-clock resetcandidate.')
