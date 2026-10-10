from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent
root=Path('/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios')
core=root/'Runner/LiveReel/LiveReelCore.swift'
backup=p/'source_backup'/core.name
if not backup.exists():
 shutil.copy2(core,backup)
 manifest=json.loads((p/'source_hashes_before.json').read_text());manifest['Runner/LiveReel/'+core.name]=hashlib.sha256(core.read_bytes()).hexdigest();(p/'source_hashes_before.json').write_text(json.dumps(manifest,indent=2))
s=backup.read_text().replace('    private(set) var processed = 0','    private(set) var processed = 0\n    func setDiagnosticPhase(_ phase: Int) { processed = phase }')
core.write_text(s)
# Reuse the same movie decoder and timing implementation; only phase and clip duration change.
s=(p/'DeviceTests.swift').read_text()
s=s.replace('testMYHaveDeviceProfile','testMYHaveAuxiliaryPhaseSweep')
start=s.index('    let cases = try read("cases.json")')
end=s.index('    // Full engine replay',start)
s=s[:start]+s[end:]
s=s.replace('for pass in 0..<2 {','for pass in 0..<8 {')
s=s.replace('        engine.reset()','        engine.reset()\n        engine.vision.setDiagnosticPhase(pass)')
s=s.replace('let seconds=pts-first!;if seconds+1e-6 < deadline','let seconds=pts-first!;if seconds > 16 {reader.cancelReading(); break};if seconds+1e-6 < deadline')
s=s.replace('XCTAssertEqual(reader.status,.completed)','XCTAssertEqual(reader.status,.cancelled)')
s=s.replace('"my_have_" + variant','"my_have_phase_" + variant')
(p/'PhaseTest.swift').write_text(s)
(root/'RunnerTests/RunnerTests.swift').write_text((p/'source_backup/RunnerTests.swift').read_text()+'\n'+s)
print('Diagnostic phase sweep: same first16seconds, one engine/reset, force all8 body/face extraction phases.')
