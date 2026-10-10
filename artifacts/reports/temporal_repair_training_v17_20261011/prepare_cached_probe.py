from pathlib import Path
import shutil,subprocess
O=Path(__file__).resolve().parent;d=O/'native_cached';d.mkdir(exist_ok=True)
for p in (O/'native').glob('*.swift'):shutil.copy2(p,d/p.name)
p=d/'LiveReelCore.swift';s=p.read_text();needle='    private let handRequest = VNDetectHumanHandPoseRequest()'
s=s.replace(needle,'''    // Report-only cache of stateless detections. Tracking/signers/cadence remain per stream.
    private struct Detections { let hands: [LiveHand]; let bodies: [VNHumanBodyPoseObservation]; let faces: [VNFaceObservation] }
    private static var replayDetections: [String: Detections] = [:]
'''+needle,1)
old='''        try autoreleasepool {
            try VNImageRequestHandler(cvPixelBuffer: detection, orientation: .up, options: [:]).perform(requests)
        }
        var hands = (handRequest.results ?? []).compactMap { Self.parseHand($0) }'''
new='''        let cacheKey="\\(seconds.bitPattern):\\(width):\\(height):\\(auxiliary || displayAuxiliary):\\(signerLock)"
        let raw: Detections
        if let cached=Self.replayDetections[cacheKey] { raw=cached }
        else {
            try autoreleasepool { try VNImageRequestHandler(cvPixelBuffer: detection, orientation: .up, options: [:]).perform(requests) }
            raw=Detections(hands:(handRequest.results ?? []).compactMap { Self.parseHand($0) }, bodies: (auxiliary || displayAuxiliary) ? (bodyRequest.results ?? []) : [], faces:(auxiliary || displayAuxiliary) ? (faceRequest.results ?? []) : [])
            Self.replayDetections[cacheKey]=raw
        }
        var hands = raw.hands'''
assert old in s;s=s.replace(old,new,1).replace('let faces = faceRequest.results ?? [], bodies = bodyRequest.results ?? []','let faces = raw.faces, bodies = raw.bodies',1)
s=s.replace('    func setDiagnosticPhase(_ phase: Int) { processed = phase }','    func setDiagnosticPhase(_ phase: Int) { processed = phase }\n    static func clearReplayDetections() { replayDetections = [:] }',1)
p.write_text(s)
p=d/'main.swift';s=p.read_text().replace('for item in cases {','for item in cases {\n LiveVision.clearReplayDetections()',1);p.write_text(s)
with (O/'cached_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(O/'cached_probe')]+[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','main.swift']],stdout=f,stderr=subprocess.STDOUT,check=True)
