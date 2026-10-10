from pathlib import Path
import shutil,subprocess
O=Path(__file__).resolve().parent;d=O/'native_cached_fast';d.mkdir(exist_ok=True)
for p in (O/'native_cached').glob('*.swift'):shutil.copy2(p,d/p.name)
p=d/'LiveReelCore.swift';s=p.read_text().replace('    static func clearReplayDetections() { replayDetections = [:] }','''    static func clearReplayDetections() { replayDetections = [:] }
    func hasReplayDetection(width: Int, height: Int, seconds: Double) -> Bool {
        let auxiliary=processed % auxiliaryCadence == 0
        return Self.replayDetections["\\(seconds.bitPattern):\\(width):\\(height):\\(auxiliary):\\(signerLock)"] != nil
    }''');p.write_text(s)
p=d/'main.swift';s=p.read_text();a=s.index('     let frame=limited(source,1280)');b=s.index('\n    }\n   }\n   for event',a)
s=s[:a]+'''     let sw=CVPixelBufferGetWidth(source),sh=CVPixelBufferGetHeight(source)
     let scale=min(1.0,1280.0/Double(max(sw,sh)))
     let w=Int((Double(sw)*scale).rounded()),h=Int((Double(sh)*scale).rounded())
     let needed=(item["events"] as! [[String:Any]]).contains { event in
      let core=event["core"] as! [Double]
      return seconds >= core[0]-0.2-1e-9 && seconds < core[1]+0.1-1e-9
     }
     let cached=vision.hasReplayDetection(width:w,height:h,seconds:seconds)
     let frame=(needed || !cached) ? limited(source,1280) : source
     let observation=try vision.observe(detection:cached ? frame : try scaler.resize(frame),width:w,height:h,seconds:seconds)
     if needed {
      CVPixelBufferLockBaseAddress(frame,.readOnly)
      defer {CVPixelBufferUnlockBaseAddress(frame,.readOnly)}
      let image=LiveBGRAImage(base:CVPixelBufferGetBaseAddress(frame)!.assumingMemoryBound(to:UInt8.self),width:w,height:h,bytesPerRow:CVPixelBufferGetBytesPerRow(frame))
      observations.append(observation);hands.append(try encoder.frameHand(observation,image:image))
     }'''+s[b:];p.write_text(s)
with (O/'cached_fast_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(O/'cached_fast_probe')]+[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','main.swift']],stdout=f,stderr=subprocess.STDOUT,check=True)
