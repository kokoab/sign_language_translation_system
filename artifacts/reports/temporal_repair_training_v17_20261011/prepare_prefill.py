from pathlib import Path
import shutil,subprocess
O=Path(__file__).resolve().parent;d=O/'native_prefill';d.mkdir(exist_ok=True)
for p in (O/'native_context').glob('*.swift'):shutil.copy2(p,d/p.name)
p=d/'main.swift';s=p.read_text();s=s.replace('for item in cases {','''struct ReplayFrame {
 let seconds: Double; let width: Int; let height: Int; let offsets: [Int]; let pixels: CVPixelBuffer?
}
for item in cases {''',1)
a=s.index(' for offset in [0.0,1.0/60,1.0/30] {');b=s.index('   for event in item["events"]',a)
s=s[:a]+''' let source=try VideoFrames(URL(fileURLWithPath:item["path"] as! String))
 let warm=LiveVision(),scaler=LiveDetectionScaler()
 var deadlines=[0.0,1.0/60,1.0/30],replay=[ReplayFrame]()
 var dummy: CVPixelBuffer?
 CVPixelBufferCreate(kCFAllocatorDefault,1,1,kCVPixelFormatType_32BGRA,nil,&dummy)
 while let original=source.next() {
  let seconds=source.lastPTS
  if seconds > (item["limit"] as! Double) { break }
  var accepted=[Int]()
  for i in deadlines.indices where seconds+1e-6>=deadlines[i] { deadlines[i]=max(deadlines[i]+0.05,seconds);accepted.append(i) }
  if accepted.isEmpty { continue }
  try autoreleasepool {
   let frame=limited(original,1280),w=CVPixelBufferGetWidth(frame),h=CVPixelBufferGetHeight(frame)
   let detection=try scaler.resize(frame)
   // Populate both exact request signatures once; discard warm-up tracking state.
   for phase in [0,1] { warm.setDiagnosticPhase(phase);_ = try warm.observe(detection:detection,width:w,height:h,seconds:seconds) }
   let needed=(item["events"] as! [[String:Any]]).contains { event in
    let core=event["core"] as! [Double]
    return seconds>=core[0]-0.2-1e-9 && seconds<core[1]+0.1-1e-9
   }
   replay.append(ReplayFrame(seconds:seconds,width:w,height:h,offsets:accepted,pixels:needed ? frame : nil))
  }
 }
 print(item["id"]!,"decoded once",replay.count);fflush(stdout)
 for (offsetIndex,offset) in [0.0,1.0/60,1.0/30].enumerated() {
  for phase in 0..<8 {
   let vision=LiveVision();vision.setDiagnosticPhase(phase)
   var observations=[LiveObservation](),hands=[LiveHandFrame]()
   for r in replay where r.offsets.contains(offsetIndex) {
    precondition(vision.hasReplayDetection(width:r.width,height:r.height,seconds:r.seconds))
    let observation=try vision.observe(detection:dummy!,width:r.width,height:r.height,seconds:r.seconds)
    if let frame=r.pixels {
     try autoreleasepool {
      CVPixelBufferLockBaseAddress(frame,.readOnly)
      defer { CVPixelBufferUnlockBaseAddress(frame,.readOnly) }
      let image=LiveBGRAImage(base:CVPixelBufferGetBaseAddress(frame)!.assumingMemoryBound(to:UInt8.self),width:r.width,height:r.height,bytesPerRow:CVPixelBufferGetBytesPerRow(frame))
      observations.append(observation);hands.append(try encoder.frameHand(observation,image:image))
     }
    }
   }
'''+s[b:];p.write_text(s)
with (O/'prefill_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(O/'prefill_probe')]+[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','main.swift']],stdout=f,stderr=subprocess.STDOUT,check=True)
