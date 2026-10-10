"""Build an input-only native extraction probe; no word or boundary model is run."""
from pathlib import Path
import shutil,subprocess,json,hashlib
O=Path(__file__).resolve().parent;P=O.parent/'clean_lineage_recovery_v17_20261011';d=O/'native';d.mkdir(exist_ok=True)
pins={}
for name in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift']:
 p=P/'control'/name;shutil.copy2(p,d/name);pins[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
s=(P/'control/main.swift').read_text().split('func runVideos(')[0]
# Reject transformations rather than silently generating incorrectly oriented inputs.
s=s.replace('if !transform.isIdentity { FileHandle.standardError.write("warning: non-identity transform in \\(url.lastPathComponent)\\n".data(using: .utf8)!) }','if !transform.isIdentity { throw LiveReelError.input("Source requires explicit display transform") }')
s+='''
let a=CommandLine.arguments
LiveModelLocator.packageDirectory=URL(fileURLWithPath:a[1])
LiveHandEncoder.packageName="MobileCLIP2S0ImageEncoderV17FP16"
let encoder=try LiveHandEncoder(units:.all)
let cases=try readJSON(a[2]) as! [[String:Any]]
var rows:[[String:Any]]=[]
var arrays=[[Float]](repeating:[],count:4)
for item in cases {
 for offset in [0.0,1.0/60,1.0/30] {
  for phase in 0..<8 {
   let vision=LiveVision();vision.setDiagnosticPhase(phase)
   let scaler=LiveDetectionScaler()
   let frames=try VideoFrames(URL(fileURLWithPath:item["path"] as! String))
   var deadline=offset
   var observations:[LiveObservation]=[],hands:[LiveHandFrame]=[]
   while let source=frames.next() {
    let seconds=frames.lastPTS
    if seconds > (item["limit"] as! Double) { break }
    if seconds+1e-6 < deadline {continue}
    deadline=max(deadline+0.05,seconds)
    try autoreleasepool {
     let frame=limited(source,1280)
     let w=CVPixelBufferGetWidth(frame),h=CVPixelBufferGetHeight(frame)
     let observation=try vision.observe(detection:try scaler.resize(frame),width:w,height:h,seconds:seconds)
     CVPixelBufferLockBaseAddress(frame,.readOnly)
     defer {CVPixelBufferUnlockBaseAddress(frame,.readOnly)}
     let image=LiveBGRAImage(base:CVPixelBufferGetBaseAddress(frame)!.assumingMemoryBound(to:UInt8.self),width:w,height:h,bytesPerRow:CVPixelBufferGetBytesPerRow(frame))
     observations.append(observation);hands.append(try encoder.frameHand(observation,image:image))
    }
   }
   for event in item["events"] as! [[String:Any]] {
    let core=event["core"] as! [Double]
    // Expand context at one edge; never trim away the verified semantic core.
    for (view,extra) in [0.0,0.05,0.10].enumerated() {
     let lo=core[0]-0.1-extra,hi=core[1]+0.1
     let ids=observations.indices.filter {lo-1e-9<=observations[$0].seconds && observations[$0].seconds<hi-1e-9}
     var row:[String:Any]=["id":event["id"]!,"phase":phase,"capture_offset":offset,"view":view,
       "core":core,"requested_interval":[lo,hi],"selected_seconds":ids.map{observations[$0].seconds},
       "frames":ids.count,"valid":false]
     let selected=ids.map{observations[$0]}
     row["core_hand_frames"]=ids.filter{ observations[$0].seconds>=core[0]-1e-9 && observations[$0].seconds<core[1]-1e-9 && (observations[$0].left != nil || observations[$0].right != nil) }.count
     if let trimmed=LiveSpanLandmarks.build(selected) {
      let retained=Array(ids[trimmed.trimStart..<trimmed.trimEnd])
      row["retained_seconds"]=retained.map{observations[$0].seconds}
      let length=retained.count
      let sample=(0..<16).map{retained[Int((Double($0)*Double(length-1)/15).rounded(.toNearestOrEven))]}
      row["core_crop_positions"]=sample.filter{observations[$0].seconds>=core[0]-1e-9 && observations[$0].seconds<core[1]-1e-9 && hands[$0].valid.contains(where:{$0>0.5})}.count
     }
     if ids.count>=4,let x=LiveSpanRecognizer.inputs(selected,ids.map{hands[$0]}) {
      row["valid"]=true;row["array_index"]=arrays[0].count/(32*61*5)
      arrays[0]+=x.landmarks;arrays[1]+=x.embeddings;arrays[2]+=x.valid;arrays[3]+=x.boxes
     }
     rows.append(row)
    }
   }
  }
 }
 print(item["id"]!,"extracted");fflush(stdout)
}
for (i,x) in arrays.enumerated() {
 try x.withUnsafeBufferPointer {Data(buffer:$0)}.write(to:URL(fileURLWithPath:a[3]+".\\(i).bin"))
}
try writeJSON(["rows":rows,"classification_run":false,"encoding":"float32_little_endian","shapes":[[32,61,5],[16,3,512],[16,3],[16,3,4]]],a[3]+".json")
'''
(d/'main.swift').write_text(s)
with (O/'native_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(O/'native_probe')]+[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','main.swift']],stdout=f,stderr=subprocess.STDOUT,check=True)
(O/'native_source_pins.json').write_text(json.dumps(pins,indent=2)+'\n')
print('Native input-only probe built')
