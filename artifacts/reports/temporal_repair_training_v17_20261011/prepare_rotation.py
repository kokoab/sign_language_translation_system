from pathlib import Path
import shutil,subprocess
O=Path(__file__).resolve().parent;d=O/'native_oriented';d.mkdir(exist_ok=True)
for p in (O/'native_cached_fast').glob('*.swift'):shutil.copy2(p,d/p.name)
p=d/'main.swift';s=p.read_text().replace('    let fps: Double','    let fps: Double\n    let rotate180: Bool\n    let displayTransform: CGAffineTransform',1)
old='        if !transform.isIdentity { throw LiveReelError.input("Source requires explicit display transform") }'
new='''        rotate180 = abs(transform.a+1)<1e-6 && abs(transform.d+1)<1e-6 && abs(transform.b)<1e-6 && abs(transform.c)<1e-6
        displayTransform=transform
        if !transform.isIdentity && !rotate180 { throw LiveReelError.input("Unsupported display transform") }'''
assert old in s;s=s.replace(old,new,1)
old='        return CMSampleBufferGetImageBuffer(sample)'
new='''        guard let raw=CMSampleBufferGetImageBuffer(sample) else { return nil }
        let frame: CVPixelBuffer
        if rotate180 {
            let w=CVPixelBufferGetWidth(raw),h=CVPixelBufferGetHeight(raw)
            precondition(abs(displayTransform.tx-Double(w))<1e-3 && abs(displayTransform.ty-Double(h))<1e-3,"Unsupported rotated translation")
            var target: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault,w,h,kCVPixelFormatType_32BGRA,nil,&target)
            guard let out=target else { return nil }
            CVPixelBufferLockBaseAddress(raw,.readOnly);CVPixelBufferLockBaseAddress(out,[])
            let src=CVPixelBufferGetBaseAddress(raw)!.assumingMemoryBound(to:UInt32.self),dst=CVPixelBufferGetBaseAddress(out)!.assumingMemoryBound(to:UInt32.self)
            let si=CVPixelBufferGetBytesPerRow(raw)/4,di=CVPixelBufferGetBytesPerRow(out)/4
            for y in 0..<h { for x in 0..<w { dst[y*di+x]=src[(h-1-y)*si+(w-1-x)] } }
            CVPixelBufferUnlockBaseAddress(out,[]);CVPixelBufferUnlockBaseAddress(raw,.readOnly)
            frame=out
        } else { frame=raw }
        if lastPTS==0,let path=ProcessInfo.processInfo.environment["DUMP_FIRST_FRAME"] {
            CVPixelBufferLockBaseAddress(frame,.readOnly)
            let stride=CVPixelBufferGetBytesPerRow(frame),h=CVPixelBufferGetHeight(frame)
            try! Data(bytes:CVPixelBufferGetBaseAddress(frame)!,count:stride*h).write(to:URL(fileURLWithPath:path+".bin"))
            try! writeJSON(["width":CVPixelBufferGetWidth(frame),"height":h,"stride":stride],path+".json")
            CVPixelBufferUnlockBaseAddress(frame,.readOnly)
        }
        return frame'''
assert old in s;s=s.replace(old,new,1);p.write_text(s)
with (O/'oriented_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(O/'oriented_probe')]+[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','main.swift']],stdout=f,stderr=subprocess.STDOUT,check=True)
