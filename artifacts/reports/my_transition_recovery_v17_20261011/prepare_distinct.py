from pathlib import Path
import shutil,subprocess,json
D=Path(__file__).resolve().parent;d=D/'distinct_native';shutil.copytree(D/'contact_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('    static var pairFloor','''    static func distinctTouch(_ features:[LiveMyLikeMotion.Row],start:Double,end:Double,nextStart:Double,nextEnd:Double)->Bool {
        (0..<2).contains {slot in
            let hand=features.filter{$0.slot==slot}
            guard stableTouch(hand,start:start,end:end) else{return false}
            let following=hand.filter{$0.t>=nextStart && $0.t<=min(nextEnd,nextStart+0.35)}
            let changed=following.filter{($0.flatValid && $0.straight<0.94) || $0.pinch<0.75}
            return changed.count>=2 && changed.last!.t-changed.first!.t>=0.05 && Double(changed.count)/Double(max(1,following.count))>=0.6
        }
    }
    static var pairFloor''');p.write_text(s)
p=d/'LiveReelDecoder.swift';s=p.read_text().replace('seconds($0.0.s)-seconds(span.e)<=0.400001}) else{continue}', 'seconds($0.0.s)-seconds(span.e)<=0.400001 && Experiment.distinctTouch(completionFeatures,start:seconds(span.s),end:seconds(span.e),nextStart:seconds($0.0.s),nextEnd:seconds($0.0.e))}) else{continue}');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
for kind,main in [('video',d/'main.swift'),('validation',D/'contact_validation/main.swift')]:
 with (D/('distinct_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/('distinct_'+kind+'_probe'))]+files+[str(main)],stdout=f,stderr=subprocess.STDOUT,check=True)
print('Compiled short stationary contact plus visibly changed following handshape')
