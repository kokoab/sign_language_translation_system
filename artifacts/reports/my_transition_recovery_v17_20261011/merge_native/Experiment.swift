import Foundation
// Report-local ablation controls. No production source imports this file.
enum Experiment {
    static var pre = 0.1, post = 0.1
    static var support=0.7, nextGap=0.4
    static var agreementFloor=2.0
    static var onsetFloor:Float=0.4
    static var contactRows=3, contactSeconds=0.08, followHorizon=0.5
    static var touchLimit=0.06, maxContact=0.35
    static func stableTouch(_ features:[LiveMyLikeMotion.Row],start:Double,end:Double)->Bool {
        let rows=features.filter{$0.t>=start-0.05 && $0.t<=end && $0.flatValid && $0.straight>=0.97 && $0.pinch>=0.8 && abs($0.px)<=0.6 && $0.py>=(-0.15) && $0.py<=1.0}
        return (0..<2).contains {slot in
            let rr=rows.filter{$0.slot==slot}
            guard rr.count>=contactRows,rr.last!.t-rr.first!.t>=contactSeconds-1e-6 else{return false}
            let dx=rr.map(\.px).max()!-rr.map(\.px).min()!,dy=rr.map(\.py).max()!-rr.map(\.py).min()!
            guard sqrt(dx*dx+dy*dy)<=touchLimit else{return false}
            let px=rr.map(\.px).reduce(0,+)/Double(rr.count),py=rr.map(\.py).reduce(0,+)/Double(rr.count)
            let all=features.filter{$0.slot==slot}
            guard var lo=all.firstIndex(where:{$0.t==rr.first!.t}),var hi=all.firstIndex(where:{$0.t==rr.last!.t}) else{return false}
            func near(_ r:LiveMyLikeMotion.Row)->Bool {
                r.flatValid && r.straight>=0.97 && r.pinch>=0.8 && hypot(r.px-px,r.py-py)<=touchLimit
            }
            while lo>0 && all[lo].t-all[lo-1].t<=0.175001 && near(all[lo-1]) {lo-=1}
            while hi+1<all.count && all[hi+1].t-all[hi].t<=0.175001 && near(all[hi+1]) {hi+=1}
            return all[hi].t-all[lo].t<=maxContact+1e-6
        }
    }
    static func distinctTouch(_ features:[LiveMyLikeMotion.Row],start:Double,end:Double,nextStart:Double,nextEnd:Double)->Bool {
        (0..<2).contains {slot in
            let hand=features.filter{$0.slot==slot}
            guard stableTouch(hand,start:start,end:end) else{return false}
            let following=hand.filter{$0.t>=nextStart && $0.t<=nextStart+followHorizon}
            let changed=following.filter{($0.flatValid && $0.straight<0.94) || $0.pinch<0.75}
            return changed.count>=2 && changed.last!.t-changed.first!.t>=0.05 && Double(changed.count)/Double(max(1,following.count))>=0.6
        }
    }
    static var contextScoreFloor=0.995
    static var successorFloor=0.85, successorGap=0.6
    static var pairFloor=0.0
    static var mix:Float=1
    static var maxDuration=0.35, floor=0.9
    static var rank=3, flat=false
    static var strong=0.6
    static var policy="calibrated"
    static func refine(_ base:[Float], adapted:[Float], labels:[String])->[Float]? {
        guard let my=labels.firstIndex(of:"MY"),let like=labels.firstIndex(of:"LIKE") else {return nil}
        let q=LiveSegmental.logSoftmax(base).map {exp($0)},p=LiveSegmental.logSoftmax(adapted).map {exp($0)}
        let top=q.indices.max(by:{q[$0]<q[$1]})!
        guard p.indices.max(by:{p[$0]<p[$1]})==my,p[my]>=floor,q[my]+q[like]>=pairFloor else{return nil}
        let order=q.indices.sorted {q[$0]>q[$1]}
        // Already-MY spans need either MY/LIKE ambiguity or strong independent MY confirmation.
        guard top != my || order.prefix(3).contains(like) || p[my]>=agreementFloor else{return nil}
        let target=min(max(p[my],top==like ? q[like] : 0),0.999999)
        guard target>q[my] else{return nil}
        let lp=LiveSegmental.logSoftmax(base)
        let mass=Double(base[my])-lp[my],adjustment=log((1-target)/max(1-q[my],1e-9))
        return base.enumerated().map {index,value in index==my ? Float(mass+log(target)) : value+Float(adjustment)}
    }
    static func plausibleMY(_ features:[LiveMyLikeMotion.Row],start:Double,end:Double)->Bool {
        let rows=features.filter { $0.t>=start-0.1 && $0.t<=end && $0.flatValid }
        if rows.count<2 {return true} // Missing landmarks cannot establish a bent-hand veto.
        return rows.contains { $0.straight>=0.97 && $0.pinch>=0.8 && abs($0.px)<=0.6 && $0.py>=(-0.15) && $0.py<=1.0 }
    }
    static func eligible(_ features:[LiveMyLikeMotion.Row],start:Double,end:Double)->Bool {
        guard end-start<=maxDuration+1e-6 else {return false}
        if !plausibleMY(features,start:start,end:end) {return false}
        if !flat {return true}
        let rows=features.filter { $0.t>=start-0.1 && $0.t<=end && $0.flatValid && $0.straight>=0.97 && $0.pinch>=0.8 && abs($0.px)<=0.6 && $0.py>=(-0.15) && $0.py<=1.0 }
        return (0..<2).contains { slot in let rr=rows.filter { $0.slot==slot }; return rr.count>=2 && rr.last!.t>=end-0.100001 && rr.last!.t-rr.first!.t>=0.05 }
    }
    static var audit = true
    static var events: [[String:Any]] = []
}
