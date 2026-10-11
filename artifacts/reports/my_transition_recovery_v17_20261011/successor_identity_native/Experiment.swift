import Foundation
// Report-local ablation controls. No production source imports this file.
enum Experiment {
    static var pre = 0.1, post = 0.1
    static var support=0.7, nextGap=0.4
    static var agreementFloor=2.0
    static var onsetFloor:Float=0.4
    static var successorFloor=0.95
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
