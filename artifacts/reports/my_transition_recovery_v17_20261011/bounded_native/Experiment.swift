import Foundation
// Report-local ablation controls. No production source imports this file.
enum Experiment {
    static var pre = 0.1, post = 0.1
    static var maxDuration=0.35, floor=0.9
    static var rank=3, flat=false
    static var policy="pair"
    static func eligible(_ observations:[LiveObservation],start:Double,end:Double)->Bool {
        guard end-start<=maxDuration+1e-6 else {return false}
        if !flat {return true}
        let rows=LiveMyLikeMotion.features(observations).filter { $0.t>=start-0.1 && $0.t<=end && $0.flatValid && $0.straight>=0.97 && $0.pinch>=0.8 && abs($0.px)<=0.6 && $0.py>=(-0.15) && $0.py<=1.0 }
        return (0..<2).contains { slot in let rr=rows.filter { $0.slot==slot }; return rr.count>=2 && rr.last!.t>=end-0.100001 && rr.last!.t-rr.first!.t>=0.05 }
    }
    static var audit = true
    static var events: [[String:Any]] = []
}
