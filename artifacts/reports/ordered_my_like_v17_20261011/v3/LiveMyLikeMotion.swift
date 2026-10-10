import Foundation

/// Experiment only: original timestamps, same-hand ordered closure; unknown never means MY.
enum LiveMyLikeMotion {
    static let mode = ProcessInfo.processInfo.environment["MOTION"] ?? "on"
    static var logged: Set<String> = []
    static var events: [[String:Any]] = []
    struct Row {
        var t: Double, slot: Int, pinch: Double, straight: Double, size: Double, px: Double, py: Double
        var flatValid: Bool
    }
    static func distance(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double {
        let d=a-b; return sqrt(d.x*d.x+d.y*d.y)
    }
    static func features(_ obs: [LiveObservation]) -> [Row] {
        var body: [SIMD2<Double>]?; var seen = -Double.infinity; var rows: [Row] = []
        for o in obs {
            func point(_ p: LivePoint) -> SIMD2<Double> { SIMD2(Double(p.x)*Double(o.width), Double(p.y)*Double(o.height)) }
            if o.body.count>=2 && o.body[0].c>=0.3 && o.body[1].c>=0.3 { body=[point(o.body[0]),point(o.body[1])];seen=o.seconds }
            guard let b=body, o.seconds-seen<=1.5 else { continue }
            let sw=distance(b[0],b[1]),mid=(b[0]+b[1])/2
            for (slot,hand) in [o.left,o.right].enumerated() {
                guard let hand, [0,4,9,12].allSatisfy({hand.points[$0].c>=0.3}) else { continue }
                let xy=hand.points.map(point);let size=distance(xy[9],xy[0]);guard size>=5 && sw>=10 else { continue }
                let required=[0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                let flatValid=required.allSatisfy{hand.points[$0].c>=0.3}
                var straight=0.0
                if flatValid {
                    straight=[(5,8),(9,12),(13,16),(17,20)].map { m,t -> Double in
                        let chain=(m..<t).map{distance(xy[$0+1],xy[$0])}.reduce(0,+)
                        return chain>0 ? distance(xy[t],xy[m])/chain : 0
                    }.min()!
                }
                let palm=[0,5,9,13,17].map{xy[$0]}.reduce(SIMD2<Double>(repeating:0),+)/5
                rows.append(Row(t:o.seconds,slot:slot,pinch:distance(xy[4],xy[12])/size,straight:straight,size:size,px:(palm.x-mid.x)/sw,py:(palm.y-mid.y)/sw,flatValid:flatValid))
            }
        }
        return rows
    }
    static func assess(_ rows: [Row], start: Double, end: Double, available: Double, nextStart: Double?) -> (String,String) {
        let limit=min(end+0.5,nextStart.map{$0-1e-5} ?? Double.infinity)
        guard limit>=end+0.08 && available>=limit-1e-6 else { return ("unknown","incomplete_future") }
        var hands: [String]=[]
        for slot in 0..<2 {
            let rr=rows.filter{$0.slot==slot && $0.t>=start-0.15-1e-6 && $0.t<=limit+1e-6}
            let contacts=rr.filter{$0.t<=end && abs($0.px)<=0.6 && $0.py>=(-0.15) && $0.py<=1.0}
            if contacts.isEmpty { continue }
            let anchors=contacts.filter{$0.flatValid && $0.straight>=0.97 && $0.pinch>=0.8}
            var closing=false
            for a in contacts {
                let tail=rr.filter{$0.t>=a.t}
                for (i,b) in tail.enumerated() where b.t-a.t>=0.05 && b.t-a.t<=0.6 {
                    if b.pinch<=0.55 && a.pinch-b.pinch>=0.35 {
                        closing=true
                        let moved=hypot(b.px-a.px,b.py-a.py)>=0.15 || b.size/a.size>=1.15
                        let contiguous=i<1 || (1...i).allSatisfy{tail[$0].t-tail[$0-1].t<=0.150001}
                        if moved && contiguous { return ("LIKE","ordered_closure_and_departure") }
                    }
                }
            }
            guard anchors.count>=2, anchors.last!.t-anchors.first!.t>=0.05 else { hands.append("unknown");continue }
            let tail=rr.filter{$0.t>=anchors.first!.t}
            let complete=tail.count>=4 && tail.last!.t>=limit-0.10 && (1..<tail.count).allSatisfy{tail[$0].t-tail[$0-1].t<=0.150001}
            hands.append(complete && tail.allSatisfy{$0.pinch>=0.70} && !closing ? "MY" : "unknown")
        }
        return !hands.isEmpty && hands.allSatisfy{$0=="MY"} ? ("MY","observed_flat_contact_without_pinch") : ("unknown","insufficient_or_ambiguous_motion")
    }
}
