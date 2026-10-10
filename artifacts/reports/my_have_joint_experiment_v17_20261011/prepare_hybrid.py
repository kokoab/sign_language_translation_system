from pathlib import Path
P=Path(__file__).resolve().parent;d=P/'hybrid';d.mkdir(exist_ok=True)
s=(P/'candidate_ordered/LiveReelDecoder.swift').read_text()
s=s.replace('    private var lastSeconds = -Double.infinity','''    private lazy var myFallback = try? LiveSpanRecognizer(name: "SpanRecognizerV17LocalALettersB8FP16", labels: recognizer.labels, letterThreshold: 0.5, units: .all)
    private var lastSeconds = -Double.infinity''')
needle='        let wordCount = recognizer.labels.count\n'
repl=needle+'''        if mode != .letters, let my = recognizer.labels.firstIndex(of: "MY"), let like = recognizer.labels.firstIndex(of: "LIKE") {
            for j in logits.indices {
                let order = (0..<wordCount).sorted { logits[j][$0] > logits[j][$1] }
                guard order[0] == like, order.prefix(3).contains(my), keys[j].e - keys[j].s + 1 >= 6,
                      ExperimentFlatPalm.accepts(batch[j].landmarks), let fallback = myFallback else { continue }
                let old = try fallback.logits([batch[j]])[0]
                guard (0..<wordCount).max(by: { old[$0] < old[$1] }) == my else { continue }
                logits[j].swapAt(my, like)
                print("HYBRID_RESCUE", keys[j].s, keys[j].e)
            }
        }
'''
assert needle in s;s=s.replace(needle,repl,1)
s += '''
// Experiment only: a second-model vote must agree with a static open-palm check.
private enum ExperimentFlatPalm {
    static func percentile(_ a: [Double], _ q: Double) -> Double {
        let s = a.sorted(), p = q * Double(a.count - 1), i = Int(p)
        return s[i] + (s[min(i + 1, s.count - 1)] - s[i]) * (p - Double(i))
    }
    static func accepts(_ x: [Float]) -> Bool {
        var used = 0
        for st in [0, 21] {
            let frames = (0..<32).filter { t in (0..<21).filter { x[(t*61+st+$0)*5+4] > 0.3 }.count >= 15 }
            if frames.count < 3 { continue }; used += 1
            func point(_ t: Int, _ j: Int) -> SIMD2<Double> { let i = (t*61+st+j)*5; return SIMD2(Double(x[i]), Double(x[i+1])) }
            func distance(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double { simd_length(a-b) }
            let size = frames.map { distance(point($0,9),point($0,0))+1e-6 }
            var straight:[Double]=[], middle:[Double]=[], pinch:[Double]=[], droop:[Double]=[]
            for (k,t) in frames.enumerated() {
                let lengths = [5,9,13,17].map { m -> Double in
                    let chain = (m..<m+3).map { distance(point(t,$0+1),point(t,$0)) }.reduce(0,+)
                    return distance(point(t,m+3),point(t,m))/(chain+1e-6)
                }
                straight.append(lengths.min()!); middle.append(lengths[1])
                pinch.append(distance(point(t,4),point(t,12))/size[k])
                droop.append(([8,12,16].map { point(t,$0).y }.reduce(0,+)/3-point(t,0).y)/size[k])
            }
            let motion = frames.map { a in frames.map { b in distance(point(a,0),point(b,0)) }.max()! }.max()! / percentile(size,0.5)
            if percentile(straight,0.5) < 0.98 || percentile(middle,0.25) < 0.98 || percentile(pinch,0.1) < 0.8 || percentile(pinch,0.9)-percentile(pinch,0.1) > 0.5 || motion > 0.75 || percentile(droop,0.9)-percentile(droop,0.1) > 0.5 { return false }
        }
        return used > 0
    }
}
'''
(d/'LiveReelDecoder.swift').write_text(s)
