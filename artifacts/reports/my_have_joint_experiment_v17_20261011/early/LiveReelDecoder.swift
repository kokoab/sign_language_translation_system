// Streaming transition-aware segmental decoder (port of active/v17/segmental_runtime_v17.py).
//
// Per frame: observation -> hand-crop embeddings (once) -> boundary BIO (6-frame lookahead) ->
// candidate spans -> span recognizer -> semi-Markov sign/rest DP -> fixed-lag commit with an early
// identity commit for the still-open segment. DualRuntime adds a letters-only decoder.

import Foundation
import simd

private let UNK = 0, O = 1, B = 2, I = 3

/// A committed word. A class on purpose: an early-shown word is later updated in place with the
/// sign's real end, exactly as the Python dict is.
final class LiveWord {
    var gloss: String
    var startFrame: Int, endFrame: Int, commitFrame: Int
    var startSeconds: Double, endSeconds: Double, commitSeconds: Double
    var score: Double
    var early: Bool
    var closed = false
    var spelled: String?
    var letters: [String] = []
    var computeSeconds: Double = 0

    init(gloss: String, startFrame: Int, endFrame: Int, commitFrame: Int, startSeconds: Double,
         endSeconds: Double, commitSeconds: Double, score: Double, early: Bool) {
        self.gloss = gloss; self.startFrame = startFrame; self.endFrame = endFrame; self.commitFrame = commitFrame
        self.startSeconds = startSeconds; self.endSeconds = endSeconds; self.commitSeconds = commitSeconds
        self.score = score; self.early = early
    }

    func copy() -> LiveWord {
        let w = LiveWord(gloss: gloss, startFrame: startFrame, endFrame: endFrame, commitFrame: commitFrame,
                         startSeconds: startSeconds, endSeconds: endSeconds, commitSeconds: commitSeconds,
                         score: score, early: early)
        w.closed = closed; w.spelled = spelled; w.letters = letters; w.computeSeconds = computeSeconds
        return w
    }

    var json: [String: Any] {
        var value: [String: Any] = [
            "gloss": gloss, "start_frame": startFrame, "end_frame": endFrame, "commit_frame": commitFrame,
            "start_seconds": startSeconds, "end_seconds": endSeconds, "commit_seconds": commitSeconds,
            "score": score, "early": early, "compute_seconds": computeSeconds,
        ]
        if let spelled { value["spelled"] = spelled; value["letters"] = letters }
        return value
    }
}

struct LivePreview {
    let gloss: String
    let score: Double
    let emit: Bool
    let top3: [(gloss: String, score: Double)]
}

struct LiveDecoderConfig {
    var wR = 2.0, wA = 0.25, wB = 0.5, c = -1.5, logTheta = log(0.5)
    var duplicateGap = 20
    var lag = 1, contextFrames = 2, earlyK = 3, earlyQ = 0.9, softExtra = 2
    var hard: Float = 0.5
    var earlyMin = 4
    var letterGap = 6
    /// Shortest span (frames, 20 Hz) the decoder may commit; stream.min_span_frames in
    /// segmental_runtime_v17.py.
    var minSpanFrames = 3
    /// Rest frames a candidate span may contain (stream.max_inner_o).
    var maxInnerO = 3
    /// Floor on P(B|I) inside a sign segment (decoder.inner_floor); nil = 1e-6 (off).
    var innerFloor: Float? = nil
    /// HAVE vs ANGRY handshape check (LivePairCheck); word decoders only.
    var haveAngryCheck = false
    /// Preserve accepted HAVE contact evidence until its completed transition.
    var haveContactClosure = false
    /// TIRED is only accepted when the bent hands roll down on the chest: top word TIRED, HAVE in the top 3 and
    /// fingertip droop below this (hand sizes) -> HAVE. nil: off. (segmental_runtime_v17.tired_rescue)
    var tiredRescueDroop: Double? = nil
    /// Fist letters (A E M N S T): letter head x hand geometry (fist_geometry_v17.py).
    var fistGeometry = true
    var earlyUnsafe: Set<String> = ["ASK", "DIFFERENT", "EAT", "FATHER", "GO", "GOOD", "HAVE", "HEAR", "I",
                                    "IMPORTANT", "MAKE", "MOTHER", "MY", "WHERE", "YOU"]

    /// stream_config_v3_final.json
    static let v3 = LiveDecoderConfig()
    /// Word mode: a fast movement inside one sign (the boundary student's rest-like dip) no longer splits the
    /// sign. Spans may contain up to 8 rest frames, P(B|I) inside a segment is floored at 0.5, and commits wait
    /// 6 frames (lag) so the second half of a sign is seen before the first half is committed. Phone word view,
    /// installed v2 recognizer: user takes GO 7 -> 1, Citizen val correct 313 -> 328 / exact 264 -> 278,
    /// tuning WER 17.7 -> 18.6% (go_split_root_cause_v17_20261009). A 6-frame minimum span was rejected
    /// (tuning WER 27.4%: short signs in sentences are deleted).
    /// Commit threshold 0.4 with the fragment+reject recognizer (tiny sign pieces trained to a flat output):
    /// Citizen val exact 286 -> 307, extra 54 -> 32, tuning WER 15.5 -> 15.0% (go_split_root_cause_v17_20261009).
    static let v3Words: LiveDecoderConfig = {
        var c = LiveDecoderConfig(); c.maxInnerO = 8; c.innerFloor = 0.5; c.lag = 6; c.haveAngryCheck = true; c.logTheta = log(0.4); c.tiredRescueDroop = 1.0; c.haveContactClosure = true
        let arm = ProcessInfo.processInfo.environment["ARM", default: "current"]
        if arm.hasPrefix("rollback") {
            c.haveContactClosure = false
            c.haveAngryCheck = arm == "rollback_geometry" || arm == "rollback_both"
            c.tiredRescueDroop = (arm == "rollback_roll" || arm == "rollback_both") ? 1.0 : nil
        }
        if ProcessInfo.processInfo.environment["MY_EARLY"] != nil { c.earlyUnsafe.remove("MY") }
        return c
    }()
}

/// HAVE (bent fingers together) vs ANGRY (spread claw): port of CropPair + HandshapePair (min) in
/// active/v17/segmental_runtime_v17.py. In a sentence HAVE leaves the chest upward into the next sign,
/// which the recognizer reads as ANGRY's movement; the handshape decides. Two logistic regressions: hand-crop
/// embeddings (crop_have_angry_v17.json) and canonicalised per-frame hand landmarks (handshape_have_angry_v17.json).
/// When HAVE or ANGRY is in the top 2, their combined probability is re-shared by the weaker of the two scores.
/// Port of fingertip_droop / tired_rescue (active/v17/segmental_runtime_v17.py).
enum LiveTiredRescue {
    /// Fingertip (8, 12, 16) height minus wrist height, in hand sizes: range over the span, mean over visible hands.
    static func droop(_ raw: [[Float]]) -> Double? {
        var out: [Double] = []
        for st in [0, 21] {
            let frames = raw.filter { f in (0..<21).filter { f[(st + $0) * 5 + 4] >= 0.3 }.count >= 15 }
            guard frames.count >= 3 else { continue }
            let sizes = frames.map { f -> Double in
                let dx = Double(f[(st + 9) * 5] - f[st * 5]), dy = Double(f[(st + 9) * 5 + 1] - f[st * 5 + 1])
                return (dx * dx + dy * dy).squareRoot()
            }.sorted()
            let n = sizes.count
            let size = (n % 2 == 1 ? sizes[n / 2] : (sizes[n / 2 - 1] + sizes[n / 2]) / 2) + 1e-6
            let tips = frames.map { f -> Double in
                let tip = (Double(f[(st + 8) * 5 + 1]) + Double(f[(st + 12) * 5 + 1]) + Double(f[(st + 16) * 5 + 1])) / 3
                return (tip - Double(f[st * 5 + 1])) / size
            }
            out.append(tips.max()! - tips.min()!)
        }
        return out.isEmpty ? nil : out.reduce(0, +) / Double(out.count)
    }

    static func apply(_ word: [Float], tired: Int, have: Int, raw: [[Float]], threshold: Double, merge: Bool = false) -> [Float] {
        let order = word.indices.sorted { word[$0] > word[$1] }
        guard order.first == tired, order.prefix(3).contains(have), let d = droop(raw), d < threshold else { return word }
        var out = word
        if merge {
            let hi = max(word[tired], word[have])
            out[have] = hi + log(exp(word[tired] - hi) + exp(word[have] - hi))
            out[tired] = -1e9
        } else {
            out.swapAt(tired, have)
        }
        return out
    }
}

enum LiveHaveContact {
    static func supported(_ raw: [[Float]], start: Int) -> Bool {
        let core = Array(raw[start...])
        guard core.count >= 2 else { return false }
        let changed = (1..<core.count).contains { t in
            (0..<42).contains { j in abs(core[t][j*5]-core[t-1][j*5]) >= 1e-6 || abs(core[t][j*5+1]-core[t-1][j*5+1]) >= 1e-6 }
        }
        guard changed else { return false }
        func median(_ a: [Double]) -> Double { let b=a.sorted(); let n=b.count; return n%2 == 1 ? b[n/2] : (b[n/2-1]+b[n/2])/2 }
        var rises: [Double] = []
        for st in [0,21] {
            let before = raw[...start].filter { $0[st*5+4]>=0.3 && $0[(st+9)*5+4]>=0.3 }
            let after = core.filter { $0[st*5+4]>=0.3 && $0[(st+9)*5+4]>=0.3 }
            guard before.count>=2, after.count>=2 else { continue }
            let size = median(after.map { f in hypot(Double(f[(st+9)*5]-f[st*5]), Double(f[(st+9)*5+1]-f[st*5+1])) })
            guard size>=1e-3 else { continue }
            rises.append((median(before.prefix(3).map { Double($0[st*5+1]) })-median(after.suffix(3).map { Double($0[st*5+1]) }))/size)
        }
        return !(rises.count == 2 && rises.max()! < -0.5)
    }
}

final class LivePairCheck {
    private struct LR {
        let mean, scale, coef: [Double]; let intercept: Double
        init(_ url: URL) throws {
            let j = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
            mean = j["mean"] as! [Double]; scale = j["scale"] as! [Double]; coef = j["coef"] as! [Double]
            intercept = (j["intercept"] as! NSNumber).doubleValue
        }
        func p(_ x: [Double]) -> Double {
            var z = intercept
            for k in 0..<x.count { z += (x[k] - mean[k]) / scale[k] * coef[k] }
            return 1 / (1 + exp(-z))
        }
    }
    private let crop: LR, shape: LR

    init(crop: URL, shape: URL) throws { self.crop = try LR(crop); self.shape = try LR(shape) }

    /// Mean left+right view embedding ++ union view embedding over valid frames (16 x 3 x 512).
    func cropScore(_ x: LiveSpanInput) -> Double? {
        var hands = [Double](repeating: 0, count: 512), union = [Double](repeating: 0, count: 512)
        var nh = 0.0, nu = 0.0
        for f in 0..<16 {
            for v in 0..<3 where x.valid[f * 3 + v] > 0.5 {
                let base = (f * 3 + v) * 512
                if v < 2 { for k in 0..<512 { hands[k] += Double(x.embeddings[base + k]) }; nh += 1 }
                else { for k in 0..<512 { union[k] += Double(x.embeddings[base + k]) }; nu += 1 }
            }
        }
        guard nh + nu > 0 else { return nil }
        return crop.p(hands.map { $0 / max(nh, 1) } + union.map { $0 / max(nu, 1) })
    }

    /// Mean P(HAVE) of the 3 most HAVE-like (frame, hand) samples of the span's raw landmarks (61 x 5 each).
    func shapeScore(_ raw: [[Float]]) -> Double? {
        var ps: [Double] = []
        for f in raw {
            for (start, left) in [(0, true), (21, false)] {
                guard (0..<21).allSatisfy({ f[(start + $0) * 5 + 4] >= 0.3 && f[(start + $0) * 5 + 3] >= 0.5 }) else { continue }
                let w = SIMD2(Double(f[start * 5]), Double(f[start * 5 + 1]))
                let xy = (0..<21).map { SIMD2(Double(f[(start + $0) * 5]), Double(f[(start + $0) * 5 + 1])) - w }
                let v = xy[9], size = simd_length(v)
                guard size >= 1e-3 else { continue }
                let a = atan2(v.x, -v.y), c = cos(a), s = sin(a)
                var feat: [Double] = []
                for j in 1..<21 {
                    var x = (c * xy[j].x - s * xy[j].y) / size
                    let y = (s * xy[j].x + c * xy[j].y) / size
                    if left { x = -x }
                    feat += [x, y]
                }
                ps.append(shape.p(feat))
            }
        }
        guard ps.count >= 2 else { return nil }
        let top = ps.sorted().suffix(3)
        return top.reduce(0, +) / Double(top.count)
    }

    /// Word logits -> log-probabilities with HAVE/ANGRY re-shared by g (unchanged when the pair is not in the top 2).
    /// Rescue only (2026-10-10): only a top-1 ANGRY moves towards HAVE, and only when g >= 0.5; a HAVE the
    /// recognizer already chose is never demoted (low cue scores in a new setting had turned HAVE into ANGRY).
    static func reshare(_ word: [Float], have: Int, angry: Int, g: Double?) -> [Float] {
        guard let g else { return word }
        let lp = LiveSegmental.logSoftmax(word)
        let top2 = Set(lp.indices.sorted { lp[$0] > lp[$1] }.prefix(2))
        guard top2.contains(have) || top2.contains(angry) else { return word }
        guard lp.indices.max(by: { lp[$0] < lp[$1] }) == angry, g >= 0.5 else { return word }
        var p = lp.map { exp($0) }
        let mass = p[have] + p[angry]
        p[have] = mass * g; p[angry] = mass * (1 - g)
        return p.map { Float(log(min(max($0, 1e-12), 1))) }
    }
}

/// Fist-letter geometry (port of active/v17/fist_geometry_v17.py): a logistic regression on
/// canonicalised hand landmarks that re-shares the A E M N S T probability as head x geometry.
final class LiveFistGeometry {
    private let classes: [Int]              // letter indices (A=0) of the model's classes
    private let mean, scale, intercept: [Double]
    private let coef: [[Double]]

    init(url: URL) throws {
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        classes = (json["classes"] as! [String]).map { Int($0.unicodeScalars.first!.value) - 65 }
        mean = json["mean"] as! [Double]
        scale = json["scale"] as! [Double]
        intercept = json["intercept"] as! [Double]
        coef = json["coef"] as! [[Double]]
    }

    /// Span landmarks [32*61*5] -> 54 features (see features() in Python), or nil.
    static func features(_ lm: [Float]) -> [Double]? {
        func at(_ t: Int, _ node: Int, _ c: Int) -> Float { lm[(t * 61 + node) * 5 + c] }
        var left: Float = 0, right: Float = 0
        for t in 0..<32 { for j in 0..<21 { left += at(t, j, 3); right += at(t, 21 + j, 3) } }
        let s = left >= right ? 0 : 21
        var rows: [[Double]] = []
        for t in 8..<24 {
            guard (0..<21).filter({ at(t, s + $0, 3) > 0.5 }).count >= 18 else { continue }
            let wrist = SIMD2(at(t, s, 0), at(t, s, 1))
            let xy = (0..<21).map { SIMD2(at(t, s + $0, 0), at(t, s + $0, 1)) - wrist }
            let palm = simd_length(xy[9])
            guard palm >= 1e-4 else { continue }
            let ey = xy[9] / palm, ex = SIMD2(ey.y, -ey.x)
            var c = xy.map { SIMD2(simd_dot($0, ex), simd_dot($0, ey)) / palm }
            if c[5].x < c[17].x { c = c.map { SIMD2(-$0.x, $0.y) } }
            var row: [Double] = []
            for j in 1..<21 where j != 9 { row += [Double(c[j].x), Double(c[j].y)] }   // joint 9 is (0, 1) by construction
            for j in 5..<21 { row.append(Double(simd_length(c[j] - c[4]))) }
            rows.append(row)
        }
        guard rows.count >= 3 else { return nil }
        return (0..<rows[0].count).map { k in
            let v = rows.map { $0[k] }.sorted(), n = v.count
            return n % 2 == 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2
        }
    }

    /// letter logits (27) -> log-probabilities with the fist group re-shared as head^w x geometry.
    func refine(_ logits: [Float], landmarks: [Float], headWeight: Double = 0.25) -> [Float] {
        let m = Double(logits.max()!)
        var p = logits.map { exp(Double($0) - m) }
        let total = p.reduce(0, +)
        p = p.map { $0 / total }
        guard let top = p.indices.max(by: { p[$0] < p[$1] }), classes.contains(top),
              let f = Self.features(landmarks) else { return logits }
        var z = (0..<classes.count).map { k in
            intercept[k] + zip(coef[k], zip(f, zip(mean, scale))).reduce(0) { $0 + $1.0 * ($1.1.0 - $1.1.1.0) / $1.1.1.1 }
        }
        let zm = z.max()!
        z = z.map { exp($0 - zm) }
        let zs = z.reduce(0, +)
        let mass = classes.reduce(0) { $0 + p[$1] }
        var new = classes.enumerated().map { pow(p[$0.element], headWeight) * z[$0.offset] / zs }
        let ns = new.reduce(0, +)
        guard ns > 0 else { return logits }
        new = new.map { $0 / ns * mass }
        for (k, letter) in classes.enumerated() { p[letter] = new[k] }
        return p.map { Float(log(min(max($0, 1e-12), 1))) }
    }
}

/// Pair-only direction/crossing check, matching refine_letter_geometry in Python.
/// All four directions (G<->Q, R<->U); the earlier Q -> G "regressions" were mislabelled clips.
func liveRefineLetterGeometry(_ logits: [Float], raw: [[Float]]) -> [Float] {
    guard logits.count == 27, let top = logits.indices.max(by: { logits[$0] < logits[$1] }),
          [6, 16, 17, 20].contains(top), !raw.isEmpty else { return logits }
    var votes: [Int] = []
    for row in raw[(raw.count / 4)..<max(raw.count / 4 + 1, 3 * raw.count / 4)] {
        let left = (0..<21).reduce(Float(0)) { $0 + row[$1 * 5 + 4] }
        let right = (21..<42).reduce(Float(0)) { $0 + row[$1 * 5 + 4] }
        let start = left >= right ? 0 : 21
        guard [5, 8, 9, 12].allSatisfy({ row[(start + $0) * 5 + 4] >= 0.3 }) else { continue }
        func p(_ joint: Int) -> SIMD2<Double> {
            SIMD2(Double(row[(start + joint) * 5]), Double(row[(start + joint) * 5 + 1]))
        }
        func dot(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double { a.x * b.x + a.y * b.y }
        if top == 6 || top == 16 {
            let direction = p(8) - p(5), size = sqrt(dot(direction, direction))
            guard size >= 0.01 else { continue }
            let down = direction.y / size
            votes.append(down > 0.8 ? 16 : abs(down) < 0.4 ? 6 : -1)
        } else {
            let across = p(5) - p(9), size = dot(across, across)
            guard size >= 0.000025 else { continue }
            let crossing = dot(p(8) - p(12), across) / size
            votes.append(crossing < -0.2 ? 17 : crossing > 0.3 ? 20 : -1)
        }
    }
    guard votes.count >= 3 else { return logits }
    var result = logits
    for target in ((top == 6 || top == 16) ? [6, 16] : [17, 20]) where target != top {
        if Double(votes.filter { $0 == target }.count) / Double(votes.count) >= 0.8 {
            result.swapAt(top, target)
            break
        }
    }
    return result
}

// MARK: - Spans and DP

enum LiveSegmental {
    static func logSoftmax(_ x: [Float]) -> [Double] {
        let m = Double(x.max()!)
        let lse = m + log(x.reduce(0.0) { $0 + exp(Double($1) - m) })
        return x.map { Double($0) - lse }
    }

    private static func argmax(_ row: [Float]) -> Int {
        var best = 0
        for k in 1..<row.count where row[k] > row[best] { best = k }
        return best
    }

    static func candidateSpans(_ bio: [[Float]], bThreshold: Float = 0.2, minLen: Int = 3, maxLen: Int = 30,
                               maxInnerO: Int = 3) -> [SpanKey] {
        let p = bio.map { $0.map { exp($0) } }
        let arg = p.map(argmax)
        let n = p.count
        var starts = Set<Int>(), ends = Set<Int>()
        var t = 0
        while t < n {
            if arg[t] == B || arg[t] == I {
                var u = t
                while u + 1 < n && (arg[u + 1] == B || arg[u + 1] == I) { u += 1 }
                starts.insert(t); ends.insert(u)
                t = u + 1
            } else {
                t += 1
            }
        }
        for t in 0..<n {
            let pb = p[t][B]
            let left: Float = t > 0 ? p[t - 1][B] : 0
            let right: Float = t + 1 < n ? p[t + 1][B] : 0
            if pb >= bThreshold && pb >= left && pb >= right {
                starts.insert(t)
                if t > 0 { ends.insert(t - 1) }
            }
        }
        var oRun = [Int](repeating: 0, count: n)
        var running = 0
        for t in 0..<n { running += arg[t] == O ? 1 : 0; oRun[t] = running }
        var spans: [SpanKey] = []
        let sortedEnds = ends.sorted()
        for s in starts.sorted() {
            for e in sortedEnds {
                let length = e - s + 1
                if length < minLen || length > maxLen { continue }
                if oRun[e] - (s > 0 ? oRun[s - 1] : 0) > maxInnerO { continue }
                spans.append(SpanKey(s, e))
            }
        }
        return spans
    }

    static func denseSpans(_ bio: [[Float]], step: Int = 2, minLen: Int = 3, maxLen: Int = 20, active: Float = 0.3,
                           maxInnerO: Int = 3) -> [SpanKey] {
        let n = bio.count
        let on = bio.map { exp($0[B]) + exp($0[I]) >= active }
        var oRun = [Int](repeating: 0, count: n)
        var running = 0
        for t in 0..<n { running += on[t] ? 0 : 1; oRun[t] = running }
        let points = (0..<n).filter { on[$0] }
        guard !points.isEmpty else { return [] }
        var grid = Set(stride(from: 0, to: points.count, by: step).map { points[$0] })
        for t in points where t == 0 || !on[t - 1] { grid.insert(t) }
        for t in points where t == n - 1 || !on[t + 1] { grid.insert(t) }
        let sorted = grid.sorted()
        var spans: [SpanKey] = []
        for s in sorted {
            for e in sorted {
                let length = e - s + 1
                if length < minLen || length > maxLen { continue }
                if oRun[e] - (s > 0 ? oRun[s - 1] : 0) > maxInnerO { continue }
                spans.append(SpanKey(s, e))
            }
        }
        return spans
    }

    static func bothSpans(_ bio: [[Float]], minLen: Int = 3, maxInnerO: Int = 3) -> [SpanKey] {
        Array(Set(candidateSpans(bio, minLen: minLen, maxInnerO: maxInnerO))
            .union(denseSpans(bio, minLen: minLen, maxInnerO: maxInnerO))).sorted()
    }

    struct Segment {
        var start: Int, end: Int
        var gloss: String?
        var logQ: Double
    }

    /// Semi-Markov DP over rest frames and labelled segments (no duplicate merge: streaming).
    static func decode(_ bio: [[Float]], spans: [SpanKey], records: [SpanKey: [Float]], labels: [String],
                       config: LiveDecoderConfig) -> [Segment] {
        let n = bio.count
        var act = [Float](repeating: 0, count: n), rest = [Float](repeating: 0, count: n), logb = [Float](repeating: 0, count: n)
        for t in 0..<n {
            let p = bio[t].map { exp($0) }
            act[t] = log(min(max(p[B] + p[I], config.innerFloor ?? 1e-6), 1))
            rest[t] = log(min(max(p[O] + p[UNK], 1e-6), 1))
            logb[t] = log(min(max(p[B], 1e-6), 1))
        }
        var cact = [Double](repeating: 0, count: n + 1)
        var running: Float = 0         // numpy cumsum of float32 accumulates in float32
        for t in 0..<n { running += act[t]; cact[t + 1] = Double(running) }
        var byEnd = [Int: [(Int, Double, String?, Double)]]()
        for span in spans {
            guard let record = records[span] else { continue }
            let q = logSoftmax(record)
            var g = 0
            for k in 1..<q.count where q[k] > q[g] { g = k }
            let lq = q[g]
            let emit = lq >= config.logTheta
            let seg = config.wR * max(lq, config.logTheta) + config.wA * (cact[span.e + 1] - cact[span.s])
                + config.wB * Double(logb[span.s]) - config.c
            byEnd[span.e, default: []].append((span.s, seg, emit ? labels[g] : nil, lq))
        }
        var best = [Double](repeating: -.infinity, count: n + 1)
        var back = [(sign: Bool, start: Int, gloss: String?, lq: Double)](repeating: (false, 0, nil, 0), count: n + 1)
        best[0] = 0
        for t in 1...max(n, 1) where n > 0 {
            best[t] = best[t - 1] + config.wA * Double(rest[t - 1])
            back[t] = (false, t - 1, nil, 0)
            for (s, seg, gloss, lq) in byEnd[t - 1] ?? [] {
                let v = best[s] + seg
                if v > best[t] { best[t] = v; back[t] = (true, s, gloss, lq) }
            }
        }
        var out: [Segment] = []
        var t = n
        while t > 0 {
            let b = back[t]
            if !b.sign { t -= 1 } else { out.append(Segment(start: b.start, end: t - 1, gloss: b.gloss, logQ: b.lq)); t = b.start }
        }
        return out.reversed()
    }
}

struct SpanKey: Hashable, Comparable {
    let s: Int, e: Int
    init(_ s: Int, _ e: Int) { self.s = s; self.e = e }
    static func < (a: SpanKey, b: SpanKey) -> Bool { a.s != b.s ? a.s < b.s : a.e < b.e }
}

/// Raw 127-logit rows shared between the word and letter decoders (nil = unscorable span).
final class LiveSpanCache {
    var values: [SpanKey: [Float]?] = [:]
}

// MARK: - Streaming runtime

enum LiveDecoderMode { case words, letters, both }

final class LiveSegmentalRuntime {
    /// Loaded once from the bundle; nil if either model is absent.
    nonisolated(unsafe) static var pairCheck: LivePairCheck? = {
        guard let c = try? LiveModelLocator.resource("crop_have_angry_v17", "json"),
              let s = try? LiveModelLocator.resource("handshape_have_angry_v17", "json") else { return nil }
        return try? LivePairCheck(crop: c, shape: s)
    }()

    /// Loaded once from the bundle (fist_geometry_v17.json); nil if absent.
    nonisolated(unsafe) static var fist: LiveFistGeometry? = try? LiveFistGeometry(
        url: LiveModelLocator.resource("fist_geometry_v17", "json"))
    let config: LiveDecoderConfig
    let mode: LiveDecoderMode
    let labels: [String]
    let boundary: LiveBoundaryStream
    let recognizer: LiveSpanRecognizer
    let encoder: LiveHandEncoder
    var rawCache: LiveSpanCache?
    var timing: [String: Double] = [:]      // seconds accumulated per stage
    var onBIO: (([Float], Bool) -> Void)?    // parity hook: every boundary row (flag: end-of-stream flush)

    private(set) var off = 0, count = 0
    private(set) var obs: [LiveObservation] = []
    private(set) var hands: [LiveHandFrame] = []
    private var bio: [[Float]] = []
    private var bioOff = 0
    private var scores: [SpanKey: [Float]?] = [:]
    private(set) var origin = 0
    private var last: LiveWord?
    private var early: [Int: (String, Int)] = [:]
    private var shownEarly: [(Int, LiveWord)] = []
    private var pendingOutput: [LiveWord] = []
    private var lastSeconds = -Double.infinity
    private(set) var preview: LivePreview?

    /// Word mode with hidden letters (.both): letters compete for a span only when the recognizer's
    /// top word probability is below this gate; nil keeps the ungated letter sink. Mirrors
    /// SegmentalRuntime.letter_gate in active/v17/segmental_runtime_v17.py.
    let letterGate: Double?

    init(boundary: LiveBoundaryModel, recognizer: LiveSpanRecognizer, encoder: LiveHandEncoder,
         config: LiveDecoderConfig = .v3, mode: LiveDecoderMode, letterGate: Double? = nil) {
        self.letterGate = letterGate
        self.boundary = LiveBoundaryStream(model: boundary)
        self.recognizer = recognizer
        self.encoder = encoder
        self.config = config
        self.mode = mode
        labels = recognizer.labels + (mode == .words ? [] : Array(recognizer.letterClasses.dropLast()))
        reset()
    }

    var lookahead: Int { boundary.model.lookahead }

    func reset() {
        boundary.reset()
        off = 0; count = 0
        obs = []; hands = []; bio = []; bioOff = 0
        scores = [:]; origin = 0; last = nil
        early = [:]; shownEarly = []; pendingOutput = []
        lastSeconds = -.infinity
        preview = nil
    }

    private func tick(_ name: String, _ started: Date) -> Date {
        timing[name, default: 0] += Date().timeIntervalSince(started)
        return Date()
    }

    private func spanObs(_ s: Int, _ e: Int) -> [Int] {
        let lo = obs[s - off].seconds - 0.1, hi = obs[e - off].seconds + 0.1
        let from = max(0, s - off - 3), to = min(obs.count, e - off + 4)
        guard from < to else { return [] }
        return (from..<to).filter { lo - 1e-9 <= obs[$0].seconds && obs[$0].seconds <= hi + 1e-9 }
    }

    private func score(_ spans: [SpanKey]) throws {
        var t = Date()
        let shared = rawCache ?? LiveSpanCache()
        var cached: [(SpanKey, [Float]?)] = []
        var batch: [LiveSpanInput] = [], keys: [SpanKey] = []
        for key in spans {
            if let hit = shared.values[key] { cached.append((key, hit)); continue }
            let ids = spanObs(key.s, key.e)
            guard ids.count >= 4,
                  let x = LiveSpanRecognizer.inputs(ids.map { obs[$0] }, ids.map { hands[$0] }) else {
                scores[key] = .some(nil); shared.values[key] = .some(nil)
                continue
            }
            batch.append(x); keys.append(key)
        }
        t = tick("span_inputs", t)
        var logits = batch.isEmpty ? [] : try recognizer.logits(batch)
        _ = tick("recognizer", t)
        let wordCount = recognizer.labels.count
        if mode != .words, config.fistGeometry, let fist = Self.fist {
            // Fist letters: head x hand geometry, once per span for both decoders (shared cache).
            logits = zip(batch, logits).map { x, l in
                l.count > wordCount ? Array(l[..<wordCount]) + fist.refine(Array(l[wordCount...]), landmarks: x.landmarks) : l
            }
        }
        if mode != .letters, config.haveAngryCheck, let check = Self.pairCheck,
           let have = recognizer.labels.firstIndex(of: "HAVE"), let angry = recognizer.labels.firstIndex(of: "ANGRY") {
            logits = zip(zip(keys, batch), logits).map { kx, l in
                let (key, x) = kx
                let word = Array(l[..<wordCount])
                let lp = LiveSegmental.logSoftmax(word)
                let top2 = Set(lp.indices.sorted { lp[$0] > lp[$1] }.prefix(2))
                guard top2.contains(have) || top2.contains(angry) else { return l }
                let raw = spanObs(key.s, key.e).map { LiveFeatures.raw(obs[$0]) }
                let g: Double? = {
                    guard let a = check.cropScore(x), let b = check.shapeScore(raw) else { return nil }
                    return min(a, b)
                }()
                return LivePairCheck.reshare(word, have: have, angry: angry, g: g) + Array(l[wordCount...])
            }
        }
        if mode != .letters, let threshold = config.tiredRescueDroop,
           let tired = recognizer.labels.firstIndex(of: "TIRED"), let have = recognizer.labels.firstIndex(of: "HAVE") {
            logits = zip(keys, logits).map { key, l in
                let word = Array(l[..<wordCount])
                guard word.indices.max(by: { word[$0] < word[$1] }) == tired else { return l }
                let raw = spanObs(key.s, key.e).map { LiveFeatures.raw(obs[$0]) }
                return LiveTiredRescue.apply(word, tired: tired, have: have, raw: raw, threshold: threshold, merge: config.haveContactClosure && contactCore(key.s, key.e)) + Array(l[wordCount...])
            }
        }
        for (key, row) in zip(keys, logits) { shared.values[key] = .some(row) }
        var rows = Array(zip(keys, logits))
        for (key, row) in cached {
            if let row { rows.append((key, row)) } else { scores[key] = .some(nil) }
        }
        let words = recognizer.labels.count
        for (key, l) in rows {
            let wordPart = Array(l[0..<words])
            var letterPart = Array(l[words...])
            if mode != .words, let top = letterPart.indices.max(by: { letterPart[$0] < letterPart[$1] }),
               [6, 16, 17, 20].contains(top) {
                letterPart = liveRefineLetterGeometry(letterPart, raw: spanObs(key.s, key.e).map { LiveFeatures.raw(obs[$0]) })
            }
            switch mode {
            case .words:
                scores[key] = .some(wordPart)
            case .letters:
                let p = LiveSegmental.logSoftmax(letterPart).map { exp($0) }
                let keep = p.dropLast().max()! >= Double(recognizer.letterThreshold)
                let letters = p.dropLast().map { keep ? Float(log(min(max($0, 1e-9), 1))) : Float(-1e9) }
                scores[key] = .some([Float](repeating: -1e9, count: words) + letters)
            case .both:
                let wordLP = LiveSegmental.logSoftmax(wordPart)
                let p = LiveSegmental.logSoftmax(letterPart).map { exp($0) }
                let confidentWord = letterGate.map { exp(wordLP.max()!) >= $0 } ?? false
                if p.dropLast().max()! >= Double(recognizer.letterThreshold) && !confidentWord {
                    let none = log(max(p.last!, 1e-9))
                    scores[key] = .some(wordLP.map { Float($0 + none) } + p.dropLast().map { Float(log(min(max($0, 1e-9), 1))) })
                } else {
                    scores[key] = .some(wordLP.map { Float($0) } + [Float](repeating: -1e9, count: p.count - 1))
                }
            }
        }
    }

    /// Add one observation; `hand` reuses crops already encoded by the other decoder.
    func observe(_ observation: LiveObservation, image: LiveBGRAImage?, hand: LiveHandFrame? = nil) throws -> [LiveWord] {
        var words: [LiveWord] = []
        if observation.seconds - lastSeconds > 0.26 && count > 0 {
            words = try finish()
            reset()
        }
        lastSeconds = observation.seconds
        obs.append(observation)
        var t = Date()
        if let hand {
            hands.append(hand)
        } else if let image {
            hands.append(try encoder.frameHand(observation, image: image))
        } else {
            hands.append(LiveHandFrame())
        }
        t = tick("hand_crops_encode", t)
        count += 1
        if let row = try boundary.update(raw: LiveFeatures.raw(observation), seconds: observation.seconds) {
            bio.append(row)
            onBIO?(row, false)
        }
        _ = tick("boundary", t)
        return words + (try step(final: false))
    }

    /// Flush at stop/finish: remaining frames get boundary estimates with the future masked.
    func finish() throws -> [LiveWord] {
        guard count > 0 else { return [] }
        let pending = try boundary.flush()
        let missing = count - (bioOff + bio.count)
        if missing > 0 {
            for row in pending.suffix(missing) { bio.append(row); onBIO?(row, true) }
        }
        return try step(final: true)
    }

    private func trim() {
        let keep = max(origin - 40, 0)
        var drop = keep - off
        if drop > 0 { obs.removeFirst(drop); hands.removeFirst(drop); off = keep }
        drop = keep - bioOff
        if drop > 0 { bio.removeFirst(drop); bioOff = keep }
        scores = scores.filter { $0.key.s >= origin }
    }

    private func seconds(_ k: Int) -> Double { obs[min(max(k - off, 0), obs.count - 1)].seconds }

    private func word(_ gloss: String, _ start: Int, _ end: Int, _ T: Int, _ logQ: Double, early: Bool = false) -> LiveWord {
        LiveWord(gloss: gloss, startFrame: start, endFrame: end, commitFrame: T, startSeconds: seconds(start),
                 endSeconds: seconds(end), commitSeconds: seconds(T), score: exp(logQ), early: early)
    }

    private func contactCore(_ s: Int, _ e: Int) -> Bool {
        let begin = max(off, s-6)
        let raw = (begin...e).map { LiveFeatures.raw(obs[$0-off]) }
        return LiveHaveContact.supported(raw, start: s-begin)
    }

    /// Keep a completed contact from being absorbed into its exit or the next BIG.
    /// An earlier open word retains ownership; do not reinterpret its ending as HAVE.
    private func closeHaveContacts(_ candidates: [SpanKey], known: Int) -> [SpanKey] {
        guard let h = labels.firstIndex(of: "HAVE") else { return candidates }
        var spans = candidates
        let origin = self.origin
        // A contact sign can finish before its transition is classified as the next sign.
        // Carry accepted core evidence only to the first two-frame rest boundary.
        let rest = bio.map { exp($0[O]) + exp($0[UNK]) + exp($0[B]) }
        var closures: [(SpanKey, [Float])] = []
        for (key, value) in scores.sorted(by: { $0.key < $1.key }) {
            if shownEarly.contains(where: { $0.0 <= key.e && $0.1.gloss != "HAVE" && !$0.1.closed }) { continue }
            guard let row = value, key.s >= origin, key.e <= known,
                  row.indices.max(by: { row[$0] < row[$1] }) == h,
                  LiveSegmental.logSoftmax(row)[h] >= config.logTheta else { continue }
            guard contactCore(key.s, key.e) else { continue }
            let limit = min(known - 1, key.s + 29)
            if key.e + 1 > limit { continue }
            guard let gap = (key.e + 1...limit).first(where: { rest[$0-bioOff] >= 0.5 && rest[$0+1-bioOff] >= 0.5 }) else { continue }
            // The supporting core must still be in this active run.
            if key.s < gap-1 && (key.s..<gap-1).contains(where: { rest[$0-bioOff] >= 0.5 && rest[$0+1-bioOff] >= 0.5 }) { continue }
            closures.append((SpanKey(key.s, gap-1), row))

        }
        for (key,row) in closures {
            spans.removeAll { span in
                guard span.s <= key.e, span.e >= key.s else { return false }
                guard span.e <= key.e, let value = scores[span] ?? nil else { return true }
                return value.indices.max(by: { value[$0] < value[$1] }) != h
            }
            if let old = scores[key] ?? nil, LiveSegmental.logSoftmax(old)[h] > LiveSegmental.logSoftmax(row)[h] { continue }
            scores[key] = .some(row)
            spans.append(key)
        }
        spans = Array(Set(spans)).sorted()
        return spans
    }

    private func step(final: Bool) throws -> [LiveWord] {
        let T = count - 1
        let known = bioOff + bio.count - 1
        if known < origin || bio.isEmpty { return [] }
        let origin = self.origin
        let sub = Array(bio[(origin - bioOff)...(known - bioOff)])
        var spans = LiveSegmental.bothSpans(sub, minLen: config.minSpanFrames, maxInnerO: config.maxInnerO).map { SpanKey($0.s + origin, $0.e + origin) }
            .filter { final || $0.e + config.contextFrames <= T }
        if config.haveContactClosure, let h = labels.firstIndex(of: "HAVE") {
            let retained = scores.keys.filter { key in
                guard key.s >= origin, key.e <= known, let row = scores[key] ?? nil else { return false }
                return row.indices.max(by: { row[$0] < row[$1] }) == h
                    && LiveSegmental.logSoftmax(row)[h] >= config.logTheta
            }
            spans = Array(Set(spans).union(retained)).sorted()
        }
        let missing = spans.filter { scores[$0] == nil }
        if !missing.isEmpty { try score(missing) }
        if config.haveContactClosure { spans = closeHaveContacts(spans, known: known) }
        var local: [SpanKey: [Float]] = [:]
        for span in spans { if case .some(.some(let row)) = scores[span] { local[SpanKey(span.s - origin, span.e - origin)] = row } }
        let localSpans = spans.map { SpanKey($0.s - origin, $0.e - origin) }
        let segs = LiveSegmental.decode(sub, spans: localSpans, records: local, labels: labels, config: config)
        let horizon = final ? known : known - config.lag
        var committed: [LiveWord] = []
        preview = nil
        for g in segs {
            if g.end + origin <= horizon { continue }
            let st = g.start + origin, en = g.end + origin
            if shownEarly.contains(where: { abs(st - $0.0) <= 2 }) { break }
            if case .some(.some(let raw)) = scores[SpanKey(st, en)] {
                let p = LiveSegmental.logSoftmax(raw).map { exp($0) }
                let order = p.indices.sorted { p[$0] != p[$1] ? p[$0] > p[$1] : $0 < $1 }.prefix(3)
                preview = LivePreview(gloss: labels[order.first!], score: p[order.first!], emit: g.gloss != nil,
                                      top3: order.map { (labels[$0], p[$0]) })
            }
            break
        }
        let gap = config.duplicateGap
        if config.earlyK > 0 && !final {
            var next: [Int: (String, Int)] = [:]
            for (index, g) in segs.enumerated() {
                guard g.end + origin > horizon, let gloss = g.gloss else { continue }
                if ProcessInfo.processInfo.environment["ARM", default: "current"] == "current" {
                // Never publish BIG ahead of an earlier HAVE that is still waiting to commit.
                if segs[..<index].contains(where: { earlier in
                    (earlier.gloss != nil || earlier.end + origin > horizon) &&
                    !shownEarly.contains(where: { abs(earlier.start + origin - $0.0) <= 2 })
                }) { continue }
                }
                let st = g.start + origin
                let previous = early[st]
                let n = previous != nil && previous!.0 == gloss ? previous!.1 + 1 : 1
                next[st] = (gloss, n)
                if ProcessInfo.processInfo.environment["MY_DEBUG"] != nil && (gloss == "MY" || gloss == "LIKE") { print("IDENTITY", T, st, g.end + origin, gloss, n, exp(g.logQ)) }

                if n >= (gloss == "MY" ? Int(ProcessInfo.processInfo.environment["MY_K"] ?? "3")! : config.earlyK)
                    && exp(g.logQ) >= (gloss == "MY" ? Double(ProcessInfo.processInfo.environment["MY_Q"] ?? "0.9")! : config.earlyQ) && !config.earlyUnsafe.contains(gloss)
                    && !gloss.hasPrefix("FS_") && g.end - g.start + 1 >= config.earlyMin
                    && !shownEarly.contains(where: { abs(st - $0.0) <= 2 }) {
                    if let last, last.gloss == gloss, st - last.endFrame <= gap {
                        shownEarly.append((st, last))
                        continue
                    }
                    let w = word(gloss, st, g.end + origin, T, g.logQ, early: true)
                    last = w
                    shownEarly.append((st, w))
                    committed.append(w)
                }
            }
            early = next
        }
        var newOrigin = origin
        var pending = horizon + 1
        for g in segs {
            let end = g.end + origin
            if end > horizon { pending = g.start + origin; break }
            if !final && config.softExtra > 0 && end + 1 <= known {
                let p = bio[end + 1 - bioOff].map { exp($0) }
                if p[O] + p[UNK] + p[B] < config.hard && end > horizon - config.softExtra {
                    pending = g.start + origin
                    break
                }
            }
            let start = g.start + origin
            if let shown = shownEarly.first(where: { abs(start - $0.0) <= 2 })?.1 {
                shown.endFrame = max(shown.endFrame, end)
                shown.endSeconds = seconds(end)
                shown.closed = true
                newOrigin = end + 1
                continue
            }
            if let gloss = g.gloss {
                let sameGap = gloss.hasPrefix("FS_") ? config.letterGap : gap
                if let last, last.gloss == gloss, start - last.endFrame <= sameGap {
                    last.endFrame = end
                } else {
                    let w = word(gloss, start, end, T, g.logQ)
                    last = w
                    committed.append(w)
                }
            }
            newOrigin = end + 1
        }
        self.origin = max(newOrigin, min(pending, horizon + 1 - 30))
        if ProcessInfo.processInfo.environment["ARM"] == "ordered" {
            // Lock identity at the original time; only publication waits for earlier segments.
            pendingOutput += committed
            let barrier = final ? Int.max : segs.filter { g in
                g.end + origin >= self.origin &&
                !shownEarly.contains(where: { abs(g.start + origin - $0.0) <= 2 })
            }.map { $0.start + origin }.min() ?? Int.max
            committed = pendingOutput.filter { $0.startFrame < barrier }.sorted { $0.startFrame < $1.startFrame }
            pendingOutput.removeAll { $0.startFrame < barrier }
        }
        shownEarly = shownEarly.filter { $0.0 >= self.origin - 2 }
        trim()
        return committed
    }
}

/// Word decoder (word boundary; words + letters) and letter decoder (letter boundary; letters only)
/// over the same frames, sharing hand crops and span logits. A letter overlapping anything already
/// committed is dropped, so the decoders never double a letter.
final class LiveDualRuntime {
    let words: LiveSegmentalRuntime
    let letters: LiveSegmentalRuntime
    private let shared = LiveSpanCache()
    private var committed: [(Double, Double)] = []

    init(words: LiveSegmentalRuntime, letters: LiveSegmentalRuntime) {
        self.words = words; self.letters = letters
        words.rawCache = shared; letters.rawCache = shared
    }

    var preview: LivePreview? { words.preview }

    var timing: [String: Double] {
        get { words.timing.merging(letters.timing, uniquingKeysWith: +) }
        set { words.timing = newValue; letters.timing = newValue }
    }

    func reset() {
        words.reset(); letters.reset()
        shared.values = [:]
        committed = []
    }

    private func merge(_ wordOut: [LiveWord], _ letterOut: [LiveWord]) -> [LiveWord] {
        var kept: [LiveWord] = []
        let all = (wordOut.map { ($0, true) } + letterOut.map { ($0, false) })
            .enumerated().sorted { $0.element.0.commitSeconds != $1.element.0.commitSeconds
                ? $0.element.0.commitSeconds < $1.element.0.commitSeconds : $0.offset < $1.offset }
            .map(\.element)
        for (w, fromWords) in all {
            if w.gloss.hasPrefix("FS_") {
                let span = max(1e-6, w.endSeconds - w.startSeconds)
                if committed.contains(where: { a, b in
                    (min(b, w.endSeconds) - max(a, w.startSeconds)) / max(1e-6, min(span, b - a)) >= 0.5
                }) { continue }
            } else if !fromWords {
                continue
            }
            kept.append(w)
            committed.append((w.startSeconds, w.endSeconds))
        }
        committed = Array(committed.suffix(32))
        let oldest = min(words.origin, letters.origin)
        shared.values = shared.values.filter { $0.key.s >= oldest }
        return kept
    }

    func observe(_ observation: LiveObservation, image: LiveBGRAImage?) throws -> [LiveWord] {
        let wordOut = try words.observe(observation, image: image)
        let letterOut = try letters.observe(observation, image: nil, hand: words.hands.last)
        return merge(wordOut, letterOut)
    }

    func finish() throws -> [LiveWord] {
        merge(try words.finish(), try letters.finish())
    }

    /// Replay with hand evidence computed elsewhere (parity fixtures).
    func observeWithHand(_ observation: LiveObservation, hand: LiveHandFrame) throws -> [LiveWord] {
        let wordOut = try words.observe(observation, image: nil, hand: hand)
        let letterOut = try letters.observe(observation, image: nil, hand: hand)
        return merge(wordOut, letterOut)
    }
}

/// Committed FS_x letters -> spelled words ("fs-JOHN"); runs shorter than two letters are dropped.
final class LiveSpellingBuffer {
    let minimum = 2, gapSeconds = 2.0, sameLetterSeconds = 0.5
    private(set) var letters: [LiveWord] = []
    private(set) var dropped: [LiveWord] = []

    var pending: String { letters.map { String($0.gloss.dropFirst(3)) }.joined() }

    func push(_ word: LiveWord) -> [LiveWord] {
        guard word.gloss.hasPrefix("FS_") else {
            // Word and letter decoders can finish in either order. Remove provisional
            // handshape letters belonging to this word before locking a preceding name.
            let replaceRun = letters.first.map { word.startSeconds <= $0.startSeconds + 0.1 } ?? false
            letters = letters.filter { letter in
                let overlap = min(letter.endSeconds, word.endSeconds) - max(letter.startSeconds, word.startSeconds)
                let shorter = max(1e-6, min(letter.endSeconds - letter.startSeconds, word.endSeconds - word.startSeconds))
                if replaceRun && overlap / shorter >= 0.5 { dropped.append(letter); return false }
                return true
            }
            return flush() + [word]
        }
        var out: [LiveWord] = []
        var last = letters.last
        if let l = last, word.startSeconds - l.endSeconds > gapSeconds {
            out = flush()
            last = nil
        }
        if let l = last, l.gloss == word.gloss, word.startSeconds - l.endSeconds <= sameLetterSeconds {
            l.endSeconds = max(l.endSeconds, word.endSeconds)
            l.endFrame = max(l.endFrame, word.endFrame)
            l.commitSeconds = max(l.commitSeconds, word.commitSeconds)
            return out
        }
        letters.append(word.copy())
        return out
    }

    func tick(_ seconds: Double, activeHands: Bool? = nil) -> [LiveWord] {
        guard activeHands != true, let l = letters.last else { return [] }
        let reference = activeHands == nil ? l.commitSeconds : l.endSeconds
        let gap = activeHands == nil ? gapSeconds + 0.5 : gapSeconds
        if seconds - reference > gap { return flush() }
        return []
    }

    /// Drop held letters that start at or after `seconds` (the FINGERSPELL sign that ends spelling).
    func drop(from seconds: Double) {
        dropped += letters.filter { $0.startSeconds >= seconds }
        letters = letters.filter { $0.startSeconds < seconds }
    }

    func flush() -> [LiveWord] {
        let run = letters
        letters = []
        guard run.count >= minimum else { dropped += run; return [] }
        let spelled = run.map { String($0.gloss.dropFirst(3)) }.joined()
        let w = LiveWord(gloss: "fs-" + spelled, startFrame: run[0].startFrame, endFrame: run.last!.endFrame,
                         commitFrame: run.last!.commitFrame, startSeconds: run[0].startSeconds,
                         endSeconds: run.last!.endSeconds, commitSeconds: run.last!.commitSeconds,
                         score: run.map(\.score).reduce(0, +) / Double(run.count), early: false)
        w.spelled = spelled
        w.letters = run.map(\.gloss)
        return [w]
    }
}

/// Split an utterance at long pauses; a spelled word never starts a clause.
func liveSplitClauses(_ words: [LiveWord], pauseSeconds: Double = 1.5) -> [[LiveWord]] {
    var clauses: [[LiveWord]] = []
    for w in words {
        if let lastClause = clauses.last, !w.gloss.hasPrefix("fs-"),
           w.startSeconds - lastClause.last!.endSeconds >= pauseSeconds {
            clauses.append([w])
        } else if clauses.isEmpty {
            clauses.append([w])
        } else {
            clauses[clauses.count - 1].append(w)
        }
    }
    return clauses
}

/// The ASL sign FINGERSPELL as the spell-mode switch (active/v17/fingerspell_trigger_v17.py): boosted
/// trees over 1 s windows (20 frames, every 2) of raw landmarks, score = mean of the last 3 windows.
/// Detects the deliberate (citation) form. After firing, the score must fall below threshold and 1.5 s
/// must pass before it can fire again, so one sign never switches spelling on and straight back off.
final class LiveFingerspellTrigger {
    private struct File: Decodable {
        struct Tree: Decodable {
            let feature: [Int]; let threshold: [Double]; let left: [Int]; let right: [Int]
            let leaf: [Int]; let value: [Double]
        }
        let baseline: Double; let threshold: Double
        let window: Int; let step: Int; let smooth: Int; let features: Int
        let trees: [Tree]
    }
    private let file: File
    static let refractory = 1.5
    static let tips = [8, 12, 16, 20], pips = [6, 10, 14, 18]
    var threshold: Double { file.threshold }
    private var frames: [[Float]] = [], times: [Double] = [], scores: [Double] = []
    private var count = 0, lastFire = -Double.infinity, armed = true
    private var aboveSince: Double?
    private(set) var score = 0.0

    init(url: URL) throws {
        file = try JSONDecoder().decode(File.self, from: Data(contentsOf: url))
        guard file.features == 22 else { throw LiveReelError.model("fingerspell trigger: expected 22 features") }
    }

    func reset() {
        frames = []; times = []; scores = []
        count = 0; lastFire = -.infinity; armed = true; aboveSince = nil; score = 0
    }

    func probability(_ row: [Double]) -> Double {
        var raw = file.baseline
        for t in file.trees {
            var i = 0
            while t.leaf[i] == 0 { i = row[t.feature[i]] <= t.threshold[i] ? t.left[i] : t.right[i] }
            raw += t.value[i]
        }
        return 1 / (1 + exp(-raw))
    }

    /// One raw frame (LiveFeatures.raw); returns the sign's (start, end) seconds when it fires.
    func push(_ raw: [Float], seconds: Double) -> (start: Double, end: Double)? {
        frames.append(raw); times.append(seconds)
        if frames.count > file.window { frames.removeFirst(); times.removeFirst() }
        count += 1
        guard count >= file.window, (count - file.window) % file.step == 0 else { return nil }
        scores.append(probability(Self.windowRow(frames)))
        if scores.count > file.smooth { scores.removeFirst() }
        score = scores.reduce(0, +) / Double(scores.count)
        guard score > file.threshold else {
            aboveSince = nil
            if seconds - lastFire >= Self.refractory { armed = true }
            return nil
        }
        if aboveSince == nil { aboveSince = times[0] }
        if armed && seconds - lastFire >= Self.refractory {
            armed = false; lastFire = seconds
            return (aboveSince!, seconds)
        }
        return nil
    }

    /// window_row: the more present/moving hand's 18 features, then 4 of the other hand's.
    static func windowRow(_ frames: [[Float]]) -> [Double] {
        var a = handFeatures(frames, 0), b = handFeatures(frames, 21)
        if b[0] > a[0] || (b[0] == a[0] && b[8] > a[8]) { swap(&a, &b) }
        return a + [b[0], b[1], b[3], b[8]]
    }

    /// hand_features: 18 features of hand block `s` (0 or 21) over one window; float32 like numpy.
    static func handFeatures(_ frames: [[Float]], _ s: Int) -> [Double] {
        var f = [Double](repeating: 0, count: 18)
        typealias P = SIMD2<Double>
        func point(_ fr: [Float], _ node: Int) -> P { P(Double(fr[node * 5]), Double(fr[node * 5 + 1])) }
        func norm(_ p: P) -> Double { (p.x * p.x + p.y * p.y).squareRoot() }
        func mean(_ v: [Double]) -> Double { v.isEmpty ? 0 : v.reduce(0, +) / Double(v.count) }
        let presentFrames = frames.filter { fr in (0..<21).filter { fr[(s + $0) * 5 + 4] > 0 }.count >= 15 }
        f[0] = Double(presentFrames.count) / Double(frames.count)
        guard presentFrames.count >= 4 else { return f.map { Double(Float($0)) } }
        var xy = presentFrames.map { fr in (0..<21).map { point(fr, s + $0) } }
        var rel = xy.map { h in h.map { $0 - h[0] } }
        var palm = rel.map { norm($0[9]) }
        let sorted = palm.sorted(), n = sorted.count
        let palmM = n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2
        let good = palm.indices.filter { palm[$0] > 0.6 * palmM }
        guard good.count >= 4, palmM >= 1e-3 else { return f.map { Double(Float($0)) } }
        xy = good.map { xy[$0] }; rel = good.map { rel[$0] }; palm = good.map { palm[$0] }
        let reln = zip(rel, palm).map { h, p in h.map { $0 / p } }
        let tip = reln.map { h in tips.map { norm(h[$0]) } }, pip = reln.map { h in pips.map { norm(h[$0]) } }
        let ext = zip(tip, pip).map { t, p in zip(t, p).map { $0 / max($1, 1e-3) } }
        f[1] = mean(ext.flatMap { $0 }); f[2] = mean(ext.map { $0.min()! })
        let v = (1..<tip.count).map { t in (0..<4).map { tip[t][$0] - tip[t - 1][$0] } }
        f[3] = mean(v.flatMap { $0 }.map(abs))
        func sign(_ x: Double) -> Double { x > 0 ? 1 : x < 0 ? -1 : 0 }
        if v.count > 2 {
            f[4] = mean((1..<v.count).flatMap { t in (0..<4).map { sign(v[t][$0]) != sign(v[t - 1][$0]) ? 1.0 : 0 } })
        }
        f[5] = mean(v.map { row in let m = mean(row); return mean(row.map { ($0 - m) * ($0 - m) }).squareRoot() })
        let wrist = xy.map { $0[0] / palmM }
        let d = wrist.last! - wrist.first!
        f[6] = d.x; f[7] = d.y
        let path = (1..<wrist.count).map { norm(wrist[$0] - wrist[$0 - 1]) }.reduce(0, +)
        f[8] = path; f[9] = norm(d) / max(path, 1e-3)
        f[10] = mean(zip(tips, tips.dropFirst()).map { a, b in mean(reln.map { norm($0[a] - $0[b]) }) })
        let direction = reln.map { h -> P in let q = h[12] - h[9]; return q / max(norm(q), 1e-6) }
        f[11] = mean(direction.map(\.x)); f[12] = mean(direction.map(\.y))
        f[13] = mean(reln.map { norm($0[4] - $0[5]) })
        let body = frames.filter { $0[57 * 5 + 4] > 0 && $0[58 * 5 + 4] > 0 }
        if !body.isEmpty {
            let width = mean(body.map { norm(point($0, 58) - point($0, 57)) })
            let cx = mean(body.map { (point($0, 57).x + point($0, 58).x) / 2 })
            let cy = mean(body.map { (point($0, 57).y + point($0, 58).y) / 2 })
            if width > 1e-3 {
                f[14] = (cy - mean(xy.map { $0[0].y })) / width
                f[15] = abs(mean(xy.map { $0[0].x }) - cx) / width
                f[16] = palmM / width
                f[17] = 1
            }
        }
        return f.map { $0.isNaN ? 0 : Double(Float(min(max($0, -20), 20))) }
    }
}
