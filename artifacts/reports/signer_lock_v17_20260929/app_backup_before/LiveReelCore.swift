// Live segmental Reel (v17, Apple Vision only): per-frame observation and feature contracts.
//
// Ports of the Python live path in the SLT repo, kept numerically identical where it matters:
//   observe_stage2_frame / AppleVisionDetector / assign_hands      (scripts/live_stage2_ctc_v17.py, active/v17/extract_v17.py)
//   raw_observation_features                                         (active/v17/stage1_window_v17.py)
//   boundary_features                                                (active/v17/temporal_boundary_v17.py)
//   landmarks_from_observations                                      (scripts/live_isolated_v17.py, active/v17/geometry_v17.py)
//   hand_box / union_box / crop_square (reflect-101, INTER_AREA/CUBIC) (active/v17/extract_hand_rgb_v17.py)
// This file has no UIKit dependency so the same code runs in the macOS replay harness.

import CoreVideo
import Foundation
import Vision

enum LiveReelError: LocalizedError {
    case model(String)
    case input(String)

    var errorDescription: String? {
        switch self {
        case .model(let value), .input(let value): return value
        }
    }
}

struct LivePoint {
    var x: Float = 0
    var y: Float = 0      // image-normalized, origin top-left
    var c: Float = 0
}

struct LiveHand {
    var points = [LivePoint](repeating: LivePoint(), count: 21)
    var chirality = "unknown"
    var score: Float = 0
}

/// Two upright open palms held continuously for one second, once per release.
/// Uses the same normalized geometry as the desktop FinishGesture detector.
struct LiveFinishGesture {
    let holdSeconds = 1.0
    private var startedAt: Double?
    private(set) var active = false
    private(set) var latched = false
    private(set) var progress = 0.0

    mutating func update(left: LiveHand?, right: LiveHand?, seconds: Double) -> Bool {
        guard Self.isOpenPalm(left), Self.isOpenPalm(right),
              let left, let right,
              Self.length(Self.xy(left.points[0]) - Self.xy(right.points[0])) >= 0.12 else {
            self = LiveFinishGesture()
            return false
        }
        active = true
        if latched { return false }
        if startedAt == nil { startedAt = seconds }
        progress = min(1, max(0, seconds - startedAt!) / holdSeconds)
        guard progress >= 1 else { return false }
        latched = true
        return true
    }

    private static func xy(_ p: LivePoint) -> SIMD2<Float> { SIMD2(p.x, p.y) }
    private static func dot(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float { a.x * b.x + a.y * b.y }
    private static func length(_ a: SIMD2<Float>) -> Float { sqrt(dot(a, a)) }

    static func isOpenPalm(_ hand: LiveHand?) -> Bool {
        guard let hand, hand.points.count == 21 else { return false }
        let required = [0, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20]
        guard required.allSatisfy({ hand.points[$0].c > 0 }) else { return false }
        let p = hand.points.map(xy)
        let wrist = p[0], palm = p[9] - wrist
        let palmLength = length(palm)
        guard palmLength >= 0.025 else { return false }
        let axis = palm / palmLength
        guard axis.y <= -0.25, wrist.y <= 0.80 else { return false }
        for (mcp, pip, tip) in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)] {
            let proximal = p[pip] - p[mcp], distal = p[tip] - p[pip]
            let scale = length(proximal) * length(distal)
            guard scale > 1e-6, dot(proximal, distal) / scale > 0.45,
                  length(p[tip] - wrist) > 1.10 * length(p[pip] - wrist),
                  dot(distal, axis) > 0.005 else { return false }
        }
        let proximal = p[3] - p[2], distal = p[4] - p[3]
        let scale = length(proximal) * length(distal)
        return scale > 1e-6 && dot(proximal, distal) / scale > 0.15
            && length(p[4] - wrist) > 1.05 * length(p[3] - wrist)
    }
}

/// One processed camera frame (the Python ObservedFrame without pixels).
struct LiveObservation {
    let seconds: Double
    let width: Int
    let height: Int
    let left: LiveHand?
    let right: LiveHand?
    let body: [LivePoint]            // 4: L/R shoulder, L/R elbow; zero unless detected this frame
    let face: [LivePoint]            // 15; zero unless detected this frame
    let faceForFeatures: Bool
    var displayBody: [LivePoint]? = nil
    var displayFace: [LivePoint]? = nil
}

/// Per-frame hand-crop evidence: embeddings [3*512], valid [3], boxes [3*4] (left, right, union).
struct LiveHandFrame {
    var embeddings = [Float](repeating: 0, count: 3 * 512)
    var valid = [Float](repeating: 0, count: 3)
    var boxes = [Float](repeating: 0, count: 12)
}

enum LiveNodes {
    static let count = 61
    static let bodyStart = 57
    static let faceStart = 42
}

// MARK: - Apple Vision

/// Hands every frame; body and face every `auxiliaryInterval` processed frames (v17 contract).
final class LiveVision {
    static let auxiliaryInterval = 8
    private let auxiliaryCadence: Int
    init(auxiliaryInterval: Int = LiveVision.auxiliaryInterval) {
        auxiliaryCadence = max(1, auxiliaryInterval)
    }
    static let minimumConfidence: Float = 0.15

    private let handRequest: VNDetectHumanHandPoseRequest = {
        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 2
        return request
    }()
    private let bodyRequest = VNDetectHumanBodyPoseRequest()
    private let faceRequest = VNDetectFaceLandmarksRequest()
    private var previousWrists: [String: SIMD2<Float>] = [:]
    private(set) var processed = 0

    private static let handJoints: [VNHumanHandPoseObservation.JointName] = [
        .wrist,
        .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
        .indexMCP, .indexPIP, .indexDIP, .indexTip,
        .middleMCP, .middlePIP, .middleDIP, .middleTip,
        .ringMCP, .ringPIP, .ringDIP, .ringTip,
        .littleMCP, .littlePIP, .littleDIP, .littleTip,
    ]
    private static let bodyJoints: [VNHumanBodyPoseObservation.JointName] = [
        .leftShoulder, .rightShoulder, .leftElbow, .rightElbow,
    ]

    /// Forget hand-slot history (after a reset); the auxiliary cadence keeps running.
    func resetTracking() { previousWrists = [:] }

    /// `detection` is the (possibly downscaled) image given to Vision; `width`/`height` are the
    /// processed frame's dimensions used for geometry and crops. Both must be upright, unmirrored.
    func observe(detection: CVPixelBuffer, width: Int, height: Int, seconds: Double,
                 displayAuxiliary: Bool = false) throws -> LiveObservation {
        let auxiliary = processed % auxiliaryCadence == 0
        processed += 1
        var requests: [VNRequest] = [handRequest]
        if auxiliary || displayAuxiliary { requests += [bodyRequest, faceRequest] }
        try autoreleasepool {
            try VNImageRequestHandler(cvPixelBuffer: detection, orientation: .up, options: [:]).perform(requests)
        }
        let hands = (handRequest.results ?? []).compactMap { Self.parseHand($0) }
        let assigned = Self.assignHands(hands, previous: previousWrists)
        for slot in ["left", "right"] {
            if let hand = assigned[slot], hand.points[0].c > 0 {
                previousWrists[slot] = SIMD2(hand.points[0].x, hand.points[0].y)
            }
        }
        let empty4 = [LivePoint](repeating: LivePoint(), count: 4)
        let empty15 = [LivePoint](repeating: LivePoint(), count: 15)
        let body = auxiliary || displayAuxiliary ? (Self.parseBody(bodyRequest.results ?? []) ?? empty4) : empty4
        let face = auxiliary || displayAuxiliary ? (Self.parseFace(faceRequest.results ?? []) ?? empty15) : empty15
        return LiveObservation(
            seconds: seconds, width: width, height: height,
            left: assigned["left"], right: assigned["right"],
            body: auxiliary ? body : empty4,
            face: auxiliary ? face : empty15,
            faceForFeatures: auxiliary,
            displayBody: displayAuxiliary ? body : nil,
            displayFace: displayAuxiliary ? face : nil
        )
    }

    static func parseHand(_ observation: VNHumanHandPoseObservation) -> LiveHand? {
        guard let recognized = try? observation.recognizedPoints(.all) else { return nil }
        var hand = LiveHand()
        for (index, joint) in handJoints.enumerated() {
            guard let point = recognized[joint], point.confidence >= minimumConfidence else { continue }
            hand.points[index] = LivePoint(x: Float(point.location.x), y: Float(1 - point.location.y),
                                           c: point.confidence)
        }
        let valid = hand.points.filter { $0.c > 0 }
        guard valid.count >= 5 else { return nil }
        hand.score = valid.map(\.c).reduce(0, +) / Float(valid.count)
        switch observation.chirality {
        case .left: hand.chirality = "left"
        case .right: hand.chirality = "right"
        default: hand.chirality = "unknown"
        }
        return hand
    }

    static func parseBody(_ observations: [VNHumanBodyPoseObservation]) -> [LivePoint]? {
        guard let observation = observations.max(by: { $0.confidence < $1.confidence }),
              let recognized = try? observation.recognizedPoints(.all) else { return nil }
        return bodyJoints.map { joint in
            guard let point = recognized[joint], point.confidence >= minimumConfidence else { return LivePoint() }
            return LivePoint(x: Float(point.location.x), y: Float(1 - point.location.y), c: point.confidence)
        }
    }

    static func parseFace(_ observations: [VNFaceObservation]) -> [LivePoint]? {
        guard let observation = observations.max(by: {
            $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height
        }), let landmarks = observation.landmarks else { return nil }
        let specifications: [(VNFaceLandmarkRegion2D?, Int)] = [
            (landmarks.leftPupil, 0), (landmarks.rightPupil, 0),
            (landmarks.leftEyebrow, 0), (landmarks.leftEyebrow, -1),
            (landmarks.rightEyebrow, 0), (landmarks.rightEyebrow, -1),
            (landmarks.noseCrest, -1),
            (landmarks.outerLips, 0), (landmarks.outerLips, 7),
            (landmarks.outerLips, 3), (landmarks.outerLips, 10),
            (landmarks.faceContour, 0), (landmarks.faceContour, 8),
            (landmarks.faceContour, -1), (landmarks.noseCrest, 0),
        ]
        let score = max(observation.confidence, minimumConfidence)
        let box = observation.boundingBox
        return specifications.map { region, requested in
            guard let region, region.pointCount > 0 else { return LivePoint() }
            let points = region.normalizedPoints
            let index = requested >= 0 ? requested : points.count + requested
            guard points.indices.contains(index) else { return LivePoint() }
            let local = points[index]
            return LivePoint(x: Float(box.minX + local.x * box.width),
                             y: Float(1 - (box.minY + local.y * box.height)), c: score)
        }
    }

    private static func cost(_ hand: LiveHand, slot: String, previous: SIMD2<Float>?) -> Float {
        var cost = -0.1 * hand.score
        if hand.chirality != "unknown" {
            cost += hand.chirality == slot ? 0 : 2
        } else if hand.points[0].c > 0 {
            let expectedLeft = hand.points[0].x >= 0.5
            if (slot == "left") != expectedLeft { cost += 0.2 }
        }
        if let previous, hand.points[0].c > 0 {
            cost += simd_distance(SIMD2(hand.points[0].x, hand.points[0].y), previous)
        }
        return cost
    }

    static func assignHands(_ hands: [LiveHand], previous: [String: SIMD2<Float>]) -> [String: LiveHand] {
        // Stable sort by score, descending, as Python's sorted(..., reverse=True) keeps ties in order.
        let usable = Array(hands.enumerated().sorted {
            $0.element.score != $1.element.score ? $0.element.score > $1.element.score : $0.offset < $1.offset
        }.map(\.element).prefix(2))
        guard let first = usable.first else { return [:] }
        if usable.count == 1 {
            let left = cost(first, slot: "left", previous: previous["left"])
            let right = cost(first, slot: "right", previous: previous["right"])
            return [left <= right ? "left" : "right": first]
        }
        let second = usable[1]
        let direct = cost(first, slot: "left", previous: previous["left"])
            + cost(second, slot: "right", previous: previous["right"])
        let swapped = cost(first, slot: "right", previous: previous["right"])
            + cost(second, slot: "left", previous: previous["left"])
        return direct <= swapped ? ["left": first, "right": second] : ["left": second, "right": first]
    }
}

// MARK: - Raw features and boundary features

/// A bounded newest-frame slot. Slow inference never queues a backlog or blocks capture.
final class LiveLatestFrameGate<Frame> {
    private let lock = NSLock()
    private var enabled = false, busy = false
    private var generation = 0
    private var pending: Frame?
    func setEnabled(_ value: Bool) {
        lock.lock(); defer { lock.unlock() }
        enabled = value; generation &+= 1; pending = nil
    }
    var activeGeneration: Int? {
        lock.lock(); defer { lock.unlock() }
        return enabled ? generation : nil
    }
    func isCurrent(_ value: Int) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return enabled && generation == value
    }
    func offer(_ frame: Frame, generation expected: Int? = nil) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard enabled, expected == nil || expected == generation else { return false }
        pending = frame
        guard !busy else { return false }
        busy = true
        return true
    }
    func take() -> (Frame, Int)? {
        lock.lock(); defer { lock.unlock() }
        guard enabled, let frame = pending else { return nil }
        pending = nil
        return (frame, generation)
    }
    func complete() -> Bool {
        lock.lock(); defer { lock.unlock() }
        if enabled && pending != nil { return true }
        busy = false
        return false
    }
}

enum LiveFeatures {
    /// raw_observation_features for one frame: [61*5] (isotropic x, y, 0, present, confidence).
    static func raw(_ o: LiveObservation) -> [Float] {
        var raw = [Float](repeating: 0, count: 61 * 5)
        func put(_ node: Int, _ p: LivePoint) {
            raw[node * 5] = p.x; raw[node * 5 + 1] = p.y; raw[node * 5 + 4] = p.c
        }
        if let hand = o.left { for j in 0..<21 { put(j, hand.points[j]) } }
        if let hand = o.right { for j in 0..<21 { put(21 + j, hand.points[j]) } }
        for j in 0..<4 { put(57 + j, o.body[j]) }
        if o.faceForFeatures { for j in 0..<15 { put(42 + j, o.face[j]) } }
        let w = Float(o.width), h = Float(o.height), longest = Float(max(o.width, o.height))
        for node in 0..<61 {
            let present: Float = raw[node * 5 + 4] > 0 ? 1 : 0
            raw[node * 5 + 3] = present
            raw[node * 5] = (raw[node * 5] * w - w / 2) / longest * present
            raw[node * 5 + 1] = (raw[node * 5 + 1] * h - h / 2) / longest * present
        }
        return raw
    }

    static let boundaryDimension = 450

    /// boundary_features(raw, times, hand_geometry=True) for a short history; returns the last row.
    static func boundaryRow(raws: [[Float]], times: [Double]) -> [Float] {
        let n = raws.count
        var present = [[Bool]](repeating: [Bool](repeating: false, count: 61), count: n)
        var normal = [[SIMD2<Float>]](repeating: [SIMD2<Float>](repeating: .zero, count: 61), count: n)
        var center = SIMD2<Float>.zero
        var scale: Float = 0.3
        var lastBody = -Double.infinity
        for i in 0..<n {
            let raw = raws[i]
            for node in 0..<61 { present[i][node] = raw[node * 5 + 3] > 0.5 && raw[node * 5 + 4] > 0 }
            if present[i][57] && present[i][58] {
                let a = SIMD2(raw[57 * 5], raw[57 * 5 + 1]), b = SIMD2(raw[58 * 5], raw[58 * 5 + 1])
                let width = simd_distance(b, a)
                if width > 0.02 { center = (a + b) / 2; scale = width; lastBody = times[i] }
            }
            if times[i] - lastBody > 0.8 { center = .zero; scale = 0.3 }
            for node in 0..<61 where present[i][node] {
                let xy = SIMD2(raw[node * 5], raw[node * 5 + 1])
                normal[i][node] = simd_clamp((xy - center) / scale, SIMD2(repeating: -4), SIMD2(repeating: 4))
            }
        }
        let last = n - 1
        var row = [Float](repeating: 0, count: boundaryDimension)
        for node in 0..<61 {
            row[node * 2] = normal[last][node].x
            row[node * 2 + 1] = normal[last][node].y
        }
        if n > 1 {
            let dt = Float(times[last] - times[last - 1])
            if Double(dt) <= 0.26 {
                for node in 0..<61 where present[last][node] && present[last - 1][node] {
                    let v = simd_clamp((normal[last][node] - normal[last - 1][node]) / dt,
                                       SIMD2(repeating: -10), SIMD2(repeating: 10)) / 10
                    row[122 + node * 2] = v.x
                    row[122 + node * 2 + 1] = v.y
                }
            }
        }
        let raw = raws[last]
        for node in 0..<61 {
            let p: Float = present[last][node] ? 1 : 0
            row[244 + node] = p
            row[305 + node] = min(max(raw[node * 5 + 4], 0), 1) * p
        }
        for (h, start) in [0, 21].enumerated() {
            let wrist = SIMD2(raw[start * 5], raw[start * 5 + 1])
            let mcp = SIMD2(raw[(start + 9) * 5], raw[(start + 9) * 5 + 1])
            let palm = simd_distance(mcp, wrist)
            guard present[last][start], present[last][start + 9], palm > 0.005 else { continue }
            for j in 0..<21 where present[last][start + j] {
                let xy = SIMD2(raw[(start + j) * 5], raw[(start + j) * 5 + 1])
                let local = simd_clamp((xy - wrist) / max(palm, 0.005), SIMD2(repeating: -4), SIMD2(repeating: 4))
                row[366 + h * 42 + j * 2] = local.x
                row[366 + h * 42 + j * 2 + 1] = local.y
            }
        }
        return row
    }
}

// MARK: - Span landmarks (landmarks_from_observations)

enum LiveSpanLandmarks {
    static let frames = 32

    private struct Track { var xy = [SIMD2<Float>](repeating: .zero, count: 61); var c = [Float](repeating: 0, count: 61) }

    /// [32*61*5] features plus the trim range, or nil when fewer than two frames have a hand.
    static func build(_ observations: [LiveObservation]) -> (features: [Float], trimStart: Int, trimEnd: Int)? {
        guard observations.count >= 4 else { return nil }
        let width = observations[0].width, height = observations[0].height
        guard observations.allSatisfy({ $0.width == width && $0.height == height }) else { return nil }
        var tracks = observations.map { o -> Track in
            var t = Track()
            if let hand = o.left { for j in 0..<21 { t.xy[j] = SIMD2(hand.points[j].x, hand.points[j].y); t.c[j] = hand.points[j].c } }
            if let hand = o.right { for j in 0..<21 { t.xy[21 + j] = SIMD2(hand.points[j].x, hand.points[j].y); t.c[21 + j] = hand.points[j].c } }
            if o.body.contains(where: { $0.c > 0 }) {
                for j in 0..<4 { t.xy[57 + j] = SIMD2(o.body[j].x, o.body[j].y); t.c[57 + j] = o.body[j].c }
            }
            if o.faceForFeatures && o.face.contains(where: { $0.c > 0 }) {
                for j in 0..<15 { t.xy[42 + j] = SIMD2(o.face[j].x, o.face[j].y); t.c[42 + j] = o.face[j].c }
            }
            return t
        }
        let active = tracks.indices.filter { i in (0..<42).filter { tracks[i].c[$0] > 0 }.count >= 5 }
        guard active.count >= 2 else { return nil }
        let trimStart = max(0, active.first! - 2)
        let trimEnd = min(tracks.count, active.last! + 3)
        tracks = Array(tracks[trimStart..<trimEnd])
        let w = Float(width), h = Float(height), longest = Float(max(width, height))
        for i in tracks.indices {
            for node in 0..<61 {
                if tracks[i].c[node] > 0 {
                    let v = tracks[i].xy[node]
                    tracks[i].xy[node] = SIMD2((v.x * w - w / 2) / longest, (v.y * h - h / 2) / longest)
                } else {
                    tracks[i].xy[node] = .zero
                }
            }
        }
        interpolate(&tracks, nodes: 0..<42, maximumGap: 3)
        interpolate(&tracks, nodes: 42..<61, maximumGap: 16)
        let normalized = normalize(tracks)
        return (resample(normalized, count: tracks.count), trimStart, trimEnd)
    }

    private static func interpolate(_ tracks: inout [Track], nodes: Range<Int>, maximumGap: Int) {
        for node in nodes {
            let valid = tracks.indices.filter { tracks[$0].c[node] > 0 }
            for (left, right) in zip(valid, valid.dropFirst()) {
                let gap = right - left - 1
                guard gap > 0, gap <= maximumGap else { continue }
                let start = tracks[left].xy[node], end = tracks[right].xy[node]
                let confidence = min(tracks[left].c[node], tracks[right].c[node]) * 0.5
                for offset in 1...gap {
                    let fraction = Float(offset) / Float(gap + 1)
                    tracks[left + offset].xy[node] = start + fraction * (end - start)
                    tracks[left + offset].c[node] = confidence
                }
            }
        }
    }

    private static func median(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count % 2 == 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    }

    /// body_relative_normalize, returning per frame [61*5] (x, y, depth, present, confidence).
    private static func normalize(_ tracks: [Track]) -> [[Float]] {
        let n = tracks.count
        var widths: [Float] = []
        var known: [(Int, SIMD2<Float>)] = []
        var shoulderValid = [Bool](repeating: false, count: n)
        var shoulderWidth = [Float](repeating: 0, count: n)
        for i in 0..<n where tracks[i].c[57] > 0 && tracks[i].c[58] > 0 {
            let width = simd_distance(tracks[i].xy[58], tracks[i].xy[57])
            guard width > 1e-5 else { continue }
            shoulderValid[i] = true; shoulderWidth[i] = width
            widths.append(width)
            known.append((i, (tracks[i].xy[57] + tracks[i].xy[58]) / 2))
        }
        var centers = [SIMD2<Float>](repeating: .zero, count: n)
        var scale: Float
        if !known.isEmpty {
            for i in 0..<n {
                if i <= known[0].0 { centers[i] = known[0].1; continue }
                if i >= known.last!.0 { centers[i] = known.last!.1; continue }
                let r = known.firstIndex { $0.0 >= i }!
                let a = known[r - 1], b = known[r]
                let f = Float(i - a.0) / Float(b.0 - a.0)
                centers[i] = a.1 + f * (b.1 - a.1)
            }
            scale = median(widths)
        } else {
            var xs: [Float] = [], ys: [Float] = []
            for start in [0, 21] {
                for t in tracks where t.c[start] > 0 { xs.append(t.xy[start].x); ys.append(t.xy[start].y) }
            }
            let fallback = xs.isEmpty ? SIMD2<Float>.zero : SIMD2(median(xs), median(ys))
            centers = [SIMD2<Float>](repeating: fallback, count: n)
            var palms: [Float] = []
            for start in [0, 21] {
                for t in tracks where t.c[start] > 0 && t.c[start + 9] > 0 {
                    let length = simd_distance(t.xy[start + 9], t.xy[start])
                    if length > 1e-5 { palms.append(length) }
                }
            }
            scale = palms.isEmpty ? 1 : median(palms)
        }
        if !scale.isFinite || scale <= 1e-5 { scale = 1 }
        var out = [[Float]](repeating: [Float](repeating: 0, count: 61 * 5), count: n)
        for i in 0..<n {
            for node in 0..<61 where tracks[i].c[node] > 0 {
                let v = (tracks[i].xy[node] - centers[i]) / scale
                out[i][node * 5] = v.x
                out[i][node * 5 + 1] = v.y
                out[i][node * 5 + 3] = 1
                out[i][node * 5 + 4] = min(max(tracks[i].c[node], 0), 1)
            }
        }
        for start in [0, 21] {
            var lengths = [Float](repeating: 0, count: n)
            var valid = [Bool](repeating: false, count: n)
            for i in 0..<n where tracks[i].c[start] > 0 && tracks[i].c[start + 9] > 0 {
                lengths[i] = simd_distance(tracks[i].xy[start + 9], tracks[i].xy[start])
                valid[i] = lengths[i] > 1e-5
            }
            let reference = median(zip(lengths, valid).filter(\.1).map(\.0))
            guard valid.contains(true) else { continue }
            for i in 0..<n where valid[i] {
                let depth = log(reference / lengths[i])
                for node in start..<(start + 21) where out[i][node * 5 + 3] > 0 { out[i][node * 5 + 2] = depth }
            }
        }
        if !widths.isEmpty {
            let reference = median(widths)
            for i in 0..<n where shoulderValid[i] {
                let depth = log(reference / shoulderWidth[i])
                for node in 42..<61 where out[i][node * 5 + 3] > 0 { out[i][node * 5 + 2] = depth }
            }
        }
        return out
    }

    /// resample_features to 32 frames: linear for x/y/depth/confidence, nearest for presence.
    private static func resample(_ input: [[Float]], count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: frames * 61 * 5)
        for target in 0..<frames {
            let position = count == 1 ? 0 : Double(target) / Double(frames - 1) * Double(count - 1)
            let left = min(Int(position.rounded(.down)), count - 1)
            let right = min(left + 1, count - 1)
            let fraction = Float(position - Double(left))
            let nearest = Int(position.rounded(.toNearestOrEven))
            for node in 0..<61 {
                let presence: Float = input[nearest][node * 5 + 3] >= 0.5 ? 1 : 0
                let base = (target * 61 + node) * 5
                out[base + 3] = presence
                guard presence > 0 else { continue }
                for channel in [0, 1, 2, 4] {
                    let a = input[left][node * 5 + channel], b = input[right][node * 5 + channel]
                    out[base + channel] = a + fraction * (b - a)
                }
            }
        }
        return out
    }
}

// MARK: - Hand boxes and crops (cv2-compatible)

/// A BGRA8 image in memory (camera frame or decoded video frame).
struct LiveBGRAImage {
    let base: UnsafePointer<UInt8>
    let width: Int
    let height: Int
    let bytesPerRow: Int
}

enum LiveHandCrops {
    static let cropSize = 256
    static let handBoxScale: Double = 1.70
    static let unionBoxScale: Float = 1.20
    static let minimumBoxFraction: Double = 0.14

    /// hand_box: [x0, y0, x1, y1] in pixels (float32 like the Python array), or nil.
    static func handBox(_ hand: LiveHand, width: Int, height: Int) -> [Float]? {
        let valid = hand.points.filter { $0.c > 0 }
        guard valid.count >= 5 else { return nil }
        let x0 = Double(valid.map(\.x).min()!), x1 = Double(valid.map(\.x).max()!)
        let y0 = Double(valid.map(\.y).min()!), y1 = Double(valid.map(\.y).max()!)
        let cx = 0.5 * (x0 + x1) * Double(width), cy = 0.5 * (y0 + y1) * Double(height)
        let detected = max((x1 - x0) * Double(width), (y1 - y0) * Double(height))
        let side = max(detected * handBoxScale, minimumBoxFraction * Double(max(width, height)))
        return [Float(cx - side / 2), Float(cy - side / 2), Float(cx + side / 2), Float(cy + side / 2)]
    }

    static func unionBox(_ boxes: [[Float]]) -> [Float]? {
        guard !boxes.isEmpty else { return nil }
        let x0 = boxes.map { $0[0] }.min()!, y0 = boxes.map { $0[1] }.min()!
        let x1 = boxes.map { $0[2] }.max()!, y1 = boxes.map { $0[3] }.max()!
        let cx = 0.5 * (x0 + x1), cy = 0.5 * (y0 + y1)
        let side = max(max(x1 - x0, y1 - y0) * unionBoxScale, 1)
        return [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
    }

    /// Boxes and 256x256 RGB crops for the left, right and union views of one frame.
    static func crops(_ o: LiveObservation, image: LiveBGRAImage) -> (valid: [Float], boxes: [Float], crops: [[UInt8]?]) {
        var valid = [Float](repeating: 0, count: 3)
        var boxes = [Float](repeating: 0, count: 12)
        var crops: [[UInt8]?] = [nil, nil, nil]
        var seen: [[Float]] = []
        let w = Float(o.width), h = Float(o.height)
        for (view, hand) in [o.left, o.right].enumerated() {
            guard let hand, let box = handBox(hand, width: o.width, height: o.height) else { continue }
            valid[view] = 1
            boxes[view * 4 ..< view * 4 + 4] = [box[0] / w, box[1] / h, box[2] / w, box[3] / h]
            seen.append(box)
            crops[view] = cropSquare(image, box: box)
        }
        if let box = unionBox(seen) {
            valid[2] = 1
            boxes[8..<12] = [box[0] / w, box[1] / h, box[2] / w, box[3] / h]
            crops[2] = cropSquare(image, box: box)
        }
        return (valid, boxes, crops)
    }

    private static func reflect101(_ p: Int, _ n: Int) -> Int {
        guard n > 1 else { return 0 }
        var p = p
        while p < 0 || p >= n { p = p < 0 ? -p : 2 * n - 2 - p }
        return p
    }

    /// crop_square: round the box, reflect-101 outside the frame, then INTER_AREA (shrink) or
    /// INTER_CUBIC (enlarge) to 256x256. Returns RGB bytes, row-major.
    static func cropSquare(_ image: LiveBGRAImage, box: [Float]) -> [UInt8] {
        let r = box.map { Int(Double($0).rounded(.toNearestOrEven)) }
        let cw = max(r[2] - r[0], 1), ch = max(r[3] - r[1], 1)
        let xs = (0..<cw).map { reflect101(r[0] + $0, image.width) }
        let ys = (0..<ch).map { reflect101(r[1] + $0, image.height) }
        // Source crop as RGB floats is avoided: taps read straight from the BGRA frame.
        if max(cw, ch) > cropSize {
            return resizeArea(image, xs: xs, ys: ys)
        }
        return resizeCubic(image, xs: xs, ys: ys)
    }

    @inline(__always) private static func pixel(_ image: LiveBGRAImage, _ x: Int, _ y: Int, _ rgb: Int) -> Int32 {
        // BGRA: R at +2, G at +1, B at +0.
        Int32(image.base[y * image.bytesPerRow + x * 4 + (2 - rgb)])
    }

    /// cv2 INTER_CUBIC (A = -0.75) with 11-bit fixed-point coefficients, replicate at the crop edge.
    private static func resizeCubic(_ image: LiveBGRAImage, xs: [Int], ys: [Int]) -> [UInt8] {
        let size = cropSize
        func table(_ source: Int) -> (index: [Int], coef: [Int32]) {
            let scale = 1.0 / (Double(size) / Double(source))   // cv2: scale_x = 1/inv_scale_x
            var index = [Int](repeating: 0, count: size * 4)
            var coef = [Int32](repeating: 0, count: size * 4)
            for d in 0..<size {
                var f = Float((Double(d) + 0.5) * scale - 0.5)
                let s = Int(f.rounded(.down))
                f -= Float(s)
                let a: Float = -0.75
                var c = [Float](repeating: 0, count: 4)
                c[0] = ((a * (f + 1) - 5 * a) * (f + 1) + 8 * a) * (f + 1) - 4 * a
                c[1] = ((a + 2) * f - (a + 3)) * f * f + 1
                c[2] = ((a + 2) * (1 - f) - (a + 3)) * (1 - f) * (1 - f) + 1
                c[3] = 1 - c[0] - c[1] - c[2]
                for k in 0..<4 {
                    index[d * 4 + k] = min(max(s + k - 1, 0), source - 1)
                    coef[d * 4 + k] = Int32((c[k] * 2048).rounded(.toNearestOrEven))
                }
            }
            return (index, coef)
        }
        let tx = table(xs.count), ty = table(ys.count)
        // Horizontal pass on every source row that a vertical tap uses.
        var rows = [Int32](repeating: 0, count: ys.count * size * 3)
        let needed = Set(ty.index)
        for sy in needed {
            let y = ys[sy]
            for d in 0..<size {
                for rgb in 0..<3 {
                    var sum: Int32 = 0
                    for k in 0..<4 { sum += pixel(image, xs[tx.index[d * 4 + k]], y, rgb) * tx.coef[d * 4 + k] }
                    rows[(sy * size + d) * 3 + rgb] = sum
                }
            }
        }
        var out = [UInt8](repeating: 0, count: size * size * 3)
        for d in 0..<size {
            for x in 0..<size {
                for rgb in 0..<3 {
                    var sum: Int64 = 0
                    for k in 0..<4 {
                        sum += Int64(rows[(ty.index[d * 4 + k] * size + x) * 3 + rgb]) * Int64(ty.coef[d * 4 + k])
                    }
                    let value = (sum + (1 << 21)) >> 22
                    out[(d * size + x) * 3 + rgb] = UInt8(clamping: value)
                }
            }
        }
        return out
    }

    /// cv2 INTER_AREA for a shrinking resize (computeResizeAreaTab weights, float accumulation).
    private static func resizeArea(_ image: LiveBGRAImage, xs: [Int], ys: [Int]) -> [UInt8] {
        let size = cropSize
        func table(_ source: Int) -> [[(Int, Float)]] {
            let scale = 1.0 / (Double(size) / Double(source))   // cv2: scale_x = 1/inv_scale_x
            return (0..<size).map { d in
                var taps: [(Int, Float)] = []
                let fsx1 = Double(d) * scale, fsx2 = fsx1 + scale
                let cell = min(scale, Double(source) - fsx1)
                var sx1 = Int(ceil(fsx1)), sx2 = Int(floor(fsx2))
                sx2 = min(sx2, source - 1)
                sx1 = min(sx1, sx2)
                if Double(sx1) - fsx1 > 1e-3 { taps.append((sx1 - 1, Float((Double(sx1) - fsx1) / cell))) }
                if sx1 < sx2 { for sx in sx1..<sx2 { taps.append((sx, Float(1 / cell))) } }
                if fsx2 - Double(sx2) > 1e-3 {
                    taps.append((sx2, Float(min(min(fsx2 - Double(sx2), 1), cell) / cell)))
                }
                return taps
            }
        }
        let tx = table(xs.count), ty = table(ys.count)
        var rows = [Float](repeating: 0, count: ys.count * size * 3)
        let needed = Set(ty.flatMap { $0.map(\.0) })
        for sy in needed {
            let y = ys[sy]
            for d in 0..<size {
                var acc = SIMD3<Float>.zero
                for (sx, weight) in tx[d] {
                    let x = xs[sx]
                    acc += weight * SIMD3(Float(pixel(image, x, y, 0)), Float(pixel(image, x, y, 1)), Float(pixel(image, x, y, 2)))
                }
                rows[(sy * size + d) * 3] = acc.x
                rows[(sy * size + d) * 3 + 1] = acc.y
                rows[(sy * size + d) * 3 + 2] = acc.z
            }
        }
        var out = [UInt8](repeating: 0, count: size * size * 3)
        for d in 0..<size {
            for x in 0..<size {
                for rgb in 0..<3 {
                    var sum: Float = 0
                    for (sy, weight) in ty[d] { sum += weight * rows[(sy * size + x) * 3 + rgb] }
                    out[(d * size + x) * 3 + rgb] = UInt8(clamping: Int(sum.rounded(.toNearestOrEven)))
                }
            }
        }
        return out
    }
}
