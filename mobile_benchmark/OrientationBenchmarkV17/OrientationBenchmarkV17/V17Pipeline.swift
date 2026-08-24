import AVFoundation
import CoreImage
import CoreML
import Foundation
import ImageIO
import Vision

enum BenchmarkError: LocalizedError {
    case invalidVideo(String)
    case extraction(String)
    case model(String)

    var errorDescription: String? {
        switch self {
        case .invalidVideo(let value), .extraction(let value), .model(let value):
            return value
        }
    }
}

struct V17Diagnostics: Codable {
    let sourceFrames: Int
    let trimmedFrames: Int
    let observedHandFrames: Int
    let handPresenceFraction: Double
    let facePresenceFraction: Double
    let bodyPresenceFraction: Double
    let extractionMilliseconds: Double
    let visionCoarseRotationClockwise: Int
    let visionOrientationScores: [String: Double]
}

struct V17FeatureResult {
    let features: MLMultiArray
    let diagnostics: V17Diagnostics
}

private struct V17VideoSamplePlan {
    let asset: AVURLAsset
    let times: [CMTime]
    let sourceFrames: Int
}

private struct PointTrack {
    var xy = SIMD2<Float>.zero
    var confidence: Float = 0
}

private struct HandTrack {
    var points = Array(repeating: PointTrack(), count: 21)
    var chirality = "unknown"
    var score: Float = 0
}

private struct FrameTrack {
    var points = Array(repeating: PointTrack(), count: 61)
}

private struct CoarseOrientation {
    let correctionClockwise: Int
    let imageOrientation: CGImagePropertyOrientation
    let scores: [String: Double]
}

enum V17Pipeline {
    static let frameCount = 32
    static let nodeCount = 61
    static let channelCount = 5
    static let inputName = "landmarks"
    private static let maximumStage2Windows = 8
    private static let stage2SourceFramesPerWindow = 32
    private static let handFramesPerWindow = 16
    private static let handViews = 3
    private static let cropSize = 256
    private static let ciContext = CIContext(options: [.cacheIntermediates: false])

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

    static func extract(from url: URL, inputMirrored: Bool) async throws -> V17FeatureResult {
        let started = ContinuousClock.now
        let images = try await sampledImages(from: url)
        guard images.count >= 4 else {
            throw BenchmarkError.invalidVideo("At least four decodable frames are required")
        }
        let sourceWidth = images[0].width
        let sourceHeight = images[0].height
        guard images.allSatisfy({ $0.width == sourceWidth && $0.height == sourceHeight }) else {
            throw BenchmarkError.invalidVideo("Video dimensions changed within the clip")
        }

        let coarse = try selectCoarseOrientation(images, inputMirrored: inputMirrored)
        let swapsDimensions = coarse.correctionClockwise == 90
            || coarse.correctionClockwise == 270
        let width = swapsDimensions ? sourceHeight : sourceWidth
        let height = swapsDimensions ? sourceWidth : sourceHeight

        let handRequest = VNDetectHumanHandPoseRequest()
        handRequest.maximumHandCount = 2
        let bodyRequest = VNDetectHumanBodyPoseRequest()
        let faceRequest = VNDetectFaceLandmarksRequest()
        var tracks: [FrameTrack] = []
        var previousWrists: [String: SIMD2<Float>] = [:]
        var observedHandFrames = 0
        let imageOrientation = coarse.imageOrientation

        for (index, image) in images.enumerated() {
            var requests: [VNRequest] = [handRequest]
            let includeAuxiliary = index % 8 == 0
            if includeAuxiliary {
                requests.append(bodyRequest)
                requests.append(faceRequest)
            }
            let handler = VNImageRequestHandler(
                cgImage: image, orientation: imageOrientation, options: [:]
            )
            try handler.perform(requests)
            var frame = FrameTrack()
            let hands = try (handRequest.results ?? []).compactMap(parseHand)
            let assigned = assignHands(hands, previous: previousWrists)
            for (slot, range) in [("left", 0..<21), ("right", 21..<42)] {
                guard let hand = assigned[slot] else { continue }
                for (source, destination) in zip(0..<21, range) {
                    frame.points[destination] = hand.points[source]
                }
                if hand.points[0].confidence > 0 {
                    previousWrists[slot] = hand.points[0].xy
                }
            }
            if assigned.values.contains(where: { hand in
                hand.points.contains(where: { $0.confidence > 0 })
            }) {
                observedHandFrames += 1
            }
            if includeAuxiliary {
                if let body = try parseBody(bodyRequest.results ?? []) {
                    for offset in 0..<4 { frame.points[57 + offset] = body[offset] }
                }
                if let face = parseFace(faceRequest.results ?? []) {
                    for offset in 0..<15 { frame.points[42 + offset] = face[offset] }
                }
            }
            tracks.append(frame)
        }
        guard observedHandFrames >= 2 else {
            throw BenchmarkError.extraction("Apple Vision found fewer than two usable hand frames")
        }

        let untrimmedCount = tracks.count
        tracks = trimToHandActivity(tracks)
        isotropicTransform(&tracks, width: width, height: height)
        interpolate(&tracks, range: 0..<42, maximumGap: 3)
        interpolate(&tracks, range: 42..<61, maximumGap: 16)
        let normalized = normalize(tracks)
        let resampled = resample(normalized, targetCount: frameCount)
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: frameCount), NSNumber(value: nodeCount), NSNumber(value: channelCount)],
            dataType: .float32
        )
        var handPresent = 0.0
        var facePresent = 0.0
        var bodyPresent = 0.0
        for frame in 0..<frameCount {
            for node in 0..<nodeCount {
                for channel in 0..<channelCount {
                    let flat = ((frame * nodeCount + node) * channelCount + channel)
                    array[flat] = NSNumber(value: resampled[frame][node][channel])
                }
                let presence = Double(resampled[frame][node][3])
                if node < 42 { handPresent += presence }
                else if node < 57 { facePresent += presence }
                else { bodyPresent += presence }
            }
        }
        let elapsed = started.duration(to: .now)
        let milliseconds = Double(elapsed.components.seconds) * 1_000
            + Double(elapsed.components.attoseconds) / 1.0e15
        return V17FeatureResult(
            features: array,
            diagnostics: V17Diagnostics(
                sourceFrames: untrimmedCount,
                trimmedFrames: tracks.count,
                observedHandFrames: observedHandFrames,
                handPresenceFraction: handPresent / Double(frameCount * 42),
                facePresenceFraction: facePresent / Double(frameCount * 15),
                bodyPresenceFraction: bodyPresent / Double(frameCount * 4),
                extractionMilliseconds: milliseconds,
                visionCoarseRotationClockwise: coarse.correctionClockwise,
                visionOrientationScores: coarse.scores
            )
        )
    }

    static func prepareStage2(
        from url: URL, inputMirrored: Bool
    ) async throws -> (input: Stage2PreparedInput, diagnostics: V17Diagnostics) {
        let started = ContinuousClock.now
        let plan = try await samplePlan(from: url, maximumFrames: 256)
        guard plan.times.count >= 4 else {
            throw BenchmarkError.invalidVideo("At least four decodable frames are required")
        }
        let probeIndices = [0.25, 0.5, 0.75].map { fraction in
            Int((fraction * Double(plan.times.count - 1)).rounded())
        }
        let probeImages = try await loadImages(plan: plan, indices: probeIndices)
        let coarse = try selectCoarseOrientation(
            probeImages, inputMirrored: inputMirrored
        )
        let ranges = stride(from: 0, to: plan.times.count, by: stage2SourceFramesPerWindow)
            .compactMap { start -> Range<Int>? in
                let end = min(plan.times.count, start + stage2SourceFramesPerWindow)
                return end - start >= 4 ? start..<end : nil
            }
        guard !ranges.isEmpty, ranges.count <= maximumStage2Windows else {
            throw BenchmarkError.invalidVideo("Stage-2 requires one to eight usable windows")
        }

        let landmarks = try MLMultiArray(
            shape: [1, 8, 32, 61, 5].map(NSNumber.init(value:)), dataType: .float32
        )
        let handValid = try MLMultiArray(
            shape: [1, 8, 16, 3].map(NSNumber.init(value:)), dataType: .float32
        )
        let handBoxes = try MLMultiArray(
            shape: [1, 8, 16, 3, 4].map(NSNumber.init(value:)), dataType: .float32
        )
        let windowMask = try MLMultiArray(
            shape: [1, 8].map(NSNumber.init(value:)), dataType: .float32
        )
        for index in 0..<landmarks.count { landmarks[index] = 0 }
        for index in 0..<handValid.count { handValid[index] = 0 }
        for index in 0..<handBoxes.count { handBoxes[index] = 0 }
        for index in 0..<windowMask.count { windowMask[index] = 0 }

        var crops: [Stage2CropRecord] = []
        var observedHandFrames = 0
        var handPresence = 0.0
        var facePresence = 0.0
        var bodyPresence = 0.0
        var referenceWidth: Int?
        var referenceHeight: Int?
        for (windowIndex, range) in ranges.enumerated() {
            let raw = try await loadImages(plan: plan, indices: Array(range))
            let upright = try raw.map { try orientedImage($0, orientation: coarse.imageOrientation) }
            guard let first = upright.first,
                  upright.allSatisfy({ $0.width == first.width && $0.height == first.height }) else {
                throw BenchmarkError.invalidVideo("Video dimensions changed within a Stage-2 window")
            }
            if let width = referenceWidth, let height = referenceHeight,
               (width != first.width || height != first.height) {
                throw BenchmarkError.invalidVideo("Video dimensions changed between Stage-2 windows")
            }
            referenceWidth = first.width
            referenceHeight = first.height

            let extracted = try extractWindowFeatures(upright)
            observedHandFrames += extracted.observedHandFrames
            if let features = extracted.features {
                for frame in 0..<frameCount {
                    for node in 0..<nodeCount {
                        for channel in 0..<channelCount {
                            let source = (frame * nodeCount + node) * channelCount + channel
                            let destination = (((windowIndex * frameCount + frame) * nodeCount + node) * channelCount + channel)
                            landmarks[destination] = NSNumber(value: features[source])
                        }
                        let presence = Double(features[(frame * nodeCount + node) * channelCount + 3])
                        if node < 42 { handPresence += presence }
                        else if node < 57 { facePresence += presence }
                        else { bodyPresence += presence }
                    }
                }
            }
            let handResult = try extractHandCrops(upright, window: windowIndex)
            crops.append(contentsOf: handResult.crops)
            for frame in 0..<handFramesPerWindow {
                for view in 0..<handViews {
                    let flat = (windowIndex * handFramesPerWindow + frame) * handViews + view
                    handValid[flat] = NSNumber(value: handResult.valid[frame * handViews + view])
                    for coordinate in 0..<4 {
                        let boxFlat = flat * 4 + coordinate
                        handBoxes[boxFlat] = NSNumber(
                            value: handResult.boxes[(frame * handViews + view) * 4 + coordinate]
                        )
                    }
                }
            }
            windowMask[windowIndex] = 1
        }
        guard observedHandFrames >= 2, !crops.isEmpty else {
            throw BenchmarkError.extraction("Apple Vision found fewer than two usable hand frames")
        }
        let elapsed = Self.milliseconds(started.duration(to: .now))
        let windowFrameTotal = Double(ranges.count * frameCount)
        let diagnostics = V17Diagnostics(
            sourceFrames: plan.sourceFrames,
            trimmedFrames: plan.times.count,
            observedHandFrames: observedHandFrames,
            handPresenceFraction: handPresence / (windowFrameTotal * 42),
            facePresenceFraction: facePresence / (windowFrameTotal * 15),
            bodyPresenceFraction: bodyPresence / (windowFrameTotal * 4),
            extractionMilliseconds: elapsed,
            visionCoarseRotationClockwise: coarse.correctionClockwise,
            visionOrientationScores: coarse.scores
        )
        return (
            Stage2PreparedInput(
                landmarks: landmarks,
                handValid: handValid,
                handBoxes: handBoxes,
                windowMask: windowMask,
                crops: crops,
                windows: ranges.count
            ),
            diagnostics
        )
    }

    private static func extractWindowFeatures(
        _ images: [CGImage]
    ) throws -> (features: [Float]?, observedHandFrames: Int) {
        let handRequest = VNDetectHumanHandPoseRequest()
        handRequest.maximumHandCount = 2
        let bodyRequest = VNDetectHumanBodyPoseRequest()
        let faceRequest = VNDetectFaceLandmarksRequest()
        var tracks: [FrameTrack] = []
        var previousWrists: [String: SIMD2<Float>] = [:]
        var observed = 0
        for (index, image) in images.enumerated() {
            var requests: [VNRequest] = [handRequest]
            let auxiliary = index % 8 == 0
            if auxiliary { requests.append(contentsOf: [bodyRequest, faceRequest]) }
            try VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])
                .perform(requests)
            var frame = FrameTrack()
            let assigned = assignHands(
                try (handRequest.results ?? []).compactMap(parseHand),
                previous: previousWrists
            )
            for (slot, range) in [("left", 0..<21), ("right", 21..<42)] {
                guard let hand = assigned[slot] else { continue }
                for (source, destination) in zip(0..<21, range) {
                    frame.points[destination] = hand.points[source]
                }
                if hand.points[0].confidence > 0 { previousWrists[slot] = hand.points[0].xy }
            }
            if assigned.values.contains(where: { $0.points.contains(where: { $0.confidence > 0 }) }) {
                observed += 1
            }
            if auxiliary {
                if let body = try parseBody(bodyRequest.results ?? []) {
                    for offset in 0..<4 { frame.points[57 + offset] = body[offset] }
                }
                if let face = parseFace(faceRequest.results ?? []) {
                    for offset in 0..<15 { frame.points[42 + offset] = face[offset] }
                }
            }
            tracks.append(frame)
        }
        guard observed >= 2, let first = images.first else { return (nil, observed) }
        isotropicTransform(&tracks, width: first.width, height: first.height)
        interpolate(&tracks, range: 0..<42, maximumGap: 3)
        interpolate(&tracks, range: 42..<61, maximumGap: 16)
        let output = resample(normalize(tracks), targetCount: frameCount)
        return (output.flatMap { $0.flatMap { $0 } }, observed)
    }

    private static func extractHandCrops(
        _ images: [CGImage], window: Int
    ) throws -> (crops: [Stage2CropRecord], valid: [Float], boxes: [Float]) {
        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 2
        let indices = sampleIndices(images.count, count: handFramesPerWindow)
        var previousWrists: [String: SIMD2<Float>] = [:]
        var crops: [Stage2CropRecord] = []
        var valid = Array(repeating: Float(0), count: handFramesPerWindow * handViews)
        var boxes = Array(repeating: Float(0), count: handFramesPerWindow * handViews * 4)
        for (frameIndex, sourceIndex) in indices.enumerated() {
            let image = images[sourceIndex]
            try VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])
                .perform([request])
            let assigned = assignHands(
                try (request.results ?? []).compactMap(parseHand), previous: previousWrists
            )
            var observedBoxes: [CGRect] = []
            for (view, slot) in ["left", "right"].enumerated() {
                guard let hand = assigned[slot],
                      let box = handBox(hand, width: image.width, height: image.height) else { continue }
                if hand.points[0].confidence > 0 { previousWrists[slot] = hand.points[0].xy }
                observedBoxes.append(box)
                setBox(box, frame: frameIndex, view: view, width: image.width,
                       height: image.height, valid: &valid, boxes: &boxes)
                crops.append(Stage2CropRecord(
                    window: window, frame: frameIndex, view: view,
                    image: try cropSquare(image, box: box)
                ))
            }
            if let box = unionBox(observedBoxes) {
                setBox(box, frame: frameIndex, view: 2, width: image.width,
                       height: image.height, valid: &valid, boxes: &boxes)
                crops.append(Stage2CropRecord(
                    window: window, frame: frameIndex, view: 2,
                    image: try cropSquare(image, box: box)
                ))
            }
        }
        return (crops, valid, boxes)
    }

    private static func sampleIndices(_ frameCount: Int, count: Int) -> [Int] {
        guard count > 1 else { return [0] }
        return (0..<count).map {
            Int((Double($0) * Double(frameCount - 1) / Double(count - 1)).rounded())
        }
    }

    private static func handBox(_ hand: HandTrack, width: Int, height: Int) -> CGRect? {
        let points = hand.points.filter { $0.confidence > 0 }
        guard points.count >= 5 else { return nil }
        let xs = points.map { CGFloat($0.xy.x) }
        let ys = points.map { CGFloat($0.xy.y) }
        let x0 = xs.min()!, x1 = xs.max()!, y0 = ys.min()!, y1 = ys.max()!
        let centerX = 0.5 * (x0 + x1) * CGFloat(width)
        let centerY = 0.5 * (y0 + y1) * CGFloat(height)
        let detected = max((x1 - x0) * CGFloat(width), (y1 - y0) * CGFloat(height))
        let side = max(detected * 1.70, 0.14 * CGFloat(max(width, height)))
        return CGRect(x: centerX - side / 2, y: centerY - side / 2, width: side, height: side)
    }

    private static func unionBox(_ boxes: [CGRect]) -> CGRect? {
        guard var bounds = boxes.first else { return nil }
        for box in boxes.dropFirst() { bounds = bounds.union(box) }
        let side = max(bounds.width, bounds.height) * 1.20
        return CGRect(
            x: bounds.midX - side / 2, y: bounds.midY - side / 2,
            width: side, height: side
        )
    }

    private static func setBox(
        _ box: CGRect, frame: Int, view: Int, width: Int, height: Int,
        valid: inout [Float], boxes: inout [Float]
    ) {
        let flat = frame * handViews + view
        valid[flat] = 1
        let values = [
            Float(box.minX / CGFloat(width)), Float(box.minY / CGFloat(height)),
            Float(box.maxX / CGFloat(width)), Float(box.maxY / CGFloat(height)),
        ]
        for coordinate in 0..<4 { boxes[flat * 4 + coordinate] = values[coordinate] }
    }

    private static func cropSquare(_ image: CGImage, box: CGRect) throws -> CGImage {
        let source = CIImage(cgImage: image).clampedToExtent()
        let ciBox = CGRect(
            x: box.minX,
            y: CGFloat(image.height) - box.maxY,
            width: box.width,
            height: box.height
        )
        let scale = CGFloat(cropSize) / max(ciBox.width, 1)
        let transformed = source
            .cropped(to: ciBox)
            .transformed(by: CGAffineTransform(translationX: -ciBox.minX, y: -ciBox.minY))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let output = ciContext.createCGImage(
            transformed, from: CGRect(x: 0, y: 0, width: cropSize, height: cropSize)
        ) else { throw BenchmarkError.extraction("Could not create a 256x256 hand crop") }
        return output
    }

    private static func orientedImage(
        _ image: CGImage, orientation: CGImagePropertyOrientation
    ) throws -> CGImage {
        let value = CIImage(cgImage: image).oriented(forExifOrientation: Int32(orientation.rawValue))
        guard let output = ciContext.createCGImage(value, from: value.extent) else {
            throw BenchmarkError.extraction("Could not apply the selected video orientation")
        }
        return output
    }

    private static func selectCoarseOrientation(
        _ images: [CGImage], inputMirrored: Bool
    ) throws -> CoarseOrientation {
        let corrections = [0, 90, 180, 270]
        let probeIndices = images.count <= 3
            ? Array(images.indices)
            : [0.25, 0.5, 0.75].map { fraction in
                Int((fraction * Double(images.count - 1)).rounded())
            }
        var scores: [String: Double] = [:]
        var bodyScores: [Int: Double] = [:]
        var faceScores: [Int: Double] = [:]
        var axisScores: [Int: Double] = [:]
        for correction in corrections {
            let orientation = imageOrientation(
                correctionClockwise: correction, mirrored: inputMirrored
            )
            var bodyValues: [Double] = []
            var faceValues: [Double] = []
            var axisValues: [Double] = []
            for index in probeIndices {
                let bodyRequest = VNDetectHumanBodyPoseRequest()
                let faceRequest = VNDetectFaceLandmarksRequest()
                let handler = VNImageRequestHandler(
                    cgImage: images[index], orientation: orientation, options: [:]
                )
                try handler.perform([bodyRequest, faceRequest])
                let body = try parseBody(bodyRequest.results ?? [])
                let face = parseFace(faceRequest.results ?? [])
                let components = orientationScoreComponents(body: body, face: face)
                bodyValues.append(components.body)
                faceValues.append(components.face)
                axisValues.append(components.axis)
            }
            let bodyMean = bodyValues.reduce(0, +) / Double(bodyValues.count)
            let faceMean = faceValues.reduce(0, +) / Double(faceValues.count)
            let axisMean = axisValues.reduce(0, +) / Double(axisValues.count)
            bodyScores[correction] = bodyMean
            faceScores[correction] = faceMean
            axisScores[correction] = axisMean
            scores["\(correction)_body"] = bodyMean
            scores["\(correction)_face"] = faceMean
            scores["\(correction)_axis"] = axisMean
            scores["\(correction)_combined"] = bodyMean + faceMean
        }
        let families = [[0, 180], [90, 270]]
        let selectedFamily = families.max { left, right in
            let leftAxis = left.map { axisScores[$0] ?? 0 }.max() ?? 0
            let rightAxis = right.map { axisScores[$0] ?? 0 }.max() ?? 0
            if leftAxis != rightAxis { return leftAxis < rightAxis }
            let leftBody = left.map { bodyScores[$0] ?? 0 }.max() ?? 0
            let rightBody = right.map { bodyScores[$0] ?? 0 }.max() ?? 0
            return leftBody < rightBody
        } ?? [0, 180]
        let selected = selectedFamily.max { left, right in
            let leftFace = faceScores[left] ?? 0
            let rightFace = faceScores[right] ?? 0
            if leftFace != rightFace { return leftFace < rightFace }
            return (bodyScores[left] ?? 0) < (bodyScores[right] ?? 0)
        } ?? 0
        return CoarseOrientation(
            correctionClockwise: selected,
            imageOrientation: imageOrientation(
                correctionClockwise: selected, mirrored: inputMirrored
            ),
            scores: scores
        )
    }

    private static func imageOrientation(
        correctionClockwise: Int, mirrored: Bool
    ) -> CGImagePropertyOrientation {
        switch (correctionClockwise, mirrored) {
        case (0, false): return .up
        case (90, false): return .right
        case (180, false): return .down
        case (270, false): return .left
        case (0, true): return .upMirrored
        case (90, true): return .rightMirrored
        case (180, true): return .downMirrored
        case (270, true): return .leftMirrored
        default: return mirrored ? .upMirrored : .up
        }
    }

    private static func orientationScoreComponents(
        body: [PointTrack]?, face: [PointTrack]?
    ) -> (body: Double, face: Double, axis: Double) {
        let bodyScore = Double(
            (body ?? []).map(\.confidence).reduce(0, +) / Float(max(body?.count ?? 0, 1))
        )
        let axisPoints: [PointTrack]
        if let body, body.count >= 2,
           body[0].confidence > 0, body[1].confidence > 0 {
            axisPoints = [body[0], body[1]]
        } else if let face, face.count >= 2,
                  face[0].confidence > 0, face[1].confidence > 0 {
            axisPoints = [face[0], face[1]]
        } else {
            axisPoints = []
        }
        var axisScore = 0.0
        if axisPoints.count == 2 {
            let delta = axisPoints[1].xy - axisPoints[0].xy
            let angle = Double(atan2(delta.y, delta.x)) * 180 / Double.pi
            let remainder = (angle + 90).truncatingRemainder(dividingBy: 180)
            let wrapped = remainder < 0 ? remainder + 180 : remainder
            let folded = abs(wrapped - 90)
            axisScore = 1 - folded / 90
        }
        guard let face else { return (bodyScore, 0, axisScore) }
        let eyes = face[0..<2].filter { $0.confidence > 0 }
        let mouth = face[7..<11].filter { $0.confidence > 0 }
        guard !eyes.isEmpty, !mouth.isEmpty else { return (bodyScore, 0, axisScore) }
        let eyeY = eyes.map { Double($0.xy.y) }.reduce(0, +) / Double(eyes.count)
        let mouthY = mouth.map { Double($0.xy.y) }.reduce(0, +) / Double(mouth.count)
        return (bodyScore, 10 * (mouthY - eyeY), axisScore)
    }

    private static func sampledImages(from url: URL) async throws -> [CGImage] {
        let plan = try await samplePlan(from: url, maximumFrames: 96)
        return try await loadImages(plan: plan, indices: Array(plan.times.indices))
    }

    private static func samplePlan(
        from url: URL, maximumFrames: Int
    ) async throws -> V17VideoSamplePlan {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration)
        let seconds = CMTimeGetSeconds(duration)
        guard seconds.isFinite, seconds > 0 else {
            throw BenchmarkError.invalidVideo("Video duration is invalid")
        }
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first else {
            throw BenchmarkError.invalidVideo("File has no video track")
        }
        let frameRate = max(Double(try await track.load(.nominalFrameRate)), 1.0)
        let estimatedFrames = max(4, Int((seconds * frameRate).rounded()))
        let count = min(maximumFrames, estimatedFrames)
        let end = max(0, seconds - 1.0 / frameRate)
        let times = (0..<count).map { index in
            let fraction = count == 1 ? 0 : Double(index) / Double(count - 1)
            return CMTime(seconds: fraction * end, preferredTimescale: 600)
        }
        return V17VideoSamplePlan(asset: asset, times: times, sourceFrames: estimatedFrames)
    }

    private static func loadImages(
        plan: V17VideoSamplePlan, indices: [Int]
    ) async throws -> [CGImage] {
        guard !indices.isEmpty, indices.allSatisfy({ plan.times.indices.contains($0) }) else {
            throw BenchmarkError.invalidVideo("Invalid Stage-2 video sample indices")
        }
        let generator = AVAssetImageGenerator(asset: plan.asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 1280, height: 1280)
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.02, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.02, preferredTimescale: 600)
        return try await Task.detached(priority: .userInitiated) {
            var output: [CGImage] = []
            output.reserveCapacity(indices.count)
            for index in indices {
                let image = try generator.copyCGImage(at: plan.times[index], actualTime: nil)
                output.append(image)
            }
            return output
        }.value
    }

    private static func milliseconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) * 1_000
            + Double(duration.components.attoseconds) / 1.0e15
    }

    private static func parseHand(_ observation: VNHumanHandPoseObservation) throws -> HandTrack? {
        let recognized = try observation.recognizedPoints(.all)
        var hand = HandTrack()
        for (index, joint) in handJoints.enumerated() {
            guard let point = recognized[joint], point.confidence >= 0.15 else { continue }
            hand.points[index] = PointTrack(
                xy: SIMD2(Float(point.location.x), Float(1 - point.location.y)),
                confidence: point.confidence
            )
        }
        guard hand.points.filter({ $0.confidence > 0 }).count >= 5 else { return nil }
        let valid = hand.points.filter { $0.confidence > 0 }
        hand.score = valid.map(\.confidence).reduce(0, +) / Float(valid.count)
        switch observation.chirality {
        case .left: hand.chirality = "left"
        case .right: hand.chirality = "right"
        default: hand.chirality = "unknown"
        }
        return hand
    }

    private static func parseBody(
        _ observations: [VNHumanBodyPoseObservation]
    ) throws -> [PointTrack]? {
        guard let observation = observations.max(by: { $0.confidence < $1.confidence }) else {
            return nil
        }
        let recognized = try observation.recognizedPoints(.all)
        return bodyJoints.map { joint in
            guard let point = recognized[joint], point.confidence >= 0.15 else {
                return PointTrack()
            }
            return PointTrack(
                xy: SIMD2(Float(point.location.x), Float(1 - point.location.y)),
                confidence: point.confidence
            )
        }
    }

    private static func parseFace(_ observations: [VNFaceObservation]) -> [PointTrack]? {
        guard let observation = observations.max(by: {
            $0.boundingBox.width * $0.boundingBox.height
                < $1.boundingBox.width * $1.boundingBox.height
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
        return specifications.map { region, requestedIndex in
            guard let region else { return PointTrack() }
            let points = region.normalizedPoints
            let index = requestedIndex >= 0 ? requestedIndex : points.count + requestedIndex
            guard points.indices.contains(index) else { return PointTrack() }
            let local = points[index]
            let box = observation.boundingBox
            return PointTrack(
                xy: SIMD2(
                    Float(box.minX + local.x * box.width),
                    Float(1 - (box.minY + local.y * box.height))
                ),
                confidence: max(observation.confidence, 0.15)
            )
        }
    }

    private static func handCost(
        _ hand: HandTrack, slot: String, previous: SIMD2<Float>?
    ) -> Float {
        var cost = -0.1 * hand.score
        if hand.chirality != "unknown" {
            cost += hand.chirality == slot ? 0 : 2
        } else if hand.points[0].confidence > 0 {
            let expectedLeft = hand.points[0].xy.x >= 0.5
            if (slot == "left") != expectedLeft { cost += 0.2 }
        }
        if let previous, hand.points[0].confidence > 0 {
            cost += simd_distance(hand.points[0].xy, previous)
        }
        return cost
    }

    private static func assignHands(
        _ hands: [HandTrack], previous: [String: SIMD2<Float>]
    ) -> [String: HandTrack] {
        let usable = Array(hands.sorted { $0.score > $1.score }.prefix(2))
        var output: [String: HandTrack] = [:]
        guard let first = usable.first else { return output }
        if usable.count == 1 {
            let leftCost = handCost(first, slot: "left", previous: previous["left"])
            let rightCost = handCost(first, slot: "right", previous: previous["right"])
            output[leftCost <= rightCost ? "left" : "right"] = first
            return output
        }
        let second = usable[1]
        let direct = handCost(first, slot: "left", previous: previous["left"])
            + handCost(second, slot: "right", previous: previous["right"])
        let swapped = handCost(first, slot: "right", previous: previous["right"])
            + handCost(second, slot: "left", previous: previous["left"])
        if direct <= swapped {
            output["left"] = first; output["right"] = second
        } else {
            output["left"] = second; output["right"] = first
        }
        return output
    }

    private static func trimToHandActivity(_ input: [FrameTrack]) -> [FrameTrack] {
        let active = input.indices.filter { index in
            input[index].points[0..<42].filter { $0.confidence > 0 }.count >= 5
        }
        guard let first = active.first, let last = active.last else { return input }
        let start = max(0, first - 2)
        let end = min(input.count, last + 3)
        return end - start >= 4 ? Array(input[start..<end]) : input
    }

    private static func isotropicTransform(
        _ frames: inout [FrameTrack], width: Int, height: Int
    ) {
        let longest = Float(max(width, height))
        for frame in frames.indices {
            for node in 0..<nodeCount where frames[frame].points[node].confidence > 0 {
                let value = frames[frame].points[node].xy
                frames[frame].points[node].xy = SIMD2(
                    (value.x * Float(width) - Float(width) / 2) / longest,
                    (value.y * Float(height) - Float(height) / 2) / longest
                )
            }
        }
    }

    private static func interpolate(
        _ frames: inout [FrameTrack], range: Range<Int>, maximumGap: Int
    ) {
        for node in range {
            let valid = frames.indices.filter { frames[$0].points[node].confidence > 0 }
            for pair in zip(valid, valid.dropFirst()) {
                let gap = pair.1 - pair.0 - 1
                guard gap > 0, gap <= maximumGap else { continue }
                let start = frames[pair.0].points[node]
                let end = frames[pair.1].points[node]
                let confidence = min(start.confidence, end.confidence) * 0.5
                for offset in 1...gap {
                    let fraction = Float(offset) / Float(gap + 1)
                    frames[pair.0 + offset].points[node] = PointTrack(
                        xy: start.xy + fraction * (end.xy - start.xy),
                        confidence: confidence
                    )
                }
            }
        }
    }

    private static func median(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2)
            ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    }

    private static func medianPoint(_ values: [SIMD2<Float>]) -> SIMD2<Float> {
        SIMD2(median(values.map(\.x)), median(values.map(\.y)))
    }

    private static func interpolatedCenters(
        count: Int, known: [(Int, SIMD2<Float>)]
    ) -> [SIMD2<Float>] {
        guard !known.isEmpty else { return Array(repeating: .zero, count: count) }
        return (0..<count).map { frame in
            if frame <= known[0].0 { return known[0].1 }
            if frame >= known.last!.0 { return known.last!.1 }
            let right = known.firstIndex { $0.0 >= frame }!
            let left = right - 1
            let a = known[left], b = known[right]
            let fraction = Float(frame - a.0) / Float(b.0 - a.0)
            return a.1 + fraction * (b.1 - a.1)
        }
    }

    private static func normalize(_ frames: [FrameTrack]) -> [[[Float]]] {
        var shoulderWidths: [(Int, Float)] = []
        var shoulderCenters: [(Int, SIMD2<Float>)] = []
        for frame in frames.indices {
            let left = frames[frame].points[57], right = frames[frame].points[58]
            guard left.confidence > 0, right.confidence > 0 else { continue }
            let width = simd_distance(left.xy, right.xy)
            guard width > 1e-5 else { continue }
            shoulderWidths.append((frame, width))
            shoulderCenters.append((frame, (left.xy + right.xy) / 2))
        }
        var centers: [SIMD2<Float>]
        var scale: Float
        if !shoulderWidths.isEmpty {
            centers = interpolatedCenters(count: frames.count, known: shoulderCenters)
            scale = median(shoulderWidths.map(\.1))
        } else {
            let wrists = frames.flatMap { frame in
                [frame.points[0], frame.points[21]].compactMap {
                    $0.confidence > 0 ? $0.xy : nil
                }
            }
            centers = Array(repeating: medianPoint(wrists), count: frames.count)
            var palms: [Float] = []
            for frame in frames {
                for start in [0, 21] {
                    let wrist = frame.points[start], mcp = frame.points[start + 9]
                    if wrist.confidence > 0, mcp.confidence > 0 {
                        let length = simd_distance(wrist.xy, mcp.xy)
                        if length > 1e-5 { palms.append(length) }
                    }
                }
            }
            scale = palms.isEmpty ? 1 : median(palms)
        }
        if !scale.isFinite || scale <= 1e-5 { scale = 1 }
        var output = Array(
            repeating: Array(repeating: Array(repeating: Float(0), count: 5), count: nodeCount),
            count: frames.count
        )
        var handPalmLengths = [[Float](), [Float]()]
        for frame in frames {
            for (hand, start) in [0, 21].enumerated() {
                let wrist = frame.points[start], mcp = frame.points[start + 9]
                if wrist.confidence > 0, mcp.confidence > 0 {
                    let length = simd_distance(wrist.xy, mcp.xy)
                    if length > 1e-5 { handPalmLengths[hand].append(length) }
                }
            }
        }
        let palmReference = handPalmLengths.map { median($0) }
        let shoulderReference = median(shoulderWidths.map(\.1))
        for frame in frames.indices {
            for node in 0..<nodeCount {
                let point = frames[frame].points[node]
                guard point.confidence > 0 else { continue }
                let xy = (point.xy - centers[frame]) / scale
                output[frame][node][0] = xy.x
                output[frame][node][1] = xy.y
                output[frame][node][3] = 1
                output[frame][node][4] = min(max(point.confidence, 0), 1)
            }
            for (hand, start) in [0, 21].enumerated() {
                let wrist = frames[frame].points[start], mcp = frames[frame].points[start + 9]
                if wrist.confidence > 0, mcp.confidence > 0, palmReference[hand] > 0 {
                    let length = simd_distance(wrist.xy, mcp.xy)
                    if length > 1e-5 {
                        let depth = log(palmReference[hand] / length)
                        for node in start..<(start + 21) where output[frame][node][3] > 0 {
                            output[frame][node][2] = depth
                        }
                    }
                }
            }
            if shoulderReference > 0,
               frames[frame].points[57].confidence > 0,
               frames[frame].points[58].confidence > 0 {
                let width = simd_distance(
                    frames[frame].points[57].xy, frames[frame].points[58].xy
                )
                if width > 1e-5 {
                    let depth = log(shoulderReference / width)
                    for node in 42..<61 where output[frame][node][3] > 0 {
                        output[frame][node][2] = depth
                    }
                }
            }
        }
        return output
    }

    private static func resample(_ input: [[[Float]]], targetCount: Int) -> [[[Float]]] {
        if input.count == targetCount {
            return quantized(input)
        }
        var output = Array(
            repeating: Array(repeating: Array(repeating: Float(0), count: 5), count: nodeCount),
            count: targetCount
        )
        for target in 0..<targetCount {
            let position = Float(target) * Float(input.count - 1) / Float(targetCount - 1)
            let left = Int(floor(position))
            let right = min(left + 1, input.count - 1)
            let fraction = position - Float(left)
            let nearest = Int(position.rounded())
            for node in 0..<nodeCount {
                let presence: Float = input[nearest][node][3] >= 0.5 ? 1 : 0
                output[target][node][3] = presence
                guard presence > 0 else { continue }
                for channel in [0, 1, 2, 4] {
                    output[target][node][channel] = input[left][node][channel]
                        + fraction * (input[right][node][channel] - input[left][node][channel])
                }
            }
        }
        return quantized(output)
    }

    private static func quantized(_ input: [[[Float]]]) -> [[[Float]]] {
        input.map { frame in
            frame.map { channels in channels.map { Float(Float16($0)) } }
        }
    }
}
