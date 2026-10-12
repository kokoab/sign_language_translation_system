import CoreML
import CoreVideo
import Foundation
import ImageIO

struct Stage2MobileManifest: Codable {
    let format: String
    let version: Int
    let candidateID: String
    let checkpointSHA256: String
    let vocabularyManifestSHA256: String
    let mobileCLIP2CheckpointSHA256: String
    let imageEncoderPackageTreeSHA256: String
    let frozenEncoderPackageTreeSHA256: String
    let stage1PackageTreeSHA256: String
    let phraseHeadPackageTreeSHA256: String
    let isolatedCorrectionPackageTreeSHA256: String
    let stage1CheckpointSHA256: String
    let phraseCheckpointSHA256: String
    let isolatedCorrectionCheckpointSHA256: String
    let blankIndex: Int
    let maximumWindows: Int
    let tokensPerWindow: Int
}

struct Stage2CropRecord {
    let window: Int
    let frame: Int
    let view: Int
    let image: CGImage
}

struct Stage2PreparedInput {
    let landmarks: MLMultiArray
    let rawLandmarks: MLMultiArray
    let sourceFrameMask: MLMultiArray
    let handValid: MLMultiArray
    let handBoxes: MLMultiArray
    let windowMask: MLMultiArray
    let crops: [Stage2CropRecord]
    let windows: Int
}

struct Stage2Prediction {
    let tokenIndices: [Int]
    let labels: [String]
    let cropEncodingMilliseconds: Double
    let coldInferenceMilliseconds: Double
    let inferenceTimingsMilliseconds: [Double]
    let modelBytes: UInt64
    let predictionAggregation: String
    let sequenceVoteCounts: [String: Int]
    let handEmbeddings: MLMultiArray
    let inferenceRoute: String
    let selectedActivityWindow: Int?
    let routeInferenceMilliseconds: Double
}

struct Stage2ActivityAlignedInput {
    let landmarks: MLMultiArray
    let handEmbeddings: MLMultiArray
    let handValid: MLMultiArray
    let handBoxes: MLMultiArray
    let windowMask: MLMultiArray
    let sourceFrameRange: Range<Int>
    let sourceWindow: Int
}

enum Stage2InputBundleV17 {
    private static let magic = Array("SLTHRGB1".utf8)
    private static let maximumWindows = 8
    private static let framesPerWindow = 16
    private static let views = 3
    private static let landmarkValuesPerWindow = 32 * 61 * 5

    static func load(landmarksURL: URL, cropBundleURL: URL) throws -> Stage2PreparedInput {
        let landmarkData = try Data(contentsOf: landmarksURL)
        guard landmarkData.count.isMultiple(of: landmarkValuesPerWindow * 4) else {
            throw BenchmarkError.model("Stage-2 landmark file has an invalid byte count")
        }
        let windows = landmarkData.count / (landmarkValuesPerWindow * 4)
        guard 1...maximumWindows ~= windows else {
            throw BenchmarkError.model("Stage-2 input must contain one to eight windows")
        }
        let landmarks = try MLMultiArray(
            shape: [1, maximumWindows, 32, 61, 5].map(NSNumber.init(value:)),
            dataType: .float32
        )
        for index in 0..<landmarks.count { landmarks[index] = 0 }
        landmarkData.withUnsafeBytes { raw in
            let values = raw.bindMemory(to: Float.self)
            for index in 0..<values.count { landmarks[index] = NSNumber(value: values[index]) }
        }

        let data = try Data(contentsOf: cropBundleURL)
        var cursor = 0
        func readBytes(_ count: Int) throws -> Data {
            guard count >= 0, cursor + count <= data.count else {
                throw BenchmarkError.model("Truncated Stage-2 hand-crop bundle")
            }
            defer { cursor += count }
            return data.subdata(in: cursor..<(cursor + count))
        }
        func readUInt32() throws -> UInt32 {
            let bytes = [UInt8](try readBytes(4))
            return UInt32(bytes[0]) | UInt32(bytes[1]) << 8
                | UInt32(bytes[2]) << 16 | UInt32(bytes[3]) << 24
        }
        func readFloat() throws -> Float { Float(bitPattern: try readUInt32()) }
        guard [UInt8](try readBytes(magic.count)) == magic else {
            throw BenchmarkError.model("Stage-2 hand-crop bundle magic is invalid")
        }
        guard Int(try readUInt32()) == windows else {
            throw BenchmarkError.model("Landmark and hand-crop window counts differ")
        }
        let valid = try MLMultiArray(
            shape: [1, maximumWindows, framesPerWindow, views].map(NSNumber.init(value:)),
            dataType: .float32
        )
        let boxes = try MLMultiArray(
            shape: [1, maximumWindows, framesPerWindow, views, 4].map(NSNumber.init(value:)),
            dataType: .float32
        )
        for index in 0..<valid.count { valid[index] = 0 }
        for index in 0..<boxes.count { boxes[index] = 0 }
        let values = windows * framesPerWindow * views
        for index in 0..<values { valid[index] = NSNumber(value: try readFloat()) }
        for index in 0..<(values * 4) { boxes[index] = NSNumber(value: try readFloat()) }
        var crops: [Stage2CropRecord] = []
        for window in 0..<windows {
            for frame in 0..<framesPerWindow {
                for view in 0..<views {
                    let length = Int(try readUInt32())
                    let jpeg = try readBytes(length)
                    let flat = (window * framesPerWindow + frame) * views + view
                    if valid[flat].floatValue > 0.5 {
                        guard length > 0,
                              let source = CGImageSourceCreateWithData(jpeg as CFData, nil),
                              let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
                              image.width == 256, image.height == 256 else {
                            throw BenchmarkError.model("A valid Stage-2 hand crop is not 256x256 JPEG")
                        }
                        crops.append(Stage2CropRecord(
                            window: window, frame: frame, view: view, image: image
                        ))
                    } else if length != 0 {
                        throw BenchmarkError.model("An invalid Stage-2 crop unexpectedly has pixels")
                    }
                }
            }
        }
        guard cursor == data.count else {
            throw BenchmarkError.model("Stage-2 hand-crop bundle has trailing bytes")
        }
        let windowMask = try MLMultiArray(
            shape: [1, maximumWindows].map(NSNumber.init(value:)), dataType: .float32
        )
        for index in 0..<windowMask.count { windowMask[index] = 0 }
        for index in 0..<windows { windowMask[index] = 1 }
        let rawLandmarks = try MLMultiArray(
            shape: [1, maximumWindows, 32, 61, 3].map(NSNumber.init(value:)),
            dataType: .float32
        )
        let sourceFrameMask = try MLMultiArray(
            shape: [1, maximumWindows, 32].map(NSNumber.init(value:)),
            dataType: .float32
        )
        for index in 0..<rawLandmarks.count { rawLandmarks[index] = 0 }
        for index in 0..<sourceFrameMask.count { sourceFrameMask[index] = 0 }
        return Stage2PreparedInput(
            landmarks: landmarks,
            rawLandmarks: rawLandmarks,
            sourceFrameMask: sourceFrameMask,
            handValid: valid,
            handBoxes: boxes,
            windowMask: windowMask,
            crops: crops,
            windows: windows
        )
    }
}

final class Stage2MobileModelV17 {
    let manifest: Stage2MobileManifest
    private let imageEncoder: MLModel
    private let frozenEncoder: MLModel
    private let phraseHead: MLModel
    private let stage1: MLModel
    private let isolatedCorrection: MLModel
    private let modelBytes: UInt64

    init(resourceRoot: URL? = nil) throws {
        func resource(_ name: String, _ extensionValue: String) -> URL? {
            if let resourceRoot {
                let candidate = resourceRoot.appendingPathComponent("\(name).\(extensionValue)")
                return FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
            }
            return Bundle.main.url(forResource: name, withExtension: extensionValue)
        }
        guard let manifestURL = resource("Stage2MobileV17_manifest", "json") else {
            throw BenchmarkError.model("Stage2MobileV17_manifest.json is not bundled")
        }
        manifest = try JSONDecoder().decode(
            Stage2MobileManifest.self, from: Data(contentsOf: manifestURL)
        )
        guard manifest.format == "slt_stage2_mobile_coreml_v17",
              manifest.version == 2,
              manifest.blankIndex == 0,
              manifest.maximumWindows == 8,
              manifest.tokensPerWindow == 8 else {
            throw BenchmarkError.model("Stage-2 mobile model manifest contract failed")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let resources = [
            "MobileCLIP2S0ImageEncoderV17FP32",
            "Stage2FrozenEncoderV17FP32",
            "Stage2PhraseV17FP32",
            "Stage1UnifiedMultimodalV17FP32",
            "Stage2IsolatedCorrectionV17FP32",
        ]
        let urls = try resources.map { name -> URL in
            guard let url = resource(name, "mlmodelc") else {
                throw BenchmarkError.model("\(name).mlmodelc is not bundled")
            }
            return url
        }
        imageEncoder = try MLModel(contentsOf: urls[0], configuration: configuration)
        frozenEncoder = try MLModel(contentsOf: urls[1], configuration: configuration)
        phraseHead = try MLModel(contentsOf: urls[2], configuration: configuration)
        stage1 = try MLModel(contentsOf: urls[3], configuration: configuration)
        isolatedCorrection = try MLModel(contentsOf: urls[4], configuration: configuration)
        modelBytes = urls.reduce(0) { $0 + Self.directoryBytes($1) }
    }

    func predict(
        input: Stage2PreparedInput,
        labels: [String],
        iterations: Int,
        warmupIterations: Int = 5
    ) throws -> Stage2Prediction {
        guard labels.count == 100, iterations >= 1, warmupIterations >= 0 else {
            throw BenchmarkError.model("Stage-2 requires 100 labels and a positive iteration count")
        }
        let embeddings = try MLMultiArray(
            shape: [1, 8, 16, 3, 512].map(NSNumber.init(value:)), dataType: .float32
        )
        for index in 0..<embeddings.count { embeddings[index] = 0 }
        let cropStarted = ContinuousClock.now
        for crop in input.crops {
            try autoreleasepool {
                let imageValue = try MLFeatureValue(
                    cgImage: crop.image,
                    pixelsWide: 256,
                    pixelsHigh: 256,
                    pixelFormatType: kCVPixelFormatType_32BGRA,
                    options: nil
                )
                let output = try imageEncoder.prediction(
                    from: MLDictionaryFeatureProvider(dictionary: ["image": imageValue])
                )
                guard let value = output.featureValue(for: "embedding")?.multiArrayValue,
                      value.count == 512 else {
                    throw BenchmarkError.model("MobileCLIP2 returned an invalid embedding")
                }
                let start = (((crop.window * 16 + crop.frame) * 3 + crop.view) * 512)
                for index in 0..<512 { embeddings[start + index] = value[index] }
            }
        }
        let cropMilliseconds = Self.milliseconds(cropStarted.duration(to: .now))
        let encoderProvider = try MLDictionaryFeatureProvider(dictionary: [
            "landmarks": input.landmarks,
            "hand_embeddings": embeddings,
            "hand_valid": input.handValid,
            "hand_boxes": input.handBoxes,
            "window_mask": input.windowMask,
        ])
        let clock = ContinuousClock()
        var started = clock.now
        var frozen = try frozenEncoder.prediction(from: encoderProvider)
        guard var features = frozen.featureValue(for: "var_2922")?.multiArrayValue else {
            throw BenchmarkError.model("Frozen Stage-2 encoder output is missing")
        }
        var headProvider = try MLDictionaryFeatureProvider(dictionary: [
            "frozen_features": features, "window_mask": input.windowMask,
        ])
        var phraseOutput = try phraseHead.prediction(from: headProvider)
        let cold = Self.milliseconds(started.duration(to: clock.now))
        var timings: [Double] = []
        timings.reserveCapacity(iterations)
        var logitSums = Array(repeating: 0.0, count: 64 * 101)
        var sequenceVoteCounts: [String: Int] = [:]
        func record(_ prediction: MLFeatureProvider, milliseconds: Double) throws {
            timings.append(milliseconds)
            guard let logits = prediction.featureValue(
                for: "var_359"
            )?.multiArrayValue, logits.count == logitSums.count else {
                throw BenchmarkError.model("Stage-2 phrase head output is invalid")
            }
            for index in 0..<logitSums.count {
                logitSums[index] += logits[index].doubleValue
            }
            let vote = Self.collapse(
                logits: logits,
                windowCount: input.windows,
                tokensPerWindow: manifest.tokensPerWindow,
                blankIndex: manifest.blankIndex
            ).map(String.init).joined(separator: ",")
            sequenceVoteCounts[vote, default: 0] += 1
        }
        if warmupIterations == 0, iterations == 1 {
            try record(phraseOutput, milliseconds: cold)
        } else {
            for _ in 0..<warmupIterations {
                frozen = try frozenEncoder.prediction(from: encoderProvider)
                features = frozen.featureValue(for: "var_2922")!.multiArrayValue!
                headProvider = try MLDictionaryFeatureProvider(dictionary: [
                    "frozen_features": features, "window_mask": input.windowMask,
                ])
                phraseOutput = try phraseHead.prediction(from: headProvider)
            }
            for _ in 0..<iterations {
                started = clock.now
                frozen = try frozenEncoder.prediction(from: encoderProvider)
                features = frozen.featureValue(for: "var_2922")!.multiArrayValue!
                headProvider = try MLDictionaryFeatureProvider(dictionary: [
                    "frozen_features": features, "window_mask": input.windowMask,
                ])
                phraseOutput = try phraseHead.prediction(from: headProvider)
                try record(phraseOutput, milliseconds: Self.milliseconds(
                    started.duration(to: clock.now)
                ))
            }
        }
        var collapsed: [Int] = []
        var previous = -1
        for time in 0..<(input.windows * manifest.tokensPerWindow) {
            let start = time * 101
            var best = 0
            for token in 1..<101 where logitSums[start + token] > logitSums[start + best] {
                best = token
            }
            if best != manifest.blankIndex, best != previous { collapsed.append(best) }
            previous = best
        }
        collapsed = Self.removeAdjacentDuplicates(collapsed)
        var inferenceRoute = "phrase_ctc"
        var selectedActivityWindow: Int?
        var routeInferenceMilliseconds = 0.0
        if collapsed.count == 1 {
            let routeStarted = clock.now
            let aligned = try Self.activityAligned(input, embeddings: embeddings)
            selectedActivityWindow = aligned.sourceWindow
            let stage1Provider = try MLDictionaryFeatureProvider(dictionary: [
                "landmarks": Self.window(
                    aligned.landmarks, at: 0, shape: [1, 32, 61, 5]
                ),
                "hand_embeddings": Self.window(
                    aligned.handEmbeddings, at: 0, shape: [1, 16, 3, 512]
                ),
                "hand_valid": Self.window(
                    aligned.handValid, at: 0, shape: [1, 16, 3]
                ),
                "hand_boxes": Self.window(
                    aligned.handBoxes, at: 0, shape: [1, 16, 3, 4]
                ),
            ])
            let stage1Output = try stage1.prediction(from: stage1Provider)
            guard let stage1Logits = stage1Output.featureValue(
                for: "var_2927"
            )?.multiArrayValue, stage1Logits.count == 100 else {
                throw BenchmarkError.model("Stage-1 hybrid logits are invalid")
            }
            let alignedEncoder = try frozenEncoder.prediction(
                from: MLDictionaryFeatureProvider(dictionary: [
                    "landmarks": aligned.landmarks,
                    "hand_embeddings": aligned.handEmbeddings,
                    "hand_valid": aligned.handValid,
                    "hand_boxes": aligned.handBoxes,
                    "window_mask": aligned.windowMask,
                ])
            )
            guard let alignedFeatures = alignedEncoder.featureValue(
                for: "var_2922"
            )?.multiArrayValue else {
                throw BenchmarkError.model("Activity-aligned Stage-2 features are missing")
            }
            let correctionProvider = try MLDictionaryFeatureProvider(dictionary: [
                "frozen_features": Self.window(
                    alignedFeatures, at: 0, shape: [1, 32, 612]
                ),
                "stage1_logits": stage1Logits,
            ])
            let correctionOutput = try isolatedCorrection.prediction(
                from: correctionProvider
            )
            guard let hybridLogits = correctionOutput.featureValue(
                for: "var_99"
            )?.multiArrayValue, hybridLogits.count == 100 else {
                throw BenchmarkError.model("Isolated hybrid logits are invalid")
            }
            var best = 0
            for index in 1..<100 where hybridLogits[index].doubleValue
                > hybridLogits[best].doubleValue {
                best = index
            }
            collapsed = [best + 1]
            inferenceRoute = "isolated_full_activity_hybrid"
            routeInferenceMilliseconds = Self.milliseconds(
                routeStarted.duration(to: clock.now)
            )
        }
        let glosses = try collapsed.map { token -> String in
            guard 1...labels.count ~= token else {
                throw BenchmarkError.model("Stage-2 emitted an out-of-vocabulary token")
            }
            return labels[token - 1]
        }
        return Stage2Prediction(
            tokenIndices: collapsed,
            labels: glosses,
            cropEncodingMilliseconds: cropMilliseconds,
            coldInferenceMilliseconds: cold,
            inferenceTimingsMilliseconds: timings,
            modelBytes: modelBytes,
            predictionAggregation: inferenceRoute == "phrase_ctc"
                ? "mean_logits_then_greedy_ctc_collapse"
                : "single_ctc_route_then_full_activity_aligned_stage1_temporal_hybrid",
            sequenceVoteCounts: sequenceVoteCounts,
            handEmbeddings: embeddings,
            inferenceRoute: inferenceRoute,
            selectedActivityWindow: selectedActivityWindow,
            routeInferenceMilliseconds: routeInferenceMilliseconds
        )
    }

    static func activityAligned(
        _ input: Stage2PreparedInput, embeddings: MLMultiArray
    ) throws -> Stage2ActivityAlignedInput {
        let landmarkFrames = input.windows * 32
        guard input.windows > 0, input.windows <= 8,
              input.landmarks.count >= landmarkFrames * 61 * 5,
              embeddings.count >= input.windows * 16 * 3 * 512 else {
            throw BenchmarkError.model("Cannot activity-align malformed Stage-2 input")
        }
        var presence = Array(repeating: 0.0, count: landmarkFrames)
        var motion = Array(repeating: 0.0, count: landmarkFrames)
        for frame in 0..<landmarkFrames {
            var present = 0.0
            var displacement = 0.0
            var pairs = 0
            for node in 0..<42 {
                let current = (frame * 61 + node) * 5
                let currentPresent = input.landmarks[current + 3].doubleValue > 0.5
                present += currentPresent ? 1 : 0
                guard frame > 0, currentPresent else { continue }
                let previous = ((frame - 1) * 61 + node) * 5
                guard input.landmarks[previous + 3].doubleValue > 0.5 else { continue }
                let dx = input.landmarks[current].doubleValue
                    - input.landmarks[previous].doubleValue
                let dy = input.landmarks[current + 1].doubleValue
                    - input.landmarks[previous + 1].doubleValue
                displacement += (dx * dx + dy * dy).squareRoot()
                pairs += 1
            }
            presence[frame] = present / 42
            motion[frame] = pairs == 0 ? 0 : displacement / Double(pairs)
        }
        let sortedMotion = motion.sorted()
        let percentile70 = sortedMotion[Int(Double(sortedMotion.count - 1) * 0.70)]
        let threshold = max(percentile70, (motion.max() ?? 0) * 0.12, 1.0e-6)
        let moving = motion.indices.filter {
            motion[$0] >= threshold && motion[$0] > 1.0e-6
        }
        let present = presence.indices.filter { presence[$0] > 0.08 }
        let start: Int
        let end: Int
        if let first = moving.first, let last = moving.last {
            start = max(0, first - 8)
            end = min(landmarkFrames, last + 9)
        } else if let first = present.first, let last = present.last {
            start = max(0, first - 2)
            end = min(landmarkFrames, last + 3)
        } else {
            start = 0
            end = landmarkFrames
        }
        guard end - start >= 2 else {
            throw BenchmarkError.model("Activity-aligned Stage-2 range is too short")
        }

        let alignedLandmarks = try Self.resample(
            input.landmarks, sourceFrames: landmarkFrames, frameStride: 61 * 5,
            range: start..<end, targetFrames: 32, shape: [1, 8, 32, 61, 5]
        )
        let handStart = start / 2
        let handEnd = min(input.windows * 16, max(handStart + 2, (end + 1) / 2))
        let alignedEmbeddings = try Self.resample(
            embeddings, sourceFrames: input.windows * 16, frameStride: 3 * 512,
            range: handStart..<handEnd, targetFrames: 16, shape: [1, 8, 16, 3, 512]
        )
        let alignedValid = try Self.resample(
            input.handValid, sourceFrames: input.windows * 16, frameStride: 3,
            range: handStart..<handEnd, targetFrames: 16, shape: [1, 8, 16, 3],
            nearest: true
        )
        let alignedBoxes = try Self.resample(
            input.handBoxes, sourceFrames: input.windows * 16, frameStride: 3 * 4,
            range: handStart..<handEnd, targetFrames: 16, shape: [1, 8, 16, 3, 4]
        )
        let mask = try MLMultiArray(
            shape: [1, 8].map(NSNumber.init(value:)), dataType: .float32
        )
        for index in 0..<mask.count { mask[index] = index == 0 ? 1 : 0 }
        return Stage2ActivityAlignedInput(
            landmarks: alignedLandmarks,
            handEmbeddings: alignedEmbeddings,
            handValid: alignedValid,
            handBoxes: alignedBoxes,
            windowMask: mask,
            sourceFrameRange: start..<end,
            sourceWindow: min(input.windows - 1, (start + end) / 2 / 32)
        )
    }

    private static func resample(
        _ source: MLMultiArray,
        sourceFrames: Int,
        frameStride: Int,
        range: Range<Int>,
        targetFrames: Int,
        shape: [Int],
        nearest: Bool = false
    ) throws -> MLMultiArray {
        guard source.dataType == .float32,
              source.count >= sourceFrames * frameStride,
              range.lowerBound >= 0,
              range.upperBound <= sourceFrames,
              range.count >= 2,
              targetFrames >= 2,
              shape.reduce(1, *) >= targetFrames * frameStride else {
            throw BenchmarkError.model("Cannot resample malformed Stage-2 activity input")
        }
        let output = try MLMultiArray(
            shape: shape.map(NSNumber.init(value:)),
            dataType: .float32
        )
        for index in 0..<output.count { output[index] = 0 }
        for target in 0..<targetFrames {
            let position = Double(target) * Double(range.count - 1)
                / Double(targetFrames - 1) + Double(range.lowerBound)
            let left = Int(floor(position))
            let right = min(range.upperBound - 1, left + 1)
            let fraction = nearest ? 0 : position - Double(left)
            let selected = Int(position.rounded())
            for value in 0..<frameStride {
                let result: Double
                if nearest {
                    result = source[selected * frameStride + value].doubleValue
                } else {
                    let a = source[left * frameStride + value].doubleValue
                    let b = source[right * frameStride + value].doubleValue
                    result = a + (b - a) * fraction
                }
                output[target * frameStride + value] = NSNumber(value: result)
            }
        }
        return output
    }

    private static func removeAdjacentDuplicates(_ values: [Int]) -> [Int] {
        values.reduce(into: []) { output, value in
            if output.last != value { output.append(value) }
        }
    }

    private static func window(
        _ source: MLMultiArray, at window: Int, shape: [Int]
    ) throws -> MLMultiArray {
        let output = try MLMultiArray(
            shape: shape.map(NSNumber.init(value:)), dataType: .float32
        )
        let stride = shape.reduce(1, *)
        guard source.dataType == .float32,
              window >= 0,
              (window + 1) * stride <= source.count else {
            throw BenchmarkError.model("Cannot select the activity-aligned model window")
        }
        let start = window * stride
        for index in 0..<stride { output[index] = source[start + index] }
        return output
    }

    private static func collapse(
        logits: MLMultiArray, windowCount: Int, tokensPerWindow: Int, blankIndex: Int
    ) -> [Int] {
        var output: [Int] = []
        var previous = -1
        for time in 0..<(windowCount * tokensPerWindow) {
            let start = time * 101
            var best = 0
            for token in 1..<101 where logits[start + token].doubleValue > logits[start + best].doubleValue {
                best = token
            }
            if best != blankIndex, best != previous { output.append(best) }
            previous = best
        }
        return output
    }

    private static func milliseconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) * 1_000
            + Double(duration.components.attoseconds) / 1.0e15
    }

    private static func directoryBytes(_ url: URL) -> UInt64 {
        guard let enumerator = FileManager.default.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey]
        ) else { return 0 }
        var total: UInt64 = 0
        for case let item as URL in enumerator {
            total += UInt64((try? item.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }
}
