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
    let contextHeadPackageTreeSHA256: String
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
        return Stage2PreparedInput(
            landmarks: landmarks,
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
    private let contextHead: MLModel
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
              manifest.version == 1,
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
            "Stage2CompactContextV17FP32",
        ]
        let urls = try resources.map { name -> URL in
            guard let url = resource(name, "mlmodelc") else {
                throw BenchmarkError.model("\(name).mlmodelc is not bundled")
            }
            return url
        }
        imageEncoder = try MLModel(contentsOf: urls[0], configuration: configuration)
        frozenEncoder = try MLModel(contentsOf: urls[1], configuration: configuration)
        contextHead = try MLModel(contentsOf: urls[2], configuration: configuration)
        modelBytes = urls.reduce(0) { $0 + Self.directoryBytes($1) }
    }

    func predict(
        input: Stage2PreparedInput, labels: [String], iterations: Int
    ) throws -> Stage2Prediction {
        guard labels.count == 100, iterations >= 1 else {
            throw BenchmarkError.model("Stage-2 requires 100 labels and a positive iteration count")
        }
        let embeddings = try MLMultiArray(
            shape: [1, 8, 16, 3, 512].map(NSNumber.init(value:)), dataType: .float32
        )
        for index in 0..<embeddings.count { embeddings[index] = 0 }
        let cropStarted = ContinuousClock.now
        for crop in input.crops {
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
        var output = try contextHead.prediction(from: headProvider)
        let cold = Self.milliseconds(started.duration(to: clock.now))
        for _ in 0..<5 {
            frozen = try frozenEncoder.prediction(from: encoderProvider)
            features = frozen.featureValue(for: "var_2922")!.multiArrayValue!
            headProvider = try MLDictionaryFeatureProvider(dictionary: [
                "frozen_features": features, "window_mask": input.windowMask,
            ])
            output = try contextHead.prediction(from: headProvider)
        }
        var timings: [Double] = []
        timings.reserveCapacity(iterations)
        var logitSums = Array(repeating: 0.0, count: 64 * 101)
        var sequenceVoteCounts: [String: Int] = [:]
        for _ in 0..<iterations {
            started = clock.now
            frozen = try frozenEncoder.prediction(from: encoderProvider)
            features = frozen.featureValue(for: "var_2922")!.multiArrayValue!
            headProvider = try MLDictionaryFeatureProvider(dictionary: [
                "frozen_features": features, "window_mask": input.windowMask,
            ])
            output = try contextHead.prediction(from: headProvider)
            timings.append(Self.milliseconds(started.duration(to: clock.now)))
            guard let timedLogits = output.featureValue(for: "var_409")?.multiArrayValue,
                  timedLogits.count == logitSums.count else {
                throw BenchmarkError.model("Stage-2 context head output is invalid")
            }
            for index in 0..<logitSums.count {
                logitSums[index] += timedLogits[index].doubleValue
            }
            let vote = Self.collapse(
                logits: timedLogits,
                windowCount: input.windows,
                tokensPerWindow: manifest.tokensPerWindow,
                blankIndex: manifest.blankIndex
            ).map(String.init).joined(separator: ",")
            sequenceVoteCounts[vote, default: 0] += 1
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
            predictionAggregation: "mean_logits_across_timed_iterations_then_greedy_ctc",
            sequenceVoteCounts: sequenceVoteCounts
        )
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
