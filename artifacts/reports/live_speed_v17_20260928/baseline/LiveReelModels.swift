// Core ML wrappers for the live segmental Reel: boundary students, hand-crop encoder, span recognizer.

import CoreML
import CoreVideo
import Foundation

enum LiveModelLocator {
    /// Directory holding .mlpackage sources (macOS harness); nil means the app bundle's .mlmodelc.
    nonisolated(unsafe) static var packageDirectory: URL?

    static func load(_ name: String, units: MLComputeUnits) throws -> MLModel {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units
        if let directory = packageDirectory {
            let package = directory.appendingPathComponent(name + ".mlpackage")
            let compiled = try MLModel.compileModel(at: package)
            return try MLModel(contentsOf: compiled, configuration: configuration)
        }
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            throw LiveReelError.model("\(name).mlmodelc is not bundled")
        }
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    static func resource(_ name: String, _ ext: String) throws -> URL {
        if let directory = packageDirectory {
            return directory.appendingPathComponent(name + "." + ext)
        }
        guard let url = Bundle.main.url(forResource: name, withExtension: ext) else {
            throw LiveReelError.model("\(name).\(ext) is not bundled")
        }
        return url
    }
}

extension MLMultiArray {
    /// Contiguous float values of an output (float16 or float32), in row-major order.
    func floats() -> [Float] {
        // Outputs may carry padded strides (GPU / Neural Engine); map each logical index.
        let dims = shape.map(\.intValue), steps = strides.map(\.intValue)
        var offsets = [Int](repeating: 0, count: count)
        var logical = 0
        func walk(_ axis: Int, _ base: Int) {
            if axis == dims.count { offsets[logical] = base; logical += 1; return }
            for i in 0..<dims[axis] { walk(axis + 1, base + i * steps[axis]) }
        }
        walk(0, 0)
        var out = [Float](repeating: 0, count: count)
        switch dataType {
        case .float16:
            let p = dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<count { out[i] = Float(p[offsets[i]]) }
        case .float32:
            let p = dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<count { out[i] = p[offsets[i]] }
        case .double:
            let p = dataPointer.assumingMemoryBound(to: Double.self)
            for i in 0..<count { out[i] = Float(p[offsets[i]]) }
        default:
            for i in 0..<count { out[i] = self[i].floatValue }
        }
        return out
    }

    func int32s() -> [Int32] {
        let p = dataPointer.assumingMemoryBound(to: Int32.self)
        return (0..<count).map { p[$0] }
    }

    func fill(_ values: [Float], at offset: Int = 0) {
        withUnsafeMutableBufferPointer(ofType: Float.self) { p, _ in
            values.withUnsafeBufferPointer { v in
                p.baseAddress!.advanced(by: offset).update(from: v.baseAddress!, count: v.count)
            }
        }
    }
}

/// AV boundary student: features [1,64,450] + valid [1,64] -> O/B/I logits at the read position.
final class LiveBoundaryModel {
    let lookahead: Int
    private let model: MLModel
    private let output: String
    private let features: MLMultiArray
    private let valid: MLMultiArray

    init(name: String, lookahead: Int = 6, units: MLComputeUnits = .all) throws {
        model = try LiveModelLocator.load(name, units: units)
        output = model.modelDescription.outputDescriptionsByName.keys.sorted().first!
        self.lookahead = lookahead
        features = try MLMultiArray(shape: [1, 64, 450], dataType: .float32)
        valid = try MLMultiArray(shape: [1, 64], dataType: .float32)
    }

    /// `rows` are up to 64 feature rows placed right-aligned; `mask` marks real rows (after padding).
    func logits(window: [[Float]], mask: [Bool]) throws -> [Float] {
        precondition(window.count == 64 && mask.count == 64)
        features.withUnsafeMutableBufferPointer(ofType: Float.self) { p, _ in
            for (i, row) in window.enumerated() {
                row.withUnsafeBufferPointer { r in
                    p.baseAddress!.advanced(by: i * 450).update(from: r.baseAddress!, count: 450)
                }
            }
        }
        valid.withUnsafeMutableBufferPointer(ofType: Float.self) { p, _ in
            for i in 0..<64 { p[i] = mask[i] ? 1 : 0 }
        }
        let result = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "features": MLFeatureValue(multiArray: features), "valid": MLFeatureValue(multiArray: valid),
        ]))
        return result.featureValue(for: output)!.multiArrayValue!.floats()
    }
}

/// AVBoundaryStream: bounded raw history -> boundary_features -> BIO for the frame `lookahead` back.
final class LiveBoundaryStream {
    let model: LiveBoundaryModel
    private var raws: [[Float]] = []
    private var times: [Double] = []
    private var rows: [[Float]] = []     // last 64 boundary feature rows
    private var last = -Double.infinity
    static let frames = 64

    init(model: LiveBoundaryModel) { self.model = model }

    func reset() { raws = []; times = []; rows = []; last = -.infinity }

    private static func logSoftmaxWithUnknown(_ logits: [Float]) -> [Float] {
        let m = logits.max()!
        let lse = m + log(logits.map { exp($0 - m) }.reduce(0, +))
        return [log(Float(1e-6))] + logits.map { $0 - lse }
    }

    /// Log-probs (UNK, O, B, I) for the frame `lookahead` steps back, or nil while warming up.
    func update(raw: [Float], seconds: Double) throws -> [Float]? {
        if seconds - last > 0.26 { reset() }
        last = seconds
        raws.append(raw); times.append(seconds)
        while times.count > 2 && times[1] < seconds - 1.2 { raws.removeFirst(); times.removeFirst() }
        rows.append(LiveFeatures.boundaryRow(raws: raws, times: times))
        if rows.count > Self.frames { rows.removeFirst(rows.count - Self.frames) }
        guard rows.count > model.lookahead else { return nil }
        let pad = Self.frames - rows.count
        let zero = [Float](repeating: 0, count: LiveFeatures.boundaryDimension)
        let window = [[Float]](repeating: zero, count: max(pad, 0)) + rows
        let mask = (0..<Self.frames).map { $0 >= max(pad, 0) }
        return Self.logSoftmaxWithUnknown(try model.logits(window: window, mask: mask))
    }

    /// Estimates for the last `lookahead` frames with the unseen future masked (clip-end windows).
    func flush() throws -> [[Float]] {
        guard !rows.isEmpty else { return [] }
        let n = rows.count, look = model.lookahead, read = Self.frames - 1 - look
        let zero = [Float](repeating: 0, count: LiveFeatures.boundaryDimension)
        var out: [[Float]] = []
        for k in stride(from: look, through: 1, by: -1) {
            let target = n - k
            guard target >= 0 else { continue }
            var window: [[Float]] = [], mask: [Bool] = []
            for i in 0..<Self.frames {
                let index = i + target - read
                let ok = index >= 0 && index < n
                window.append(ok ? rows[index] : zero)
                mask.append(ok)
            }
            out.append(Self.logSoftmaxWithUnknown(try model.logits(window: window, mask: mask)))
        }
        return out
    }
}

/// MobileCLIP2-S0 image encoder: 256x256 RGB crop -> 512-d embedding.
final class LiveHandEncoder {
    /// FP32 matches the desktop default exactly; FP16 is faster on the Neural Engine (tune -2/226).
    nonisolated(unsafe) static var packageName = "MobileCLIP2S0ImageEncoderV17FP32"
    private let model: MLModel
    private var buffer: CVPixelBuffer?

    init(name: String = LiveHandEncoder.packageName, units: MLComputeUnits = .all) throws {
        model = try LiveModelLocator.load(name, units: units)
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, 256, 256, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pixelBuffer)
        guard let pixelBuffer else { throw LiveReelError.model("Could not allocate the crop buffer") }
        buffer = pixelBuffer
    }

    func embed(rgb: [UInt8]) throws -> [Float] {
        guard let buffer else { throw LiveReelError.model("No crop buffer") }
        CVPixelBufferLockBaseAddress(buffer, [])
        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<256 {
            for x in 0..<256 {
                let s = (y * 256 + x) * 3, d = y * stride + x * 4
                base[d] = rgb[s + 2]; base[d + 1] = rgb[s + 1]; base[d + 2] = rgb[s]; base[d + 3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        let result = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: buffer),
        ]))
        return result.featureValue(for: "embedding")!.multiArrayValue!.floats()
    }

    /// SpanRecognizer.frame_hand: embeddings, valid and boxes for one frame.
    func frameHand(_ o: LiveObservation, image: LiveBGRAImage) throws -> LiveHandFrame {
        let crops = LiveHandCrops.crops(o, image: image)
        var frame = LiveHandFrame()
        frame.valid = crops.valid
        frame.boxes = crops.boxes
        for view in 0..<3 {
            guard let crop = crops.crops[view] else { continue }
            frame.embeddings.replaceSubrange(view * 512 ..< view * 512 + 512, with: try embed(rgb: crop))
        }
        return frame
    }
}

/// One span's recognizer inputs (float32, row-major).
struct LiveSpanInput {
    let landmarks: [Float]      // 32*61*5
    let embeddings: [Float]     // 16*3*512
    let valid: [Float]          // 16*3
    let boxes: [Float]          // 16*3*4
}

/// Unified span recognizer with the letter head, fixed batch 8: word logits [8,100] + letters [8,27].
final class LiveSpanRecognizer {
    static let batch = 8
    let labels: [String]
    let letterClasses: [String]
    let letterThreshold: Float
    private let model: MLModel
    private let wordOutput: String
    private let letterOutput: String
    private let landmarks, embeddings, valid, boxes: MLMultiArray

    init(name: String = "SpanRecognizerV17LocalALettersB8FP16", labels: [String], letterThreshold: Float = 0.5,
         units: MLComputeUnits = .cpuAndGPU) throws {
        model = try LiveModelLocator.load(name, units: units)
        let outputs = model.modelDescription.outputDescriptionsByName
        guard outputs["word_logits"] != nil, let letters = outputs.keys.first(where: { $0 != "word_logits" }) else {
            throw LiveReelError.model("Span recognizer needs word and letter outputs")
        }
        wordOutput = "word_logits"; letterOutput = letters
        self.labels = labels
        letterClasses = (0..<26).map { "FS_" + String(UnicodeScalar(65 + $0)!) } + ["NONE"]
        self.letterThreshold = letterThreshold
        landmarks = try MLMultiArray(shape: [8, 32, 61, 5], dataType: .float32)
        embeddings = try MLMultiArray(shape: [8, 16, 3, 512], dataType: .float32)
        valid = try MLMultiArray(shape: [8, 16, 3], dataType: .float32)
        boxes = try MLMultiArray(shape: [8, 16, 3, 4], dataType: .float32)
    }

    /// SpanRecognizer.inputs: span landmarks plus hand evidence at 16 positions of the trimmed span.
    static func inputs(_ observations: [LiveObservation], _ hands: [LiveHandFrame]) -> LiveSpanInput? {
        guard let built = LiveSpanLandmarks.build(observations) else { return nil }
        let length = built.trimEnd - built.trimStart
        var e = [Float](), v = [Float](), b = [Float]()
        e.reserveCapacity(16 * 1536)
        for k in 0..<16 {
            let position = Int((Double(k) * Double(length - 1) / 15).rounded(.toNearestOrEven)) + built.trimStart
            let hand = hands[position]
            e += hand.embeddings
            v += hand.valid.map { $0 > 0.5 ? 1 : 0 }
            b += hand.boxes
        }
        return LiveSpanInput(landmarks: built.features, embeddings: e, valid: v, boxes: b)
    }

    /// [N][127]: 100 word logits then 27 letter logits, in batches of 8 (short batches padded).
    func logits(_ batch: [LiveSpanInput]) throws -> [[Float]] {
        var out: [[Float]] = []
        var i = 0
        while i < batch.count {
            let chunk = Array(batch[i..<min(i + Self.batch, batch.count)])
            for slot in 0..<Self.batch {
                let x = chunk[min(slot, chunk.count - 1)]
                landmarks.fill(x.landmarks, at: slot * 32 * 61 * 5)
                embeddings.fill(x.embeddings, at: slot * 16 * 3 * 512)
                valid.fill(x.valid, at: slot * 16 * 3)
                boxes.fill(x.boxes, at: slot * 16 * 3 * 4)
            }
            let result = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "landmarks": MLFeatureValue(multiArray: landmarks),
                "hand_embeddings": MLFeatureValue(multiArray: embeddings),
                "hand_valid": MLFeatureValue(multiArray: valid),
                "hand_boxes": MLFeatureValue(multiArray: boxes),
            ]))
            let words = result.featureValue(for: wordOutput)!.multiArrayValue!.floats()
            let letters = result.featureValue(for: letterOutput)!.multiArrayValue!.floats()
            for slot in 0..<chunk.count {
                out.append(Array(words[slot * 100 ..< slot * 100 + 100]) + Array(letters[slot * 27 ..< slot * 27 + 27]))
            }
            i += chunk.count
        }
        return out
    }
}
