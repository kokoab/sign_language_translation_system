// Deterministic parity and microbenchmarks for exact CPU optimizations.
import CoreML
import CryptoKit
import Foundation

@main struct SpeedChecks {
    static func hash<T>(_ values: [T]) -> String {
        values.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
    }

    static func main() throws {
        var results: [[String: Any]] = []
        for (width, height) in [(640, 480), (480, 640)] {
            let pixels = (0..<(width * height * 4)).map { UInt8(truncatingIfNeeded: ($0 &* 37) ^ ($0 >> 9)) }
            pixels.withUnsafeBufferPointer { p in
                let image = LiveBGRAImage(base: p.baseAddress!, width: width, height: height, bytesPerRow: width * 4)
                for side in [1, 17, 64, 128, 179, 240, 255, 256, 257, 400, 800] {
                    for origin: Float in [-30.5, 100.25] {
                        let box = [origin, origin + 20, origin + Float(side), origin + 20 + Float(side)]
                        let reference = LiveHandCrops.cropSquare(image, box: box)
                        let start = Date()
                        for _ in 0..<12 {
                            precondition(LiveHandCrops.cropSquare(image, box: box) == reference)
                        }
                        results.append(["id": "crop-\(width)-\(height)-\(side)-\(origin)",
                                        "sha256": hash(reference), "ms": Date().timeIntervalSince(start) * 1000 / 12])
                    }
                }
            }
        }
        var raw: [[Float]] = [], times: [Double] = []
        for frame in 0..<25 {
            var row = [Float](repeating: 0, count: 61 * 5)
            for node in 0..<61 {
                row[node * 5] = Float((frame * 7 + node * 3) % 19) / 50
                row[node * 5 + 1] = Float((frame * 11 + node * 2) % 17) / 50
                row[node * 5 + 3] = (frame + node) % 7 == 0 ? 0 : 1
                row[node * 5 + 4] = .init((frame + node) % 11) / 10
            }
            if frame % 8 != 0 { for node in 57..<59 { row[node * 5 + 3] = 0; row[node * 5 + 4] = 0 } }
            raw.append(row); times.append(Double(frame) * 0.1)
        }
        for n in [1, 2, 8, 17, 25] {
            let r = Array(raw.prefix(n)), t = Array(times.prefix(n))
            let reference = LiveFeatures.boundaryRow(raws: r, times: t)
            let start = Date()
            for _ in 0..<1000 { precondition(LiveFeatures.boundaryRow(raws: r, times: t) == reference) }
            results.append(["id": "boundary-\(n)", "sha256": hash(reference), "ms": Date().timeIntervalSince(start)])
        }
        for type: MLMultiArrayDataType in [.float32, .float16, .double] {
            let array = try MLMultiArray(shape: [1, 16, 3, 512], dataType: type)
            for i in 0..<array.count { array[i] = NSNumber(value: Float(i % 101) / 8) }
            let expected = (0..<array.count).map { array[$0].floatValue }
            precondition(array.floats() == expected)
            let start = Date()
            for _ in 0..<100 { precondition(array.floats() == expected) }
            results.append(["id": "array-\(type.rawValue)", "sha256": hash(expected), "ms": Date().timeIntervalSince(start)])
        }
        for strides in [[5, 1], [1, 2]] {
            let pointer = UnsafeMutableRawPointer.allocate(byteCount: 8 * MemoryLayout<Float>.stride, alignment: 16)
            let floats = pointer.bindMemory(to: Float.self, capacity: 8)
            for i in 0..<8 { floats[i] = Float(i) }
            let array = try MLMultiArray(dataPointer: pointer, shape: [2, 3], dataType: .float32,
                                        strides: strides.map(NSNumber.init(value:)), deallocator: { $0.deallocate() })
            let expected = (0..<2).flatMap { r in (0..<3).map { c in Float(r * strides[0] + c * strides[1]) } }
            precondition(array.floats() == expected)
            results.append(["id": "strided-\(strides)", "sha256": hash(expected)])
        }
        let data = try JSONSerialization.data(withJSONObject: ["checks": results], options: [.sortedKeys])
        try data.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
    }
}
