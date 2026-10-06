import XCTest
import CoreML
import Foundation
final class BenchTests: XCTestCase {
  func testMatchedClassifiers() throws {
    let bundle = Bundle.main
    let manifestURL = try XCTUnwrap(bundle.url(forResource: "manifest", withExtension: "json"))
    let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any])
    let count = try XCTUnwrap(manifest["count"] as? Int)
    let targets = try XCTUnwrap(manifest["targets"] as? [Int])
    let referenceModels = try XCTUnwrap(manifest["models"] as? [String: [String: [Int]]])
    let featuresURL = try XCTUnwrap(bundle.url(forResource: "features", withExtension: "bin"))
    let bytes = try Data(contentsOf: featuresURL)
    let stride = 32 * 61 * 5 * MemoryLayout<Float>.size
    XCTAssertEqual(bytes.count, count * stride)
    XCTAssertEqual(targets.count, count)
    var inputs: [MLDictionaryFeatureProvider] = []
    for i in 0..<count {
      let array = try MLMultiArray(shape: [1,32,61,5], dataType: .float32)
      XCTAssertEqual(array.strides.map { $0.intValue }, [9760,305,5,1])
      bytes.copyBytes(to: UnsafeMutableRawBufferPointer(start: array.dataPointer, count: stride), from: (i * stride)..<((i + 1) * stride))
      inputs.append(try MLDictionaryFeatureProvider(dictionary: ["landmarks": array]))
    }
    let names = ["TransformerFP32", "TransformerFP16", "SqueezeformerFP32", "SqueezeformerFP16"]
    // Each configuration occupies each order position once; no overlapping model execution.
    let orders = [[0,1,3,2], [1,2,0,3], [2,3,1,0], [3,0,2,1]]
    func median(_ values: [Double]) -> Double {
      let v = values.sorted(); return v.count % 2 == 0 ? (v[v.count/2-1]+v[v.count/2])/2 : v[v.count/2]
    }
    for (pass, order) in orders.enumerated() {
      for index in order {
        try autoreleasepool {
          let name = names[index]
          XCTAssertFalse(ProcessInfo.processInfo.isLowPowerModeEnabled)
          let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
          XCTAssertLessThan(thermalBefore, 2, "Device is too warm for the planned benchmark")
          let config = MLModelConfiguration(); config.computeUnits = .all
          let url = try XCTUnwrap(bundle.url(forResource: name, withExtension: "mlmodelc"))
          let model = try MLModel(contentsOf: url, configuration: config)
          for i in 0..<30 { _ = try model.prediction(from: inputs[i % count]) }
          var times: [Double] = []; var predictions: [Int] = []; var top5: [[Int]] = []
          for input in inputs {
            try autoreleasepool {
              let start = DispatchTime.now().uptimeNanoseconds
              let output = try model.prediction(from: input)
              let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
              let logits = try XCTUnwrap(output.featureValue(for: "logits")?.multiArrayValue)
              XCTAssertEqual(logits.count, 100)
              let values = (0..<100).map { logits[$0].doubleValue }
              XCTAssertTrue(values.allSatisfy { $0.isFinite })
              let ranking = Array(values.indices.sorted { values[$0] > values[$1] }.prefix(5))
              times.append(elapsed); predictions.append(ranking[0]); top5.append(ranking)
            }
          }
          let refs = try XCTUnwrap(referenceModels[name])
          let original = try XCTUnwrap(refs["reference_top1"])
          let mac = try XCTUnwrap(refs["mac_top1"])
          XCTAssertEqual(original.count, count); XCTAssertEqual(mac.count, count)
          let row: [String: Any] = ["model": name, "pass": pass, "count": count,
            "median_ms": median(times), "p90_ms": times.sorted()[Int(ceil(Double(count)*0.9))-1],
            "correct": zip(predictions,targets).filter { $0.0 == $0.1 }.count,
            "top5_correct": zip(top5,targets).filter { $0.0.contains($0.1) }.count,
            "changed_from_pytorch": zip(predictions,original).filter { $0.0 != $0.1 }.count,
            "changed_from_mac": zip(predictions,mac).filter { $0.0 != $0.1 }.count,
            "thermal_before": thermalBefore, "thermal_after": ProcessInfo.processInfo.thermalState.rawValue,
            "low_power": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "samples_ms": times, "predictions": predictions, "top5": top5]
          print("CLASSIFIER_RESULT " + String(data: try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]), encoding: .utf8)!)
        }
      }
    }
  }
}
