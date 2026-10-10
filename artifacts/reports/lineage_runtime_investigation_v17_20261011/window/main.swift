import Foundation
import CoreML
func readJSON(_ path: String) throws -> Any { try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) }
func writeJSON(_ value: Any, _ path: String) throws { try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]).write(to: URL(fileURLWithPath:path)) }

func runTraceWords(_ path: String, _ out: String) throws {
    let trace = try readJSON(path) as! [String: Any]
    let frames = (trace["frames"] as! [[String: Any]]).filter { $0["start"] == nil && ($0["held"] as? Bool) != true }
    func points(_ row: [String: Any], _ key: String) -> [LivePoint] {
        (row[key] as! [[Double]]).map { LivePoint(x: Float($0[0]), y: Float($0[1]), c: Float($0[2])) }
    }
    func hand(_ row: [String: Any], _ key: String) -> LiveHand? {
        guard row[key] != nil else { return nil }
        var h = LiveHand(); h.points = points(row, key)
        h.chirality = row[key + "c"] as? String ?? "unknown"
        h.score = Float(row[key + "s"] as? Double ?? 0)
        return h
    }
    let observations = frames.map { r in
        LiveObservation(seconds: r["s"] as! Double, width: r["w"] as! Int, height: r["h"] as! Int,
                        left: hand(r, "l"), right: hand(r, "r"), body: points(r, "b"), face: points(r, "f"),
                        faceForFeatures: r["ff"] as! Bool)
    }
    let hands = frames.map { r -> LiveHandFrame in
        var h = LiveHandFrame()
        let data = Data(base64Encoded: r["e"] as! String)!
        h.embeddings = data.withUnsafeBytes { $0.bindMemory(to: Float16.self).map(Float.init) }
        h.valid = (r["v"] as! [Double]).map(Float.init)
        h.boxes = (r["x"] as! [Double]).map(Float.init)
        return h
    }
    var rows: [[String: Any]] = []
    var arrays = [[Float]](repeating: [], count: 4)
    for st in stride(from:44,through:60,by:2) {
      for en in stride(from:max(st+3,55),through:70,by:2) where en<observations.count {
        let ids=observations.indices.filter { observations[$0].seconds>=observations[st].seconds-0.1-1e-9 && observations[$0].seconds<=observations[en].seconds+0.1+1e-9 }
        if let x=LiveSpanRecognizer.inputs(ids.map{observations[$0]},ids.map{hands[$0]}) {
          rows.append(["start":st,"end":en]); arrays[0] += x.landmarks; arrays[1] += x.embeddings; arrays[2] += x.valid; arrays[3] += x.boxes
        }
      }
    }
    for (i,a) in arrays.enumerated() { try a.withUnsafeBufferPointer { Data(buffer:$0) }.write(to:URL(fileURLWithPath:out+".\(i).bin")) }
    try writeJSON(rows,out+".json")
}
let a=CommandLine.arguments
try runTraceWords(a[1],a[2])
