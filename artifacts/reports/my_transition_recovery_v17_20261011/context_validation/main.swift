import Foundation
import CoreML
func readJSON(_ path: String) throws -> Any { try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) }
func writeJSON(_ value: Any, _ path: String) throws { try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]).write(to: URL(fileURLWithPath:path)) }

var sharedEngine: LiveReelEngine?
var sharedRuntime: LiveSegmentalRuntime?

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
        if let encoded = r["ef32"] as? String {
            h.embeddings = Data(base64Encoded: encoded)!.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        } else {
            let data = Data(base64Encoded: r["e"] as! String)!
            h.embeddings = data.withUnsafeBytes { $0.bindMemory(to: Float16.self).map(Float.init) }
        }
        h.valid = (r["v"] as! [Double]).map(Float.init)
        h.boxes = (r["x"] as! [Double]).map(Float.init)
        return h
    }
    if sharedEngine == nil { sharedEngine = try LiveReelEngine(recognizerUnits: .all) }
    let engine = sharedEngine!
    let base = engine.wordsGated
    if sharedRuntime == nil { sharedRuntime = LiveSegmentalRuntime(boundary: base.boundary.model, recognizer: base.recognizer, encoder: base.encoder, config: .v3Words, mode: .both, letterGate: base.letterGate) }
    let runtime = sharedRuntime!; runtime.reset(); LiveMyLikeMotion.events=[]; LiveMyLikeMotion.logged=[]
    var words: [LiveWord] = []
    for (o,h) in zip(observations,hands) {
        words += try autoreleasepool { try runtime.observe(o, image: nil, hand: h) }
    }
    words += try runtime.finish()
    let visible = words.filter { !$0.gloss.hasPrefix("FS_") }

    try writeJSON(["motion_events": LiveMyLikeMotion.events,"frames": frames.count, "words": visible.map(\.json), "hypothesis": visible.map(\.gloss)], out)
}


let args=CommandLine.arguments
LiveModelLocator.packageDirectory=URL(fileURLWithPath:args[1])
let cases = try readJSON(args[2]) as! [[String:Any]]
var records: [[String:Any]] = []
let temporary = args[3] + ".current.json"
for (index,item) in cases.enumerated() {
    Experiment.maxDuration=item["max_duration"] as? Double ?? 0.45
    Experiment.floor=item["floor"] as? Double ?? 0.6
    Experiment.rank=item["rank"] as? Int ?? 100
    Experiment.flat=item["flat"] as? Bool ?? false
    Experiment.policy=item["policy"] as? String ?? "pair"
    Experiment.events=[]
    Experiment.pairFloor=item["pair_floor"] as? Double ?? 0.0
        Experiment.agreementFloor=item["agreement_floor"] as? Double ?? 2.0
        Experiment.onsetFloor=Float(item["onset_floor"] as? Double ?? 0.4)
        Experiment.touchLimit=item["touch_limit"] as? Double ?? 0.06
        Experiment.maxContact=item["max_contact"] as? Double ?? 0.35
        Experiment.successorFloor=item["successor_floor"] as? Double ?? 0.85
        Experiment.successorGap=item["successor_gap"] as? Double ?? 0.6
        Experiment.strong=item["strong"] as? Double ?? 0.6
    Experiment.support=item["support"] as? Double ?? 0.5
    Experiment.nextGap=item["next_gap"] as? Double ?? 0.25
    let result = try autoreleasepool { () throws -> [String:Any] in
        try runTraceWords(item["path"] as! String, temporary)
        var r = try readJSON(temporary) as! [String:Any]
        r["score_events"]=Experiment.events
        r["id"]=item["id"];r["dataset"]=item["dataset"];r["reference"]=item["reference"]
        return r
    }
    records.append(result)
    if index % 25 == 0 { print(index, item["dataset"]!, item["id"]!);fflush(stdout); try writeJSON(["records":records],args[3]) }
}
try writeJSON(["records":records],args[3])
print("COMPLETE",records.count)
