// Replays parity_cases.json (desktop Python captions) through LiveCaptions.swift.
// usage: parity <live_captions_v17.json> <live_captions_v17.bin> <parity_cases.json>
import Foundation

let args = CommandLine.arguments
let started = Date()
let models = try LiveCaptionModels(spec: URL(fileURLWithPath: args[1]), weights: URL(fileURLWithPath: args[2]))
print(String(format: "loaded in %.0f ms", 1000 * Date().timeIntervalSince(started)))
let cases = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[3]))) as! [[String: Any]]
var events = 0, mismatched = 0, badCases: [String] = []
var pushTimes: [Double] = []
for c in cases {
    let captions = LiveCaptions(models)
    var ok = true
    for e in c["events"] as! [[String: Any]] {
        let expected = e["text"] as? String
        let got: String?
        switch e["op"] as! String {
        case "push":
            let t0 = Date()
            got = captions.push(e["gloss"] as! String, score: e["score"] as! Double, seconds: e["seconds"] as! Double)
            pushTimes.append(Date().timeIntervalSince(t0))
        case "tick": got = captions.tick(e["seconds"] as! Double)
        default: got = captions.finish()
        }
        events += 1
        if got != expected {
            mismatched += 1
            if ok { print("MISMATCH \(c["id"]!) \(e["op"]!) @\(e["seconds"] ?? "end")\n  python: \(expected ?? "nil")\n  swift:  \(got ?? "nil")") }
            ok = false
        }
    }
    if !ok { badCases.append(c["id"] as! String) }
}
pushTimes.sort()
print("cases \(cases.count), events \(events), mismatched events \(mismatched), mismatched cases \(badCases.count)")
print(String(format: "push median %.2f ms, p95 %.2f ms, max %.2f ms (Mac CPU)",
             1000 * pushTimes[pushTimes.count / 2], 1000 * pushTimes[pushTimes.count * 95 / 100], 1000 * pushTimes.last!))
exit(badCases.isEmpty ? 0 : 1)
