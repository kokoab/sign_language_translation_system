import Foundation
import CoreML

let args = CommandLine.arguments
LiveModelLocator.packageDirectory = URL(fileURLWithPath: args[1])
let stage3 = try LiveStage3(units: .cpuAndGPU)
let cases = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[2]))) as! [[String: Any]]
var results: [[String: Any]] = []
for row in cases {
    let glosses = row["glosses"] as! [String]
    let scores = row["confidences"] as! [Double]
    let reference = row["candidate"] as! String
    let rendered = stage3.rephrase(glosses, scores)
    // Artificial long pauses verify that this model receives one complete utterance.
    let words = zip(glosses, scores).enumerated().map { i, pair in
        LiveWord(gloss: pair.0, startFrame: i * 60, endFrame: i * 60 + 20,
                 commitFrame: i * 60 + 25, startSeconds: Double(i) * 3,
                 endSeconds: Double(i) * 3 + 1, commitSeconds: Double(i) * 3 + 1.25,
                 score: pair.1, early: false)
    }
    let utterance = stage3.renderUtterance(words)
    results.append(["glosses": glosses, "sentence": rendered.sentence, "mode": rendered.mode,
                    "pytorch": reference, "parity": rendered.sentence == reference,
                    "whole_utterance_clauses": utterance.clauses,
                    "whole_utterance_sentence": utterance.sentence])
    guard rendered.mode == "t5_efficient_tiny", rendered.sentence == reference,
          utterance.clauses == [glosses], utterance.sentence == rendered.sentence else {
        throw LiveReelError.model("Native parity or neural-only contract failed: \(glosses)")
    }
}
let long = stage3.rephrase(Array(repeating: "HELLO", count: 70), Array(repeating: 0.9, count: 70))
guard long.mode == "literal_fallback", long.sentence.lowercased().components(separatedBy: "hello").count - 1 == 70 else {
    throw LiveReelError.model("Long-input fallback discarded words")
}
let out: [String: Any] = ["rows": results, "count": results.count, "overlong_input_preserved": true]
try JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: args[3]))
print("Native Swift Core ML parity and model-owned boundaries passed for \(results.count) cases")
