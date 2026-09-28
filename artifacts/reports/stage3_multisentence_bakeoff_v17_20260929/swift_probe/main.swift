import Foundation
import CoreML

// Native Swift parity: LiveStage3 counted rendering and LiveIncrementalTranslator against the Python
// reference outputs (scripts/eval_stage3_multisentence_v17.py render-model, tiny2_full / tiny2_incrv).
let args = CommandLine.arguments
LiveModelLocator.packageDirectory = URL(fileURLWithPath: args[1])
let stage3 = try LiveStage3(units: .cpuAndGPU)
guard stage3.countedOutput else { throw LiveReelError.model("token table does not declare counted output") }
let cases = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[2]))) as! [[String: Any]]
var results: [[String: Any]] = []
var fullSame = 0, incrementalSame = 0, lockSame = 0
for row in cases {
    let glosses = row["glosses"] as! [String]
    let words = glosses.enumerated().map { i, gloss in
        LiveWord(gloss: gloss, startFrame: i * 20, endFrame: i * 20 + 10, commitFrame: i * 20 + 12,
                 startSeconds: Double(i), endSeconds: Double(i) + 0.5, commitSeconds: Double(i) + 0.6,
                 score: 0.9, early: false)
    }
    let full = stage3.renderUtterance(words).sentence
    let translator = LiveIncrementalTranslator()
    for w in words { translator.append([w], using: stage3) }
    let finished = translator.finish([], using: stage3)
    let meta = row["meta"] as! [String: Any]
    let a = full == row["full"] as! String, b = finished.sentence == row["incremental"] as! String
    let c = finished.locked == meta["locked"] as! Int && finished.tailWords == meta["finish_glosses"] as! Int
    fullSame += a ? 1 : 0; incrementalSame += b ? 1 : 0; lockSame += c ? 1 : 0
    results.append(["id": row["id"]!, "full": full, "incremental": finished.sentence, "locked": finished.locked,
                    "tail_words": finished.tailWords, "full_parity": a, "incremental_parity": b, "lock_parity": c])
}
let out: [String: Any] = ["rows": results, "count": results.count, "full_parity": fullSame,
                          "incremental_parity": incrementalSame, "lock_parity": lockSame]
try JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: args[3]))
print("rows \(results.count)  whole-buffer parity \(fullSame)  incremental parity \(incrementalSame)  lock/tail parity \(lockSame)")
