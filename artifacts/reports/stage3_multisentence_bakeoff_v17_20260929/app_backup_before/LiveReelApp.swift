// App-level pieces of the live Reel: one engine shared by the Live and Practice screens (built at
// launch, like the desktop shell's warm()), and the session store the History page reads.

import Foundation

enum LiveReelShared {
    /// Every engine call runs here: model loading, camera frames, flush and reset.
    static let queue = DispatchQueue(label: "slt.livereel.engine", qos: .userInitiated)
    static let languageQueue = DispatchQueue(label: "slt.livereel.language", qos: .userInitiated)

    // Written on `queue` / `languageQueue`, read on the main thread through `status`.
    nonisolated(unsafe) static var engine: LiveReelEngine?
    nonisolated(unsafe) static var stage3: LiveStage3?
    nonisolated(unsafe) private static var state = "idle"
    nonisolated(unsafe) private static var failure: String?

    /// Build and warm the models once; safe to call repeatedly.
    static func warm() {
        queue.async {
            guard engine == nil, state != "loading" else { return }
            state = "loading"
            do {
                // Phone settings (reports/phone_speed_v17_20260929): FP16 crop encoder and the recognizer
                // on the Neural Engine. iPhone 13 frame median 123 -> 28 ms (20 Hz needs < 50 ms); held-out
                // replay WER 10.22% -> 9.68% with the same settings on the Mac Neural Engine.
                LiveHandEncoder.packageName = "MobileCLIP2S0ImageEncoderV17FP16"
                let built = try LiveReelEngine(recognizerUnits: .all, encoderUnits: .all)
                try built.warm()
                engine = built
                state = "ready"
                languageQueue.async { stage3 = try? LiveStage3() }
            } catch {
                failure = error.localizedDescription
                state = "failed"
            }
        }
    }

    static var status: [String: Any] {
        ["ready": state == "ready", "state": state, "error": failure as Any]
    }
}

/// Live sessions saved as Documents/live_reel_sessions/<stamp>.json.
enum LiveReelSessions {
    static var directory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("live_reel_sessions", isDirectory: true)
    }

    static func save(_ value: [String: Any], started: Date) {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        let url = directory.appendingPathComponent(formatter.string(from: started) + ".json")
        if let data = try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted]) {
            try? data.write(to: url, options: .atomic)
        }
    }

    /// Newest first: stamp, started_utc, duration_seconds, glosses, sentences, complete, and the
    /// timeline of committed words ({seconds, gloss}; seconds restart at each Start).
    static func list() -> [[String: Any]] {
        let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        var rows: [[String: Any]] = []
        for url in files where url.pathExtension == "json" {
            guard let data = try? Data(contentsOf: url),
                  let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
            let events = value["events"] as? [[String: Any]] ?? []
            // Early files carried only events; derive what the page shows from them.
            let glosses = value["glosses"] as? [String] ?? events.compactMap {
                ($0["event"] as? String) == "word" ? ($0["word"] as? [String: Any])?["gloss"] as? String : nil
            }
            let sentences = value["sentences"] as? [String] ?? events.compactMap {
                ($0["event"] as? String) == "sentence" ? $0["sentence"] as? String : nil
            }
            let timeline: [[String: Any]] = events.compactMap {
                guard ($0["event"] as? String) == "word",
                      let gloss = ($0["word"] as? [String: Any])?["gloss"] as? String else { return nil }
                return ["seconds": $0["seconds"] as? Double ?? 0, "gloss": gloss]
            }
            let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
            let modified = (attributes?[.modificationDate] as? Date) ?? Date()
            rows.append([
                "stamp": url.deletingPathExtension().lastPathComponent,
                "started_utc": value["started_utc"] as? String ?? ISO8601DateFormatter().string(from: modified),
                "duration_seconds": value["duration_seconds"] as? Double ?? 0,
                "glosses": glosses,
                "sentences": sentences,
                "complete": value["complete"] as? Bool ?? true,
                "timeline": timeline,
            ])
        }
        return rows.sorted { ($0["started_utc"] as! String) > ($1["started_utc"] as! String) }
    }
}
