import Foundation

enum HaveDiagnostic {
    static var spans: [[String: Any]] = []
    static var steps: [[String: Any]] = []
    static var bio: [[Float]] = []
}

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
    let engine = try LiveReelEngine(recognizerUnits: .all)
    let base = engine.wordsGated
    if ProcessInfo.processInfo.environment["PROBE"] == "1" {
        let regions = [(1.1, 1.9), (3.9, 4.7), (6.8, 7.5), (11.3, 12.2), (20.6, 21.65), (28.2, 29.2)]
        let labels = base.recognizer.labels
        let have = labels.firstIndex(of: "HAVE")!, angry = labels.firstIndex(of: "ANGRY")!, tired = labels.firstIndex(of: "TIRED")!
        let pair = try LivePairCheck(crop: LiveModelLocator.resource("crop_have_angry_v17", "json"),
                                     shape: LiveModelLocator.resource("handshape_have_angry_v17", "json"))
        var rows: [[String: Any]] = []
        for (region, bounds) in regions.enumerated() {
            let inside = observations.indices.filter { observations[$0].seconds >= bounds.0 && observations[$0].seconds <= bounds.1 }
            for context in [0.0, 0.05, 0.1] {
                var inputs: [LiveSpanInput] = [], windows: [(Int, Int, [Int])] = []
                for a in inside {
                    for b in inside where b >= a + 3 && b <= a + 15 {
                        let ids = observations.indices.filter { observations[$0].seconds >= observations[a].seconds - context - 1e-9 && observations[$0].seconds <= observations[b].seconds + context + 1e-9 }
                        if let x = LiveSpanRecognizer.inputs(ids.map { observations[$0] }, ids.map { hands[$0] }) {
                            inputs.append(x); windows.append((a, b, ids))
                        }
                    }
                }
                let logits = try base.recognizer.logits(inputs)
                for ((window, x), l) in zip(zip(windows, inputs), logits) {
                    let (a, b, ids) = window
                    let raw = ids.map { LiveFeatures.raw(observations[$0]) }
                    let cp = pair.cropScore(x), hp = pair.shapeScore(raw)
                    let g = cp != nil && hp != nil ? min(cp!, hp!) : nil
                    let word = Array(l.prefix(labels.count))
                    let refined = LiveTiredRescue.apply(LivePairCheck.reshare(word, have: have, angry: angry, g: g), tired: tired, have: have, raw: raw, threshold: 1)
                    let p = LiveSegmental.logSoftmax(refined).map { exp($0) }
                    let order = p.indices.sorted { p[$0] > p[$1] }.prefix(3)
                    var row: [String: Any] = ["region": region, "s": a, "e": b, "start": observations[a].seconds,
                                             "end": observations[b].seconds, "context": context, "have": p[have]]
                    row["rawtop"] = word.indices.sorted { word[$0] > word[$1] }.prefix(3).map { ["g": labels[$0], "q": exp(LiveSegmental.logSoftmax(word)[$0])] as [String: Any] }; row["crop"] = cp ?? -1; row["shape"] = hp ?? -1
                    row["droop"] = LiveTiredRescue.droop(raw) ?? -1
                    row["top"] = order.map { ["g": labels[$0], "q": p[$0]] as [String: Any] }
                    rows.append(row)
                }
            }
            print("oracle region", region, "done"); fflush(stdout)
        }
        try writeJSON(["rows": rows], out)
        return
    }
    var results: [String: Any] = [:]
    let variantsPath = ProcessInfo.processInfo.environment["VARIANTS"]!
    let variants = try readJSON(variantsPath) as! [[String: Any]]
    for variant in variants {
        let name = variant["name"] as! String
        var config = LiveDecoderConfig.v3Words
        if variant["noearly_big"] as? Bool == true { config.earlyUnsafe.insert("BIG") }
        if variant["noearly_all"] as? Bool == true { config.earlyK = 0 }
        if let v = variant["context_start"] as? Double { config.diagnosticContextStart = v }
        if let v = variant["context_end"] as? Double { config.diagnosticContextEnd = v }
        if let v = variant["have_early_q"] as? Double {
            config.diagnosticHaveEarlyQ = v; config.earlyUnsafe.remove("HAVE")
        }
        if let v = variant["c"] as? Double { config.c = v }
        if let v = variant["have_cost"] as? Double { config.diagnosticHaveCost = v }
        config.diagnosticBoostHave = variant["boost_have"] as? Bool ?? false
        if let v = variant["w_b"] as? Double { config.wB = v }
        if let v = variant["lag"] as? Int { config.lag = v }
        if variant["no_floor"] as? Bool == true { config.innerFloor = nil }
        config.diagnosticRetainCandidates = variant["retain"] as? Bool ?? false
        config.diagnosticDenseStep = variant["dense_step"] as? Int ?? 2
        config.diagnosticBoundarySlack = variant["boundary_slack"] as? Int ?? 0
        config.diagnosticNoScoreFloor = variant["no_score_floor"] as? Bool ?? false
        config.diagnosticPairOdds = variant["pair_odds"] as? Bool ?? false
        config.diagnosticLockClosedHave = variant["lock_closed"] as? Bool ?? false
        config.diagnosticOrderedEarly = variant["ordered_early"] as? Bool ?? false
        config.diagnosticClosedHave = variant["closed_have"] as? Bool ?? false
        config.diagnosticTiredMerge = variant["tired_merge"] as? Bool ?? false
        config.diagnosticHaveBias = Float(variant["have_bias"] as? Double ?? 0)
        config.diagnosticTopHaveOnly = variant["top_have_only"] as? Bool ?? false
        let runtime = LiveSegmentalRuntime(boundary: base.boundary.model, recognizer: base.recognizer,
                                          encoder: base.encoder, config: config, mode: .both, letterGate: base.letterGate)
        runtime.rawCache = LiveSpanCache()
        HaveDiagnostic.spans = []; HaveDiagnostic.steps = []; HaveDiagnostic.bio = []
        runtime.onBIO = { row, _ in HaveDiagnostic.bio.append(row) }
        var words: [LiveWord] = []
        for (o, h) in zip(observations, hands) {
            words += try autoreleasepool { try runtime.observe(o, image: nil, hand: h) }
        }
        words += try runtime.finish()
        let visible = words.filter { !$0.gloss.hasPrefix("FS_") }
        results[name] = ["words": visible.map(\.json), "hypothesis": visible.map(\.gloss),
                         "spans": HaveDiagnostic.spans, "steps": HaveDiagnostic.steps, "bio": HaveDiagnostic.bio]
        print(name, visible.map(\.gloss)); fflush(stdout)
    }
    try writeJSON(["frames": frames.count, "results": results], out)
}
