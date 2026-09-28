// Stage 3 on device: evidence-encoded glosses -> English with the T5-efficient-tiny slot model.
// Port of TinyStage3Naturalizer.rephrase + render_sentence / render_utterance
// (scripts/live_isolated_v17.py, scripts/live_segmental_v17.py). Composition-model metadata
// disables phrase overrides and delegates sentence boundaries to the model. Technical
// failures use literal rendering; legacy model metadata retains its previous behavior.

import CoreML
import Foundation

final class LiveStage3 {
    let checkpointIdentifier: String
    private let encoder: MLModel
    private let decoder: MLModel
    private let wordIDs: [String: [Int32]]
    private let pieces: [String]
    private let special: Set<Int>
    private let eos: Int32, start: Int32
    private let length: Int
    private let lexicon: [String: String]
    private let templates: [[String]: String]
    private let emptyOutput: String
    private let fullUtterance: Bool

    init(units: MLComputeUnits = .cpuAndGPU) throws {
        encoder = try LiveModelLocator.load("Stage3T5EncoderV17", units: units)
        decoder = try LiveModelLocator.load("Stage3T5DecoderStepV17", units: units)
        let tokens = try JSONSerialization.jsonObject(with: Data(contentsOf: LiveModelLocator.resource("stage3_tokens", "json"))) as! [String: Any]
        checkpointIdentifier = tokens["checkpoint"] as? String ?? "unversioned"
        wordIDs = (tokens["word_ids"] as! [String: [Int]]).mapValues { $0.map(Int32.init) }
        pieces = tokens["pieces"] as! [String]
        special = Set(tokens["special_ids"] as! [Int])
        eos = Int32(tokens["eos_id"] as! Int)
        start = Int32(tokens["decoder_start_id"] as! Int)
        length = tokens["max_length"] as! Int
        fullUtterance = tokens["utterance_segmentation"] as? String == "model"
        let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: LiveModelLocator.resource(
            "stage3_mobile_naturalizer_manifest_v17", "json"))) as! [String: Any]
        lexicon = manifest["literal_lexicon"] as? [String: String] ?? [:]
        emptyOutput = manifest["empty_output"] as? String ?? ""
        var templates: [[String]: String] = [:]
        for row in manifest["reviewed_templates"] as? [[String: Any]] ?? [] {
            if let glosses = row["glosses"] as? [String], let english = row["english"] as? String { templates[glosses] = english }
        }
        self.templates = (tokens["reviewed_templates_enabled"] as? Bool ?? true) ? templates : [:]
    }

    static func bucket(_ score: Double) -> String { score >= 0.60 ? "hi" : score >= 0.40 ? "mid" : "lo" }

    func literal(_ glosses: [String]) -> String {
        guard !glosses.isEmpty else { return emptyOutput }
        let text = glosses.map { lexicon[$0] ?? $0.lowercased() }.joined(separator: " ")
        return text.prefix(1).uppercased() + text.dropFirst() + "."
    }

    /// Greedy T5 generation; throws when a word has no token entry.
    func generate(_ glosses: [String], _ scores: [Double]) throws -> String {
        var ids: [Int32] = []
        for (gloss, score) in zip(glosses, scores) {
            for word in [Self.bucket(score), gloss.lowercased()] {
                guard let part = wordIDs[word] else { throw LiveReelError.input("No Stage-3 tokens for \(word)") }
                ids += part
            }
        }
        if fullUtterance && ids.count >= length {
            throw LiveReelError.input("Finished utterance exceeds model context; refuse to drop words")
        }
        ids = Array(ids.prefix(length - 1)) + [eos]
        let inputIDs = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        let mask = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        for i in 0..<length {
            inputIDs[i] = NSNumber(value: i < ids.count ? ids[i] : 0)
            mask[i] = NSNumber(value: i < ids.count ? 1 : 0)
        }
        let hidden = try encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIDs), "attention_mask": MLFeatureValue(multiArray: mask),
        ])).featureValue(for: "hidden")!
        let decoderIDs = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        let position = try MLMultiArray(shape: [1], dataType: .int32)
        for i in 0..<length { decoderIDs[i] = 0 }
        var out: [Int32] = [start]
        decoderIDs[0] = NSNumber(value: start)
        while out.count < length {
            position[0] = NSNumber(value: out.count - 1)
            let next = try decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "decoder_input_ids": MLFeatureValue(multiArray: decoderIDs), "encoder_hidden": hidden,
                "encoder_mask": MLFeatureValue(multiArray: mask), "position": MLFeatureValue(multiArray: position),
            ])).featureValue(for: "next_token")!.multiArrayValue!.int32s()[0]
            if next == eos { break }
            decoderIDs[out.count] = NSNumber(value: next)
            out.append(next)
        }
        let text = out.dropFirst().filter { !special.contains(Int($0)) }.map { pieces[Int($0)] }.joined()
            .replacingOccurrences(of: "\u{2581}", with: " ")
        return text.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" }).joined(separator: " ")
    }

    struct Rendering {
        var sentence: String
        var mode: String
    }

    /// TinyStage3Naturalizer.rephrase
    func rephrase(_ glosses: [String], _ scores: [Double]) -> Rendering {
        if let template = templates[glosses] { return Rendering(sentence: template, mode: "reviewed_template") }
        if let sentence = try? generate(glosses, scores), !sentence.isEmpty, sentence.count <= 300 {
            return Rendering(sentence: sentence, mode: "t5_efficient_tiny")
        }
        return Rendering(sentence: literal(glosses), mode: "literal_fallback")
    }

    static func spelledText(_ token: String) -> String {
        let word = token.hasPrefix("fs-") ? String(token.dropFirst(3)) : token
        return word.prefix(1) + word.dropFirst().lowercased()
    }

    /// render_sentence: spelled words go through slot tokens FS0, FS1, ... and are restored after.
    func renderSentence(_ glosses: [String], _ scores: [Double]) -> Rendering {
        var slots: [(String, String)] = []
        var inputs: [String] = []
        for g in glosses {
            if g.hasPrefix("fs-") {
                let name = "FS\(slots.count)"
                slots.append((name, Self.spelledText(g)))
                inputs.append(name)
            } else {
                inputs.append(g)
            }
        }
        var value = rephrase(inputs, scores)
        guard !slots.isEmpty else { return value }
        func regex(_ name: String) -> NSRegularExpression {
            try! NSRegularExpression(pattern: "\\b\(name)\\b", options: [.caseInsensitive])
        }
        let sentence = value.sentence
        let range = NSRange(sentence.startIndex..., in: sentence)
        if slots.allSatisfy({ regex($0.0).numberOfMatches(in: sentence, range: range) == 1 }) {
            var restored = sentence
            for (name, word) in slots {
                restored = regex(name).stringByReplacingMatches(
                    in: restored, range: NSRange(restored.startIndex..., in: restored),
                    withTemplate: NSRegularExpression.escapedTemplate(for: word))
            }
            value.sentence = restored
            value.mode += "+slots"
        } else {
            let names = Dictionary(uniqueKeysWithValues: slots)
            let literal = inputs.map { names[$0] ?? $0.lowercased() }.joined(separator: " ")
            value = Rendering(sentence: literal.prefix(1).uppercased() + literal.dropFirst() + ".", mode: "slot_literal_fallback")
        }
        return value
    }

    /// Composition models punctuate the full utterance; older models retain pause splits.
    func renderUtterance(_ words: [LiveWord]) -> (sentence: String, clauses: [[String]]) {
        let clauses = fullUtterance ? (words.isEmpty ? [] : [words]) : liveSplitClauses(words)
        let parts = clauses.map { renderSentence($0.map(\.gloss), $0.map(\.score)).sentence.trimmingCharacters(in: .whitespaces) }
        return (parts.joined(separator: " ").trimmingCharacters(in: .whitespaces), clauses.map { $0.map(\.gloss) })
    }
}
