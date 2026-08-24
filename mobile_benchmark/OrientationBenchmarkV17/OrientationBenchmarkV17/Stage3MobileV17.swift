import CryptoKit
import Foundation

private struct Stage3ReviewedTemplateV17: Codable {
    let glosses: [String]
    let english: String
}

private struct Stage3MobileManifestV17: Codable {
    let format: String
    let version: Int
    let scope: String
    let stage2ContractSha256: String
    let vocabularyManifestSha256: String
    let recognizerCheckpointSha256: String
    let emptyOutput: String
    let literalLexicon: [String: String]
    let reviewedTemplates: [Stage3ReviewedTemplateV17]
}

struct Stage3MobileTextV17 {
    let tokenIndices: [Int]
    let glosses: [String]
    let literalEnglish: String
    let naturalEnglish: String
    let renderingMode: String
    let safeFallbackUsed: Bool
    let naturalizerManifestSHA256: String
}

final class Stage3MobileNaturalizerV17 {
    private static let expectedStage2ContractSHA256 =
        "8be66a44d337dd99484d3ee3140f3124c2e121abe20e93ce7f09b94d96ecc30d"

    private let manifest: Stage3MobileManifestV17
    private let manifestSHA256: String
    private let templates: [String: String]

    init(stage2Manifest: Stage2MobileManifest, resourceRoot: URL? = nil) throws {
        let url = resourceRoot?.appendingPathComponent(
            "stage3_mobile_naturalizer_manifest_v17.json"
        ) ?? Bundle.main.url(
            forResource: "stage3_mobile_naturalizer_manifest_v17", withExtension: "json"
        )
        guard let url, FileManager.default.fileExists(atPath: url.path) else {
            throw BenchmarkError.model("Stage-3 mobile naturalizer manifest is not bundled")
        }
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        manifest = try decoder.decode(Stage3MobileManifestV17.self, from: data)
        manifestSHA256 = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard manifest.format == "slt_stage3_mobile_naturalizer_manifest_v17",
              manifest.version == 1,
              manifest.scope == "bounded_meaning_conservative_locked_100",
              manifest.stage2ContractSha256 == Self.expectedStage2ContractSHA256,
              manifest.vocabularyManifestSha256 == stage2Manifest.vocabularyManifestSHA256,
              manifest.recognizerCheckpointSha256 == stage2Manifest.checkpointSHA256 else {
            throw BenchmarkError.model("Stage-3 manifest does not match the frozen Stage-2 contract")
        }
        var values: [String: String] = [:]
        for template in manifest.reviewedTemplates {
            guard !template.glosses.isEmpty,
                  template.glosses.allSatisfy({ !$0.isEmpty }),
                  !template.english.isEmpty else {
                throw BenchmarkError.model("Stage-3 manifest contains an invalid reviewed template")
            }
            let key = Self.key(template.glosses)
            guard values[key] == nil else {
                throw BenchmarkError.model("Stage-3 manifest contains a duplicate reviewed template")
            }
            values[key] = template.english
        }
        guard !values.isEmpty else {
            throw BenchmarkError.model("Stage-3 manifest has no reviewed templates")
        }
        templates = values
    }

    func naturalize(
        tokenIndices: [Int], glosses: [String], labels: [String]
    ) throws -> Stage3MobileTextV17 {
        guard labels.count == 100,
              tokenIndices.count == glosses.count,
              zip(tokenIndices, glosses).allSatisfy({ token, gloss in
                  1...labels.count ~= token && labels[token - 1] == gloss
              }) else {
            throw BenchmarkError.model("Stage-3 rejected an invalid Stage-2 token/gloss sequence")
        }
        let literal = literalRender(glosses)
        let natural: String
        let mode: String
        let fallback: Bool
        if glosses.isEmpty {
            natural = manifest.emptyOutput
            mode = "empty"
            fallback = true
        } else if let reviewed = templates[Self.key(glosses)] {
            natural = reviewed
            mode = "reviewed_template"
            fallback = false
        } else {
            natural = literal
            mode = "literal_fallback"
            fallback = true
        }
        return Stage3MobileTextV17(
            tokenIndices: tokenIndices,
            glosses: glosses,
            literalEnglish: literal,
            naturalEnglish: natural,
            renderingMode: mode,
            safeFallbackUsed: fallback,
            naturalizerManifestSHA256: manifestSHA256
        )
    }

    private func literalRender(_ glosses: [String]) -> String {
        guard !glosses.isEmpty else { return manifest.emptyOutput }
        let words = glosses.map { manifest.literalLexicon[$0] ?? $0.lowercased() }
        let joined = words.joined(separator: " ")
        return joined.prefix(1).uppercased() + String(joined.dropFirst()) + "."
    }

    private static func key(_ glosses: [String]) -> String {
        glosses.joined(separator: "\u{001f}")
    }
}
