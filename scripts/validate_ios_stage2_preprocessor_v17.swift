import Foundation

@main
enum ValidateIOSStage2PreprocessorV17 {
    static func main() async throws {
        guard CommandLine.arguments.count == 3 else {
            throw BenchmarkError.invalidVideo("usage: validate-ios-preprocessor VIDEO APP_BUNDLE")
        }
        let url = URL(fileURLWithPath: CommandLine.arguments[1])
        let app = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
        let result = try await V17Pipeline.prepareStage2(from: url, inputMirrored: false)
        let vocabularyURL = app.appendingPathComponent("citizen100_manifest.json")
        let vocabulary = try JSONSerialization.jsonObject(with: Data(contentsOf: vocabularyURL))
        guard let object = vocabulary as? [String: Any],
              let classes = object["classes"] as? [[String: Any]] else {
            throw BenchmarkError.model("invalid bundled vocabulary")
        }
        let labels = classes.sorted {
            ($0["class_index"] as? Int ?? -1) < ($1["class_index"] as? Int ?? -1)
        }.compactMap { $0["canonical_label"] as? String }
        let model = try Stage2MobileModelV17(resourceRoot: app)
        let prediction = try model.predict(input: result.input, labels: labels, iterations: 1)
        let stage3 = try Stage3MobileNaturalizerV17(
            stage2Manifest: model.manifest, resourceRoot: app
        ).naturalize(
            tokenIndices: prediction.tokenIndices, glosses: prediction.labels, labels: labels
        )
        let payload: [String: Any] = [
            "format": "slt_v17_ios_stage2_preprocessor_validation",
            "video": url.path,
            "windows": result.input.windows,
            "landmarkShape": result.input.landmarks.shape.map(\.intValue),
            "handValidShape": result.input.handValid.shape.map(\.intValue),
            "handBoxShape": result.input.handBoxes.shape.map(\.intValue),
            "windowMaskShape": result.input.windowMask.shape.map(\.intValue),
            "cropCount": result.input.crops.count,
            "sourceFrames": result.diagnostics.sourceFrames,
            "sampledFrames": result.diagnostics.trimmedFrames,
            "observedHandFrames": result.diagnostics.observedHandFrames,
            "coarseRotationClockwise": result.diagnostics.visionCoarseRotationClockwise,
            "extractionMilliseconds": result.diagnostics.extractionMilliseconds,
            "predictedTokenIndices": prediction.tokenIndices,
            "predictedGlosses": prediction.labels,
            "stage3NaturalEnglish": stage3.naturalEnglish,
            "stage3LiteralEnglish": stage3.literalEnglish,
            "stage3RenderingMode": stage3.renderingMode,
            "stage3SafeFallbackUsed": stage3.safeFallbackUsed,
            "testSplitAccessed": false,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]
        )
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data("\n".utf8))
    }
}
