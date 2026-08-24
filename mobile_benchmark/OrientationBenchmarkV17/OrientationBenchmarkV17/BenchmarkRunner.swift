import CoreML
import CryptoKit
import Darwin.Mach
import Foundation
import UIKit

struct BenchmarkReport: Codable, Identifiable {
    let id: UUID
    let createdUTC: String
    let executionEnvironment: String
    let deviceModel: String
    let simulatorDeviceName: String?
    let simulatorModelIdentifier: String?
    let simulatorRuntimeVersion: String?
    let hostArchitecture: String
    let iOSVersion: String
    let thermalBefore: String
    let thermalAfter: String
    let thermalsInterpretable: Bool
    let hardwarePerformanceClaim: Bool
    let inputFilename: String
    let inputRotationDegreesClockwise: Double?
    let sourceVideoSHA256: String?
    let generatedVideoSHA256: String?
    let featureSHA256: String?
    let handFeatureSHA256: String?
    let handCropBundleSHA256: String?
    let extractionSucceeded: Bool
    let extractionExecutionEnvironment: String
    let endToEndPipeline: Bool
    let simulatorVisionLimitation: String?
    let residualRollDegrees: Double?
    let expectedLabel: String
    let predictedLabel: String
    let predictedLabels: [String]
    let predictedTokenIndices: [Int]
    let correct: Bool
    let inputMirrored: Bool
    let iterations: Int
    let coldInferenceMilliseconds: Double
    let medianInferenceMilliseconds: Double
    let p90InferenceMilliseconds: Double
    let residentMemoryBeforeBytes: UInt64
    let residentMemoryAfterBytes: UInt64
    let modelBytes: UInt64
    let candidateID: String
    let checkpointSHA256: String
    let coreMLPackageTreeSHA256: String
    let imageEncoderPackageTreeSHA256: String?
    let frozenEncoderPackageTreeSHA256: String?
    let contextHeadPackageTreeSHA256: String?
    let vocabularyManifestSHA256: String?
    let cropEncodingMilliseconds: Double?
    let predictionAggregation: String
    let sequenceVoteCounts: [String: Int]
    let stage3NaturalEnglish: String
    let stage3LiteralEnglish: String
    let stage3RenderingMode: String
    let stage3SafeFallbackUsed: Bool
    let stage3NaturalizerManifestSHA256: String
    let allMobileNeuralModelsInCoreML: Bool
    let videoFileToGlossEndToEnd: Bool
    let cameraToGlossEndToEnd: Bool
    let diagnostics: V17Diagnostics
}

private struct BundledModelManifest: Codable {
    let candidateID: String
    let checkpointSHA256: String
    let coreMLPackageTreeSHA256: String
}

private struct SimulatorSuiteEntry: Codable {
    let angleDegreesClockwise: Double
    let relativeVideoPath: String
    let generatedVideoSHA256: String
    let relativeFeaturePath: String
    let featureSHA256: String
    let relativeHandCropBundlePath: String
    let handCropBundleSHA256: String
    let hostDiagnostics: V17Diagnostics
}

private struct SimulatorSuiteManifest: Codable {
    let format: String
    let version: Int
    let suiteID: String
    let expectedLabel: String
    let iterations: Int
    let sourceVideoSHA256: String
    let entries: [SimulatorSuiteEntry]
    let citizenTestAccessed: Bool
    let semlexTestAccessed: Bool
}

private struct SimulatorSuiteEntryResult: Codable {
    let angleDegreesClockwise: Double
    let success: Bool
    let reportFilename: String?
    let error: String?
}

private struct SimulatorSuiteAggregate: Codable {
    let format: String
    let version: Int
    let suiteID: String
    let startedUTC: String
    let completedUTC: String
    let executionEnvironment: String
    let simulatorDeviceName: String?
    let simulatorModelIdentifier: String?
    let simulatorRuntimeVersion: String?
    let expectedLabel: String
    let iterations: Int
    let sourceVideoSHA256: String
    let hardwarePerformanceClaim: Bool
    let thermalsInterpretable: Bool
    let featureExtractionEnvironment: String
    let endToEndPipeline: Bool
    let videoFileToGlossEndToEnd: Bool
    let cameraToGlossEndToEnd: Bool
    let allMobileNeuralModelsInCoreML: Bool
    let simulatorVisionLimitation: String
    let citizenTestAccessed: Bool
    let semlexTestAccessed: Bool
    let success: Bool
    let entries: [SimulatorSuiteEntryResult]
}

@MainActor
final class BenchmarkRunner: ObservableObject {
    @Published var labels: [String] = []
    @Published var expectedLabel = ""
    @Published var inputMirrored = false
    @Published var iterations = 200
    @Published var selectedURL: URL?
    @Published var status = "Choose a video. Any native aspect ratio or orientation is accepted."
    @Published var report: BenchmarkReport?
    @Published var reportURL: URL?
    @Published var errorMessage: String?
    @Published var isRunning = false

    func loadResources() {
        do {
            guard let url = Bundle.main.url(forResource: "citizen100_manifest", withExtension: "json") else {
                throw BenchmarkError.model("citizen100_manifest.json is not bundled")
            }
            let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
            guard let payload = object as? [String: Any],
                  let classes = payload["classes"] as? [[String: Any]] else {
                throw BenchmarkError.model("Class manifest is invalid")
            }
            labels = classes.sorted {
                ($0["class_index"] as? Int ?? -1) < ($1["class_index"] as? Int ?? -1)
            }.compactMap { $0["canonical_label"] as? String }
            guard labels.count == 100 else {
                throw BenchmarkError.model("Expected exactly 100 frozen labels")
            }
            expectedLabel = labels[0]
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func runAutomationIfRequested() async {
        let arguments = ProcessInfo.processInfo.arguments
        guard let flag = arguments.firstIndex(of: "--benchmark-suite"),
              arguments.indices.contains(flag + 1) else { return }
        let manifestName = arguments[flag + 1]
        isRunning = true
        report = nil
        reportURL = nil
        status = "Running automated simulator benchmark…"
        do {
            try await runSimulatorSuite(manifestName: manifestName)
            status = "Automated simulator benchmark complete."
        } catch {
            errorMessage = error.localizedDescription
            status = "Automated simulator benchmark failed."
        }
        isRunning = false
    }

    func run() {
        guard let selectedURL else {
            errorMessage = "Choose a video first"
            return
        }
        guard labels.contains(expectedLabel), iterations >= 20 else {
            errorMessage = "Choose a frozen label and at least 20 iterations"
            return
        }
        isRunning = true
        report = nil
        reportURL = nil
        status = "Running Apple Vision extraction…"
        let mirrored = inputMirrored
        let expected = expectedLabel
        let iterationCount = iterations
        Task {
            let access = selectedURL.startAccessingSecurityScopedResource()
            defer { if access { selectedURL.stopAccessingSecurityScopedResource() } }
            do {
                let result = try await executeBenchmark(
                    inputURL: selectedURL,
                    expected: expected,
                    mirrored: mirrored,
                    iterations: iterationCount,
                    inputRotationDegrees: nil,
                    sourceVideoSHA256: nil,
                    generatedVideoSHA256: Self.sha256File(selectedURL)
                )
                report = result
                reportURL = try save(result)
                status = "Complete. Export the JSON report after the phone has cooled."
            } catch {
                errorMessage = error.localizedDescription
                status = "Benchmark failed without changing the source video."
            }
            isRunning = false
        }
    }

    private func runSimulatorSuite(manifestName: String) async throws {
#if !targetEnvironment(simulator)
        throw BenchmarkError.model("Automated benchmark suites are restricted to iOS Simulator")
#else
        let documents = try documentsDirectory()
        let manifestURL = try safeRelativeURL(manifestName, under: documents)
        let suite = try JSONDecoder().decode(
            SimulatorSuiteManifest.self, from: Data(contentsOf: manifestURL)
        )
        guard suite.format == "slt_v17_ios_simulator_benchmark_suite",
              suite.version == 1,
              !suite.suiteID.isEmpty,
              labels.contains(suite.expectedLabel),
              suite.iterations >= 20,
              !suite.entries.isEmpty,
              suite.citizenTestAccessed == false,
              suite.semlexTestAccessed == false else {
            throw BenchmarkError.model("Simulator suite manifest contract failed")
        }
        let startedUTC = ISO8601DateFormatter().string(from: Date())
        var entryResults: [SimulatorSuiteEntryResult] = []
        for (index, entry) in suite.entries.enumerated() {
            status = "Simulator condition \(index + 1)/\(suite.entries.count): \(entry.angleDegreesClockwise)°"
            do {
                let videoURL = try safeRelativeURL(entry.relativeVideoPath, under: documents)
                let actualHash = try Self.sha256File(videoURL)
                guard actualHash == entry.generatedVideoSHA256 else {
                    throw BenchmarkError.invalidVideo(
                        "Generated-video SHA-256 mismatch at \(entry.angleDegreesClockwise)°"
                    )
                }
                let featureURL = try safeRelativeURL(entry.relativeFeaturePath, under: documents)
                let actualFeatureHash = try Self.sha256File(featureURL)
                guard actualFeatureHash == entry.featureSHA256 else {
                    throw BenchmarkError.invalidVideo(
                        "Host Apple Vision feature SHA-256 mismatch at \(entry.angleDegreesClockwise)°"
                    )
                }
                let handCropURL = try safeRelativeURL(
                    entry.relativeHandCropBundlePath, under: documents
                )
                let actualHandCropHash = try Self.sha256File(handCropURL)
                guard actualHandCropHash == entry.handCropBundleSHA256 else {
                    throw BenchmarkError.invalidVideo(
                        "Host hand-crop bundle SHA-256 mismatch at \(entry.angleDegreesClockwise)°"
                    )
                }
                let prepared = try Stage2InputBundleV17.load(
                    landmarksURL: featureURL, cropBundleURL: handCropURL
                )
                status = "Running MobileCLIP2 and sustained Stage-2 Core ML benchmark…"
                let labelValues = labels
                let iOSVersion = UIDevice.current.systemVersion
                let result = try await Task.detached(priority: .userInitiated) {
                    try Self.measureStage2(
                        prepared: prepared,
                        diagnostics: entry.hostDiagnostics,
                        inputURL: videoURL,
                        expected: suite.expectedLabel,
                        labels: labelValues,
                        mirrored: false,
                        iterations: suite.iterations,
                        iOSVersion: iOSVersion,
                        inputRotationDegrees: entry.angleDegreesClockwise,
                        sourceVideoSHA256: suite.sourceVideoSHA256,
                        generatedVideoSHA256: actualHash,
                        featureSHA256: actualFeatureHash,
                        handCropBundleSHA256: actualHandCropHash,
                        extractionExecutionEnvironment: "host_macos_apple_vision",
                        simulatorVisionLimitation: Self.simulatorVisionLimitation
                    )
                }.value
                let filename = "simulator-\(suite.suiteID)-angle-\(angleSlug(entry.angleDegreesClockwise)).json"
                _ = try save(result, filename: filename)
                report = result
                entryResults.append(
                    SimulatorSuiteEntryResult(
                        angleDegreesClockwise: entry.angleDegreesClockwise,
                        success: true,
                        reportFilename: filename,
                        error: nil
                    )
                )
            } catch {
                entryResults.append(
                    SimulatorSuiteEntryResult(
                        angleDegreesClockwise: entry.angleDegreesClockwise,
                        success: false,
                        reportFilename: nil,
                        error: error.localizedDescription
                    )
                )
            }
        }
        let environment = ProcessInfo.processInfo.environment
        let aggregate = SimulatorSuiteAggregate(
            format: "slt_v17_ios_simulator_benchmark_aggregate",
            version: 1,
            suiteID: suite.suiteID,
            startedUTC: startedUTC,
            completedUTC: ISO8601DateFormatter().string(from: Date()),
            executionEnvironment: "simulator",
            simulatorDeviceName: environment["SIMULATOR_DEVICE_NAME"],
            simulatorModelIdentifier: environment["SIMULATOR_MODEL_IDENTIFIER"],
            simulatorRuntimeVersion: environment["SIMULATOR_RUNTIME_VERSION"],
            expectedLabel: suite.expectedLabel,
            iterations: suite.iterations,
            sourceVideoSHA256: suite.sourceVideoSHA256,
            hardwarePerformanceClaim: false,
            thermalsInterpretable: false,
            featureExtractionEnvironment: "host_macos_apple_vision",
            endToEndPipeline: true,
            videoFileToGlossEndToEnd: false,
            cameraToGlossEndToEnd: false,
            allMobileNeuralModelsInCoreML: true,
            simulatorVisionLimitation: Self.simulatorVisionLimitation,
            citizenTestAccessed: false,
            semlexTestAccessed: false,
            success: entryResults.allSatisfy(\.success),
            entries: entryResults
        )
        let aggregateURL = try reportsDirectory().appendingPathComponent(
            "simulator-suite-\(suite.suiteID)-aggregate.json"
        )
        try encode(aggregate, to: aggregateURL)
        guard aggregate.success else {
            throw BenchmarkError.model("One or more simulator conditions failed; see aggregate JSON")
        }
#endif
    }

    private func executeBenchmark(
        inputURL: URL,
        expected: String,
        mirrored: Bool,
        iterations: Int,
        inputRotationDegrees: Double?,
        sourceVideoSHA256: String?,
        generatedVideoSHA256: String?
    ) async throws -> BenchmarkReport {
        status = "Running Apple Vision extraction…"
        let prepared = try await V17Pipeline.prepareStage2(
            from: inputURL, inputMirrored: mirrored
        )
        status = "Running MobileCLIP2, Stage 2, and safe Stage 3…"
        let labelValues = labels
        let iOSVersion = UIDevice.current.systemVersion
        return try await Task.detached(priority: .userInitiated) {
            try Self.measureStage2(
                prepared: prepared.input,
                diagnostics: prepared.diagnostics,
                inputURL: inputURL,
                expected: expected,
                labels: labelValues,
                mirrored: mirrored,
                iterations: iterations,
                iOSVersion: iOSVersion,
                inputRotationDegrees: inputRotationDegrees,
                sourceVideoSHA256: sourceVideoSHA256,
                generatedVideoSHA256: generatedVideoSHA256,
                featureSHA256: nil,
                handCropBundleSHA256: nil,
                extractionExecutionEnvironment: "ios_apple_vision",
                simulatorVisionLimitation: nil
            )
        }.value
    }

    private nonisolated static func measureStage2(
        prepared: Stage2PreparedInput,
        diagnostics: V17Diagnostics,
        inputURL: URL,
        expected: String,
        labels: [String],
        mirrored: Bool,
        iterations: Int,
        iOSVersion: String,
        inputRotationDegrees: Double?,
        sourceVideoSHA256: String?,
        generatedVideoSHA256: String?,
        featureSHA256: String?,
        handCropBundleSHA256: String?,
        extractionExecutionEnvironment: String,
        simulatorVisionLimitation: String?
    ) throws -> BenchmarkReport {
        let thermalBefore = thermalName(ProcessInfo.processInfo.thermalState)
        let memoryBefore = residentMemory()
        let model = try Stage2MobileModelV17()
        let prediction = try model.predict(
            input: prepared, labels: labels, iterations: iterations
        )
        let stage3 = try Stage3MobileNaturalizerV17(stage2Manifest: model.manifest)
            .naturalize(
                tokenIndices: prediction.tokenIndices,
                glosses: prediction.labels,
                labels: labels
            )
        let ordered = prediction.inferenceTimingsMilliseconds.sorted()
        let median = ordered[ordered.count / 2]
        let p90 = ordered[min(ordered.count - 1, Int(Double(ordered.count - 1) * 0.90))]
        let predicted = prediction.labels.joined(separator: " ")
        let memoryAfter = residentMemory()
        let environment = ProcessInfo.processInfo.environment
#if targetEnvironment(simulator)
        let executionEnvironment = "simulator"
        let hardwarePerformanceClaim = false
        let thermalsInterpretable = false
#else
        let executionEnvironment = "physical_device"
        let hardwarePerformanceClaim = true
        let thermalsInterpretable = true
#endif
        return BenchmarkReport(
            id: UUID(),
            createdUTC: ISO8601DateFormatter().string(from: Date()),
            executionEnvironment: executionEnvironment,
            deviceModel: environment["SIMULATOR_MODEL_IDENTIFIER"] ?? machineIdentifier(),
            simulatorDeviceName: environment["SIMULATOR_DEVICE_NAME"],
            simulatorModelIdentifier: environment["SIMULATOR_MODEL_IDENTIFIER"],
            simulatorRuntimeVersion: environment["SIMULATOR_RUNTIME_VERSION"],
            hostArchitecture: architectureIdentifier(),
            iOSVersion: iOSVersion,
            thermalBefore: thermalBefore,
            thermalAfter: thermalName(ProcessInfo.processInfo.thermalState),
            thermalsInterpretable: thermalsInterpretable,
            hardwarePerformanceClaim: hardwarePerformanceClaim,
            inputFilename: inputURL.lastPathComponent,
            inputRotationDegreesClockwise: inputRotationDegrees,
            sourceVideoSHA256: sourceVideoSHA256,
            generatedVideoSHA256: generatedVideoSHA256,
            featureSHA256: featureSHA256,
            handFeatureSHA256: nil,
            handCropBundleSHA256: handCropBundleSHA256,
            extractionSucceeded: true,
            extractionExecutionEnvironment: extractionExecutionEnvironment,
            endToEndPipeline: true,
            simulatorVisionLimitation: simulatorVisionLimitation,
            residualRollDegrees: residualRoll(
                inputRotationDegrees: inputRotationDegrees,
                correctionDegrees: diagnostics.visionCoarseRotationClockwise
            ),
            expectedLabel: expected,
            predictedLabel: predicted,
            predictedLabels: prediction.labels,
            predictedTokenIndices: prediction.tokenIndices,
            correct: predicted == expected,
            inputMirrored: mirrored,
            iterations: iterations,
            coldInferenceMilliseconds: prediction.coldInferenceMilliseconds,
            medianInferenceMilliseconds: median,
            p90InferenceMilliseconds: p90,
            residentMemoryBeforeBytes: memoryBefore,
            residentMemoryAfterBytes: memoryAfter,
            modelBytes: prediction.modelBytes,
            candidateID: model.manifest.candidateID,
            checkpointSHA256: model.manifest.checkpointSHA256,
            coreMLPackageTreeSHA256: model.manifest.contextHeadPackageTreeSHA256,
            imageEncoderPackageTreeSHA256: model.manifest.imageEncoderPackageTreeSHA256,
            frozenEncoderPackageTreeSHA256: model.manifest.frozenEncoderPackageTreeSHA256,
            contextHeadPackageTreeSHA256: model.manifest.contextHeadPackageTreeSHA256,
            vocabularyManifestSHA256: model.manifest.vocabularyManifestSHA256,
            cropEncodingMilliseconds: prediction.cropEncodingMilliseconds,
            predictionAggregation: prediction.predictionAggregation,
            sequenceVoteCounts: prediction.sequenceVoteCounts,
            stage3NaturalEnglish: stage3.naturalEnglish,
            stage3LiteralEnglish: stage3.literalEnglish,
            stage3RenderingMode: stage3.renderingMode,
            stage3SafeFallbackUsed: stage3.safeFallbackUsed,
            stage3NaturalizerManifestSHA256: stage3.naturalizerManifestSHA256,
            allMobileNeuralModelsInCoreML: true,
            videoFileToGlossEndToEnd: extractionExecutionEnvironment == "ios_apple_vision",
            cameraToGlossEndToEnd: false,
            diagnostics: diagnostics
        )
    }

    private nonisolated static func measure(
        features: MLMultiArray,
        handEmbeddings: MLMultiArray,
        handValid: MLMultiArray,
        handBoxes: MLMultiArray,
        diagnostics: V17Diagnostics,
        inputURL: URL,
        expected: String,
        labels: [String],
        mirrored: Bool,
        iterations: Int,
        iOSVersion: String,
        inputRotationDegrees: Double?,
        sourceVideoSHA256: String?,
        generatedVideoSHA256: String?,
        featureSHA256: String?,
        handFeatureSHA256: String?,
        extractionExecutionEnvironment: String,
        endToEndPipeline: Bool,
        simulatorVisionLimitation: String?
    ) throws -> BenchmarkReport {
        guard let modelURL = Bundle.main.url(
            forResource: "Stage1UnifiedMultimodalV17FP32", withExtension: "mlmodelc"
        ) else {
            throw BenchmarkError.model("Stage1UnifiedMultimodalV17FP32.mlmodelc is not bundled")
        }
        guard let manifestURL = Bundle.main.url(
            forResource: "Stage1OrientationV17_manifest", withExtension: "json"
        ) else {
            throw BenchmarkError.model("Stage1OrientationV17_manifest.json is not bundled")
        }
        let bundledManifest = try JSONDecoder().decode(
            BundledModelManifest.self, from: Data(contentsOf: manifestURL)
        )
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let thermalBefore = thermalName(ProcessInfo.processInfo.thermalState)
        let memoryBefore = residentMemory()
        let model = try MLModel(contentsOf: modelURL, configuration: configuration)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "landmarks": features,
            "hand_embeddings": handEmbeddings,
            "hand_valid": handValid,
            "hand_boxes": handBoxes,
        ])
        let clock = ContinuousClock()
        var started = clock.now
        var prediction = try model.prediction(from: provider)
        let cold = milliseconds(started.duration(to: clock.now))
        for _ in 0..<10 { prediction = try model.prediction(from: provider) }
        var timings: [Double] = []
        timings.reserveCapacity(iterations)
        for _ in 0..<iterations {
            started = clock.now
            prediction = try model.prediction(from: provider)
            timings.append(milliseconds(started.duration(to: clock.now)))
        }
        guard let outputName = model.modelDescription.outputDescriptionsByName.first(where: {
            $0.value.type == .multiArray
        })?.key,
        let logits = prediction.featureValue(for: outputName)?.multiArrayValue,
        logits.count == labels.count else {
            throw BenchmarkError.model("Core ML output is not a 100-class multi-array")
        }
        let predictedIndex = (0..<logits.count).max {
            logits[$0].doubleValue < logits[$1].doubleValue
        } ?? 0
        let ordered = timings.sorted()
        let median = ordered[ordered.count / 2]
        let p90 = ordered[min(ordered.count - 1, Int(Double(ordered.count - 1) * 0.90))]
        let predicted = labels[predictedIndex]
        let memoryAfter = residentMemory()
        let modelBytes = directoryBytes(modelURL)
        let environment = ProcessInfo.processInfo.environment
#if targetEnvironment(simulator)
        let executionEnvironment = "simulator"
        let hardwarePerformanceClaim = false
        let thermalsInterpretable = false
#else
        let executionEnvironment = "physical_device"
        let hardwarePerformanceClaim = true
        let thermalsInterpretable = true
#endif
        return BenchmarkReport(
            id: UUID(),
            createdUTC: ISO8601DateFormatter().string(from: Date()),
            executionEnvironment: executionEnvironment,
            deviceModel: environment["SIMULATOR_MODEL_IDENTIFIER"] ?? machineIdentifier(),
            simulatorDeviceName: environment["SIMULATOR_DEVICE_NAME"],
            simulatorModelIdentifier: environment["SIMULATOR_MODEL_IDENTIFIER"],
            simulatorRuntimeVersion: environment["SIMULATOR_RUNTIME_VERSION"],
            hostArchitecture: architectureIdentifier(),
            iOSVersion: iOSVersion,
            thermalBefore: thermalBefore,
            thermalAfter: thermalName(ProcessInfo.processInfo.thermalState),
            thermalsInterpretable: thermalsInterpretable,
            hardwarePerformanceClaim: hardwarePerformanceClaim,
            inputFilename: inputURL.lastPathComponent,
            inputRotationDegreesClockwise: inputRotationDegrees,
            sourceVideoSHA256: sourceVideoSHA256,
            generatedVideoSHA256: generatedVideoSHA256,
            featureSHA256: featureSHA256,
            handFeatureSHA256: handFeatureSHA256,
            handCropBundleSHA256: nil,
            extractionSucceeded: true,
            extractionExecutionEnvironment: extractionExecutionEnvironment,
            endToEndPipeline: endToEndPipeline,
            simulatorVisionLimitation: simulatorVisionLimitation,
            residualRollDegrees: residualRoll(
                inputRotationDegrees: inputRotationDegrees,
                correctionDegrees: diagnostics.visionCoarseRotationClockwise
            ),
            expectedLabel: expected,
            predictedLabel: predicted,
            predictedLabels: [predicted],
            predictedTokenIndices: [predictedIndex + 1],
            correct: predicted == expected,
            inputMirrored: mirrored,
            iterations: iterations,
            coldInferenceMilliseconds: cold,
            medianInferenceMilliseconds: median,
            p90InferenceMilliseconds: p90,
            residentMemoryBeforeBytes: memoryBefore,
            residentMemoryAfterBytes: memoryAfter,
            modelBytes: modelBytes,
            candidateID: bundledManifest.candidateID,
            checkpointSHA256: bundledManifest.checkpointSHA256,
            coreMLPackageTreeSHA256: bundledManifest.coreMLPackageTreeSHA256,
            imageEncoderPackageTreeSHA256: nil,
            frozenEncoderPackageTreeSHA256: nil,
            contextHeadPackageTreeSHA256: nil,
            vocabularyManifestSHA256: nil,
            cropEncodingMilliseconds: nil,
            predictionAggregation: "single_final_inference",
            sequenceVoteCounts: [String(predictedIndex + 1): iterations],
            stage3NaturalEnglish: predicted,
            stage3LiteralEnglish: predicted,
            stage3RenderingMode: "legacy_stage1_not_stage3",
            stage3SafeFallbackUsed: true,
            stage3NaturalizerManifestSHA256: "",
            allMobileNeuralModelsInCoreML: false,
            videoFileToGlossEndToEnd: false,
            cameraToGlossEndToEnd: false,
            diagnostics: diagnostics
        )
    }

    private func save(_ report: BenchmarkReport, filename: String? = nil) throws -> URL {
        let name = filename ?? "orientation-v17-\(UUID().uuidString.lowercased()).json"
        let url = try reportsDirectory().appendingPathComponent(name)
        try encode(report, to: url)
        return url
    }

    private func reportsDirectory() throws -> URL {
        let directory = try documentsDirectory().appendingPathComponent(
            "benchmark_reports", isDirectory: true
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func documentsDirectory() throws -> URL {
        guard let directory = FileManager.default.urls(
            for: .documentDirectory, in: .userDomainMask
        ).first else {
            throw BenchmarkError.model("Application Documents directory is unavailable")
        }
        return directory
    }

    private func safeRelativeURL(_ relativePath: String, under root: URL) throws -> URL {
        let components = NSString(string: relativePath).pathComponents
        guard !relativePath.hasPrefix("/"), !components.contains("..") else {
            throw BenchmarkError.model("Unsafe relative suite path: \(relativePath)")
        }
        let candidate = root.appendingPathComponent(relativePath).standardizedFileURL
        guard candidate.path.hasPrefix(root.standardizedFileURL.path + "/") else {
            throw BenchmarkError.model("Suite path escapes Documents: \(relativePath)")
        }
        return candidate
    }

    private func encode<T: Encodable>(_ value: T, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(value).write(to: url, options: .atomic)
    }

    private func angleSlug(_ angle: Double) -> String {
        if angle.rounded() == angle { return String(Int(angle)) }
        return String(format: "%.3f", angle).replacingOccurrences(of: ".", with: "p")
    }

    private nonisolated static func sha256File(_ url: URL) throws -> String {
        let digest = SHA256.hash(data: try Data(contentsOf: url))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private nonisolated static let simulatorVisionLimitation =
        "The installed iOS Simulator runtime omits Apple Vision pose Espresso weight files; "
        + "features were extracted by the same v17 Apple Vision pipeline on the macOS host."

    private nonisolated static func loadFeatureArray(_ url: URL) throws -> MLMultiArray {
        let expectedCount = V17Pipeline.frameCount * V17Pipeline.nodeCount
            * V17Pipeline.channelCount
        let data = try Data(contentsOf: url)
        guard data.count == expectedCount * MemoryLayout<Float>.size else {
            throw BenchmarkError.model(
                "Unexpected host feature byte count \(data.count); expected \(expectedCount * 4)"
            )
        }
        let array = try MLMultiArray(
            shape: [
                1,
                NSNumber(value: V17Pipeline.frameCount),
                NSNumber(value: V17Pipeline.nodeCount),
                NSNumber(value: V17Pipeline.channelCount),
            ],
            dataType: .float32
        )
        data.withUnsafeBytes { rawBuffer in
            let values = rawBuffer.bindMemory(to: Float.self)
            for index in 0..<expectedCount {
                array[index] = NSNumber(value: values[index])
            }
        }
        return array
    }

    private nonisolated static func loadHandFeatureArrays(
        _ url: URL
    ) throws -> (embeddings: MLMultiArray, valid: MLMultiArray, boxes: MLMultiArray) {
        let embeddingCount = 16 * 3 * 512
        let validCount = 16 * 3
        let boxCount = 16 * 3 * 4
        let expectedCount = embeddingCount + validCount + boxCount
        let data = try Data(contentsOf: url)
        guard data.count == expectedCount * MemoryLayout<Float>.size else {
            throw BenchmarkError.model(
                "Unexpected host hand-feature byte count \(data.count); expected \(expectedCount * 4)"
            )
        }
        let embeddings = try MLMultiArray(
            shape: [1, 16, 3, 512].map { NSNumber(value: $0) }, dataType: .float32
        )
        let valid = try MLMultiArray(
            shape: [1, 16, 3].map { NSNumber(value: $0) }, dataType: .float32
        )
        let boxes = try MLMultiArray(
            shape: [1, 16, 3, 4].map { NSNumber(value: $0) }, dataType: .float32
        )
        data.withUnsafeBytes { rawBuffer in
            let values = rawBuffer.bindMemory(to: Float.self)
            for index in 0..<embeddingCount {
                embeddings[index] = NSNumber(value: values[index])
            }
            for index in 0..<validCount {
                valid[index] = NSNumber(value: values[embeddingCount + index])
            }
            for index in 0..<boxCount {
                boxes[index] = NSNumber(value: values[embeddingCount + validCount + index])
            }
        }
        return (embeddings, valid, boxes)
    }

    private nonisolated static func residualRoll(
        inputRotationDegrees: Double?, correctionDegrees: Int
    ) -> Double? {
        guard let inputRotationDegrees else { return nil }
        var value = (inputRotationDegrees + Double(correctionDegrees))
            .truncatingRemainder(dividingBy: 360)
        if value > 180 { value -= 360 }
        if value <= -180 { value += 360 }
        return value
    }

    private nonisolated static func milliseconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) * 1_000
            + Double(duration.components.attoseconds) / 1.0e15
    }

    private nonisolated static func thermalName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private nonisolated static func residentMemory() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? UInt64(info.resident_size) : 0
    }

    private nonisolated static func directoryBytes(_ url: URL) -> UInt64 {
        guard let enumerator = FileManager.default.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey]
        ) else { return 0 }
        var total: UInt64 = 0
        for case let item as URL in enumerator {
            total += UInt64((try? item.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    private nonisolated static func machineIdentifier() -> String {
        var info = utsname()
        uname(&info)
        return Mirror(reflecting: info.machine).children.reduce(into: "") { value, element in
            guard let byte = element.value as? Int8, byte != 0 else { return }
            value.append(Character(UnicodeScalar(UInt8(byte))))
        }
    }

    private nonisolated static func architectureIdentifier() -> String {
#if arch(arm64)
        return "arm64"
#elseif arch(x86_64)
        return "x86_64"
#else
        return "unknown"
#endif
    }
}
