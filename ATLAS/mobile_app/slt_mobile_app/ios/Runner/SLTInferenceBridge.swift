import CryptoKit
import CoreML
import Darwin.Mach
import Flutter
import Foundation
import UIKit

private struct MobileInferenceReportV17: Codable {
  let format: String
  let version: Int
  let createdUTC: String
  let executionEnvironment: String
  let deviceModel: String
  let iOSVersion: String
  let benchmarkEnabled: Bool
  let iterations: Int
  let hardwarePerformanceClaim: Bool
  let thermalsInterpretable: Bool
  let cameraToGlossEndToEnd: Bool
  let diagnosticCaptureID: String
  let inputFilename: String
  let inputMirrored: Bool
  let sourceVideoSHA256: String?
  let predictedTokenIndices: [Int]
  let predictedGlosses: [String]
  let inferenceRoute: String
  let selectedActivityWindow: Int?
  let naturalEnglish: String
  let literalEnglish: String
  let renderingMode: String
  let safeFallbackUsed: Bool
  let totalMilliseconds: Double
  let extractionMilliseconds: Double
  let cropEncodingMilliseconds: Double
  let coldInferenceMilliseconds: Double
  let routeInferenceMilliseconds: Double
  let medianInferenceMilliseconds: Double
  let p90InferenceMilliseconds: Double
  let residentMemoryBeforeBytes: UInt64
  let residentMemoryAfterBytes: UInt64
  let thermalBefore: String
  let thermalAfter: String
  let modelBytes: UInt64
  let candidateID: String
  let checkpointSHA256: String
  let imageEncoderPackageTreeSHA256: String
  let frozenEncoderPackageTreeSHA256: String
  let stage1PackageTreeSHA256: String
  let phraseHeadPackageTreeSHA256: String
  let isolatedCorrectionPackageTreeSHA256: String
  let stage1CheckpointSHA256: String
  let phraseCheckpointSHA256: String
  let isolatedCorrectionCheckpointSHA256: String
  let vocabularyManifestSHA256: String
  let naturalizerManifestSHA256: String
  let diagnostics: V17Diagnostics
}

enum SLTInferenceBridge {
  static let channelName = "slt_mobile_app/inference_v17"
  static let benchmarkDefaultsKey = "SLTBenchmarkEnabledV17"

  private struct Resources {
    let labels: [String]
    let model: Stage2MobileModelV17
    let naturalizer: Stage3MobileNaturalizerV17
  }

  private static let cachedResources: Result<Resources, Error> = Result {
    let labels = try loadLabels()
    let model = try Stage2MobileModelV17()
    return Resources(
      labels: labels,
      model: model,
      naturalizer: try Stage3MobileNaturalizerV17(stage2Manifest: model.manifest)
    )
  }

  static func inferVideo(
    path: String,
    mirrored: Bool,
    benchmarkEnabled: Bool
  ) async throws -> [String: Any] {
    let temporaryURL = URL(fileURLWithPath: path).standardizedFileURL
    guard temporaryURL.isFileURL,
          FileManager.default.fileExists(atPath: temporaryURL.path) else {
      throw BenchmarkError.invalidVideo("The recorded video is missing")
    }

    let totalStarted = ContinuousClock.now
    let capture = try createDiagnosticCapture(from: temporaryURL)
    do {
      let memoryBefore = residentMemory()
      let thermalBefore = thermalName(ProcessInfo.processInfo.thermalState)
      let resources = try cachedResources.get()
      let prepared = try await V17Pipeline.prepareStage2(
        from: capture.video,
        inputMirrored: mirrored,
        cameraCapture: true
      )
      let iterations = benchmarkEnabled ? 20 : 1
      let prediction = try resources.model.predict(
        input: prepared.input,
        labels: resources.labels,
        iterations: iterations,
        warmupIterations: benchmarkEnabled ? 5 : 0
      )
      let text = try resources.naturalizer.naturalize(
        tokenIndices: prediction.tokenIndices,
        glosses: prediction.labels,
        labels: resources.labels
      )
      let ordered = prediction.inferenceTimingsMilliseconds.sorted()
      let median = ordered[ordered.count / 2]
      let p90 = ordered[
        min(ordered.count - 1, Int(Double(ordered.count - 1) * 0.90))
      ]
      let totalMilliseconds = milliseconds(totalStarted.duration(to: .now))
      let memoryAfter = residentMemory()
      let thermalAfter = thermalName(ProcessInfo.processInfo.thermalState)
#if targetEnvironment(simulator)
      let executionEnvironment = "simulator"
      let hardwarePerformanceClaim = false
      let thermalsInterpretable = false
#else
      let executionEnvironment = "physical_device"
      let hardwarePerformanceClaim = true
      let thermalsInterpretable = true
#endif
      let report = MobileInferenceReportV17(
        format: "slt_mobile_inference_report_v17",
        version: 3,
        createdUTC: ISO8601DateFormatter().string(from: Date()),
        executionEnvironment: executionEnvironment,
        deviceModel: deviceModel(),
        iOSVersion: UIDevice.current.systemVersion,
        benchmarkEnabled: benchmarkEnabled,
        iterations: iterations,
        hardwarePerformanceClaim: hardwarePerformanceClaim,
        thermalsInterpretable: thermalsInterpretable,
        cameraToGlossEndToEnd: true,
        diagnosticCaptureID: capture.id,
        inputFilename: capture.video.lastPathComponent,
        inputMirrored: mirrored,
        sourceVideoSHA256: sha256File(capture.video),
        predictedTokenIndices: prediction.tokenIndices,
        predictedGlosses: prediction.labels,
        inferenceRoute: prediction.inferenceRoute,
        selectedActivityWindow: prediction.selectedActivityWindow,
        naturalEnglish: text.naturalEnglish,
        literalEnglish: text.literalEnglish,
        renderingMode: text.renderingMode,
        safeFallbackUsed: text.safeFallbackUsed,
        totalMilliseconds: totalMilliseconds,
        extractionMilliseconds: prepared.diagnostics.extractionMilliseconds,
        cropEncodingMilliseconds: prediction.cropEncodingMilliseconds,
        coldInferenceMilliseconds: prediction.coldInferenceMilliseconds,
        routeInferenceMilliseconds: prediction.routeInferenceMilliseconds,
        medianInferenceMilliseconds: median,
        p90InferenceMilliseconds: p90,
        residentMemoryBeforeBytes: memoryBefore,
        residentMemoryAfterBytes: memoryAfter,
        thermalBefore: thermalBefore,
        thermalAfter: thermalAfter,
        modelBytes: prediction.modelBytes,
        candidateID: resources.model.manifest.candidateID,
        checkpointSHA256: resources.model.manifest.checkpointSHA256,
        imageEncoderPackageTreeSHA256: resources.model.manifest.imageEncoderPackageTreeSHA256,
        frozenEncoderPackageTreeSHA256: resources.model.manifest.frozenEncoderPackageTreeSHA256,
        stage1PackageTreeSHA256: resources.model.manifest.stage1PackageTreeSHA256,
        phraseHeadPackageTreeSHA256: resources.model.manifest.phraseHeadPackageTreeSHA256,
        isolatedCorrectionPackageTreeSHA256: resources.model.manifest.isolatedCorrectionPackageTreeSHA256,
        stage1CheckpointSHA256: resources.model.manifest.stage1CheckpointSHA256,
        phraseCheckpointSHA256: resources.model.manifest.phraseCheckpointSHA256,
        isolatedCorrectionCheckpointSHA256: resources.model.manifest.isolatedCorrectionCheckpointSHA256,
        vocabularyManifestSHA256: resources.model.manifest.vocabularyManifestSHA256,
        naturalizerManifestSHA256: text.naturalizerManifestSHA256,
        diagnostics: prepared.diagnostics
      )

      try saveDiagnosticTensors(
        prepared.input,
        handEmbeddings: prediction.handEmbeddings,
        to: capture.directory
      )
      let reportURL = try save(report, to: capture.directory)
      var result: [String: Any] = [
        "glosses": prediction.labels,
        "english": text.naturalEnglish,
        "literalEnglish": text.literalEnglish,
        "renderingMode": text.renderingMode,
        "safeFallbackUsed": text.safeFallbackUsed,
        "diagnosticPath": capture.directory.path,
      ]
      if benchmarkEnabled {
        result["reportPath"] = reportURL.path
        result["benchmark"] = [
          "executionEnvironment": executionEnvironment,
          "iterations": iterations,
          "hardwarePerformanceClaim": hardwarePerformanceClaim,
          "thermalsInterpretable": thermalsInterpretable,
          "totalMilliseconds": totalMilliseconds,
          "extractionMilliseconds": prepared.diagnostics.extractionMilliseconds,
          "cropEncodingMilliseconds": prediction.cropEncodingMilliseconds,
          "inferenceRoute": prediction.inferenceRoute,
          "selectedActivityWindow": prediction.selectedActivityWindow ?? -1,
          "routeInferenceMilliseconds": prediction.routeInferenceMilliseconds,
          "medianInferenceMilliseconds": median,
          "p90InferenceMilliseconds": p90,
          "residentMemoryBeforeBytes": memoryBefore,
          "residentMemoryAfterBytes": memoryAfter,
          "thermalBefore": thermalBefore,
          "thermalAfter": thermalAfter,
        ]
      }
      return result
    } catch {
      try? saveJSON([
        "format": "slt_mobile_inference_error_v17",
        "version": 1,
        "createdUTC": ISO8601DateFormatter().string(from: Date()),
        "error": error.localizedDescription,
      ], to: capture.directory.appendingPathComponent("error.json"))
      throw error
    }
  }

  static func validatedReportURL(path: String) throws -> URL {
    let candidate = URL(fileURLWithPath: path).standardizedFileURL
    let root = try diagnosticsDirectory().standardizedFileURL
    let captureDirectory = candidate.deletingLastPathComponent()
    guard captureDirectory.deletingLastPathComponent() == root,
          candidate.lastPathComponent == "report.json",
          FileManager.default.fileExists(atPath: candidate.path) else {
      throw BenchmarkError.model("The benchmark report path is invalid")
    }
    return candidate
  }

  private static func loadLabels() throws -> [String] {
    guard let url = Bundle.main.url(
      forResource: "citizen100_manifest",
      withExtension: "json"
    ) else {
      throw BenchmarkError.model("citizen100_manifest.json is not bundled")
    }
    let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
    guard let payload = object as? [String: Any],
          let classes = payload["classes"] as? [[String: Any]] else {
      throw BenchmarkError.model("The locked vocabulary manifest is invalid")
    }
    let labels = classes.sorted {
      ($0["class_index"] as? Int ?? -1) < ($1["class_index"] as? Int ?? -1)
    }.compactMap { $0["canonical_label"] as? String }
    guard labels.count == 100 else {
      throw BenchmarkError.model("Expected exactly 100 locked glosses")
    }
    return labels
  }

  private static func save(
    _ report: MobileInferenceReportV17,
    to directory: URL
  ) throws -> URL {
    let url = directory.appendingPathComponent("report.json")
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(report).write(to: url, options: .atomic)
    return url
  }

  private static func createDiagnosticCapture(
    from source: URL
  ) throws -> (id: String, directory: URL, video: URL) {
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.timeZone = TimeZone(secondsFromGMT: 0)
    formatter.dateFormat = "yyyyMMdd-HHmmss-SSS"
    let id = "\(formatter.string(from: Date()))_\(UUID().uuidString.prefix(8))"
    let directory = try diagnosticsDirectory().appendingPathComponent(
      id,
      isDirectory: true
    )
    try FileManager.default.createDirectory(
      at: directory,
      withIntermediateDirectories: false
    )
    let video = directory.appendingPathComponent("recording.mp4")
    try FileManager.default.copyItem(at: source, to: video)
    return (id, directory, video)
  }

  private static func diagnosticsDirectory() throws -> URL {
    let documents = try FileManager.default.url(
      for: .documentDirectory,
      in: .userDomainMask,
      appropriateFor: nil,
      create: true
    )
    let directory = documents.appendingPathComponent("Diagnostics", isDirectory: true)
    try FileManager.default.createDirectory(
      at: directory,
      withIntermediateDirectories: true
    )
    return directory
  }

  private static func saveDiagnosticTensors(
    _ input: Stage2PreparedInput,
    handEmbeddings: MLMultiArray,
    to directory: URL
  ) throws {
    let tensors: [(String, MLMultiArray, String)] = [
      ("raw_landmarks.npy", input.rawLandmarks, "upright x,y,confidence before normalization"),
      ("model_landmarks.npy", input.landmarks, "exact normalized Stage-2 landmark input"),
      ("source_frame_mask.npy", input.sourceFrameMask, "valid raw source frames"),
      ("hand_valid.npy", input.handValid, "valid RGB hand views"),
      ("hand_boxes.npy", input.handBoxes, "normalized RGB crop boxes"),
      ("window_mask.npy", input.windowMask, "valid 32-frame windows"),
      ("hand_features.npy", handEmbeddings, "exact MobileCLIP2 hand embeddings"),
    ]
    for (filename, array, _) in tensors {
      try saveNpy(array, to: directory.appendingPathComponent(filename))
    }
    let entries = tensors.map { filename, array, description in
      [
        "filename": filename,
        "dtype": "float32",
        "shape": array.shape.map(\.intValue),
        "description": description,
      ] as [String: Any]
    }
    try saveJSON([
      "format": "slt_mobile_diagnostic_tensors_v17",
      "version": 1,
      "cropImagesSaved": false,
      "cropImagesRegenerableFromRecordingAndBoxes": true,
      "tensors": entries,
    ], to: directory.appendingPathComponent("tensor_manifest.json"))
  }

  private static func saveNpy(_ array: MLMultiArray, to url: URL) throws {
    guard array.dataType == .float32 else {
      throw BenchmarkError.model("Diagnostic tensors must be float32")
    }
    let shape = array.shape.map(\.intValue)
    let tuple = shape.count == 1
      ? "\(shape[0]),"
      : shape.map(String.init).joined(separator: ", ")
    var header = "{'descr': '<f4', 'fortran_order': False, 'shape': (\(tuple)), }"
    let prefixBytes = 10
    let padding = (64 - ((prefixBytes + header.utf8.count + 1) % 64)) % 64
    header += String(repeating: " ", count: padding) + "\n"
    guard let headerData = header.data(using: .ascii),
          headerData.count <= Int(UInt16.max) else {
      throw BenchmarkError.model("Diagnostic NumPy header is invalid")
    }
    var output = Data([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00])
    var headerLength = UInt16(headerData.count).littleEndian
    withUnsafeBytes(of: &headerLength) { output.append(contentsOf: $0) }
    output.append(headerData)
    output.reserveCapacity(output.count + array.count * 4)
    for index in 0..<array.count {
      var bits = array[index].floatValue.bitPattern.littleEndian
      withUnsafeBytes(of: &bits) { output.append(contentsOf: $0) }
    }
    try output.write(to: url, options: .atomic)
  }

  private static func saveJSON(_ object: Any, to url: URL) throws {
    let data = try JSONSerialization.data(
      withJSONObject: object,
      options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    )
    try data.write(to: url, options: .atomic)
  }

  private static func sha256File(_ url: URL) -> String? {
    guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
    defer { try? handle.close() }
    var hasher = SHA256()
    while let data = try? handle.read(upToCount: 1 << 20), !data.isEmpty {
      hasher.update(data: data)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }

  private static func milliseconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) * 1_000
      + Double(duration.components.attoseconds) / 1.0e15
  }

  private static func thermalName(_ state: ProcessInfo.ThermalState) -> String {
    switch state {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
  }

  private static func residentMemory() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let result = withUnsafeMutablePointer(to: &info) { pointer in
      pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(
          mach_task_self_,
          task_flavor_t(MACH_TASK_BASIC_INFO),
          $0,
          &count
        )
      }
    }
    return result == KERN_SUCCESS ? UInt64(info.resident_size) : 0
  }

  private static func deviceModel() -> String {
    if let simulator = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] {
      return simulator
    }
    var systemInfo = utsname()
    uname(&systemInfo)
    return withUnsafePointer(to: &systemInfo.machine) {
      $0.withMemoryRebound(to: CChar.self, capacity: 1) {
        String(cString: $0)
      }
    }
  }
}
