import AVFoundation
import CryptoKit
import Foundation
import UIKit

enum CaptureAppError: LocalizedError {
    case invalidResource(String)
    case invalidCSV(String)
    case unsafePath(String)
    case missingField(String)
    case capture(String)

    var errorDescription: String? {
        switch self {
        case .invalidResource(let value): return "Invalid bundled resource: \(value)"
        case .invalidCSV(let value): return "Invalid CSV: \(value)"
        case .unsafePath(let value): return "Unsafe output path: \(value)"
        case .missingField(let value): return "Required field is missing: \(value)"
        case .capture(let value): return value
        }
    }
}

enum CSVCodec {
    static func rows(from text: String) throws -> [[String: String]] {
        let records = parseRecords(text)
        guard let header = records.first, !header.isEmpty else {
            throw CaptureAppError.invalidCSV("missing header")
        }
        var output: [[String: String]] = []
        for (offset, fields) in records.dropFirst().enumerated() where !fields.allSatisfy({ $0.isEmpty }) {
            guard fields.count == header.count else {
                throw CaptureAppError.invalidCSV(
                    "row \(offset + 2) has \(fields.count) fields; expected \(header.count)"
                )
            }
            output.append(Dictionary(uniqueKeysWithValues: zip(header, fields)))
        }
        return output
    }

    static func line(fields: [String], row: [String: String]) -> String {
        fields.map { quote(row[$0] ?? "") }.joined(separator: ",") + "\n"
    }

    private static func quote(_ value: String) -> String {
        guard value.contains(",") || value.contains("\"") || value.contains("\n") else {
            return value
        }
        return "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }

    private static func parseRecords(_ text: String) -> [[String]] {
        var records: [[String]] = []
        var record: [String] = []
        var field = ""
        var quoted = false
        var index = text.startIndex
        while index < text.endIndex {
            let character = text[index]
            if quoted {
                if character == "\"" {
                    let next = text.index(after: index)
                    if next < text.endIndex && text[next] == "\"" {
                        field.append("\"")
                        index = next
                    } else {
                        quoted = false
                    }
                } else {
                    field.append(character)
                }
            } else if character == "\"" {
                quoted = true
            } else if character == "," {
                record.append(field)
                field = ""
            } else if character == "\n" {
                record.append(field.trimmingCharacters(in: CharacterSet(charactersIn: "\r")))
                records.append(record)
                record = []
                field = ""
            } else {
                field.append(character)
            }
            index = text.index(after: index)
        }
        if !field.isEmpty || !record.isEmpty {
            record.append(field)
            records.append(record)
        }
        return records
    }
}

struct CapturePlan: Identifiable, Hashable {
    let fields: [String: String]
    let referenceURL: URL?

    var id: String { fields["planned_id"] ?? "" }
    var signerID: String { fields["signer_id"] ?? "" }
    var sessionID: String { fields["session_id"] ?? "" }
    var label: String { fields["canonical_label"] ?? "" }
    var expectedRawGloss: String { fields["expected_raw_gloss"] ?? "" }
    var aslLexCode: String { fields["citizen_asl_lex_code"] ?? "" }
    var promptOrder: Int { Int(fields["prompt_order"] ?? "") ?? 0 }
    var isOOV: Bool { label == "UNKNOWN" }
}

struct RecordedVideoMetadata {
    let width: Int
    let height: Int
    let fps: Double
}

@MainActor
final class CaptureStore: ObservableObject {
    static let ledgerFields = [
        "capture_id", "planned_id", "attempt", "signer_id", "session_id",
        "class_index", "canonical_label", "expected_raw_gloss", "citizen_asl_lex_code",
        "performed_gloss", "repetition", "prompt_order", "prompt_hidden_before_capture",
        "video_path", "video_sha256", "recorded_utc", "device_model", "ios_version",
        "camera", "width", "height", "fps", "orientation", "mirrored", "lighting",
        "background", "objective_qc_status", "objective_qc_reason"
    ]

    @Published var plans: [CapturePlan] = []
    @Published var selectedSigner = "S01" { didSet { selectFirstSession() } }
    @Published var selectedSession = "S01_r1" { didSet { selectFirstPending() } }
    @Published var currentIndex = 0
    @Published var lighting = "indoor_even"
    @Published var background = "ordinary_room"
    @Published var oovDescription = ""
    @Published var promptHidden = false
    @Published var countdown: Int?
    @Published var statusMessage = "Load the frozen pack and allow camera access."
    @Published var errorMessage: String?

    private var referenceByLabel: [String: URL] = [:]
    private var attemptsByPlan: [String: Int] = [:]
    private var capturedPlans: Set<String> = []
    private let documentsURL = FileManager.default.urls(
        for: .documentDirectory, in: .userDomainMask
    )[0]

    var signers: [String] {
        Array(Set(plans.map(\.signerID))).sorted()
    }

    var sessions: [String] {
        Array(Set(plans.filter { $0.signerID == selectedSigner }.map(\.sessionID))).sorted {
            sessionRank($0) < sessionRank($1)
        }
    }

    var sessionPlans: [CapturePlan] {
        plans.filter { $0.sessionID == selectedSession }.sorted { $0.promptOrder < $1.promptOrder }
    }

    var currentPlan: CapturePlan? {
        let values = sessionPlans
        return values.indices.contains(currentIndex) ? values[currentIndex] : nil
    }

    var completedInSession: Int {
        sessionPlans.filter { capturedPlans.contains($0.id) }.count
    }

    func loadFrozenPack() {
        do {
            guard
                let ledgerURL = Bundle.main.url(forResource: "capture_ledger", withExtension: "csv"),
                let reviewURL = Bundle.main.url(
                    forResource: "portrait_iphone_variant_review_v17", withExtension: "csv"
                ),
                let manifestURL = Bundle.main.url(
                    forResource: "capture_pack_manifest", withExtension: "json"
                )
            else {
                throw CaptureAppError.invalidResource("ledger, review, or manifest not bundled")
            }
            let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL))
            guard
                let object = manifest as? [String: Any],
                object["format"] as? String == "slt_v17_portrait_iphone_capture_pack",
                object["status"] as? String == "capture_pending",
                object["test_splits_accessed"] as? Bool == false,
                object["model_inference_accessed"] as? Bool == false,
                let protocolValue = object["orientation_protocol"] as? [String: Any],
                protocolValue["capture_orientation_restricted"] as? Bool == false,
                protocolValue["native_aspect_ratio_required"] as? Bool == true,
                protocolValue["pixel_stretching_allowed"] as? Bool == false
            else {
                throw CaptureAppError.invalidResource("capture pack safety fields changed")
            }
            let reviewRows = try CSVCodec.rows(from: String(contentsOf: reviewURL, encoding: .utf8))
            for row in reviewRows {
                guard row["review_status"] == "approved" else {
                    throw CaptureAppError.invalidResource("variant review is not fully approved")
                }
                if let label = row["canonical_label"],
                   let value = row["asllex_reference_url"],
                   let url = URL(string: value) {
                    referenceByLabel[label] = url
                }
            }
            let ledgerRows = try CSVCodec.rows(from: String(contentsOf: ledgerURL, encoding: .utf8))
            guard ledgerRows.count == 1_100 else {
                throw CaptureAppError.invalidResource("expected 1,100 frozen ledger rows")
            }
            plans = try ledgerRows.map { row in
                for field in ["planned_id", "signer_id", "session_id", "canonical_label", "prompt_order"]
                where (row[field] ?? "").isEmpty {
                    throw CaptureAppError.missingField(field)
                }
                return CapturePlan(fields: row, referenceURL: referenceByLabel[row["canonical_label"] ?? ""])
            }
            try loadExistingUpdates()
            selectFirstSession()
            statusMessage = "Frozen 1,100-plan pack loaded. No model is present in this app."
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func beginCountdown(camera: CameraRecorder) {
        guard let plan = currentPlan else { return }
        guard !lighting.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            errorMessage = CaptureAppError.missingField("lighting").localizedDescription
            return
        }
        guard !background.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            errorMessage = CaptureAppError.missingField("background").localizedDescription
            return
        }
        if plan.isOOV && oovDescription.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            errorMessage = CaptureAppError.missingField("OOV performed_gloss").localizedDescription
            return
        }
        promptHidden = true
        countdown = 3
        statusMessage = "Prompt hidden. Hold neutral position."
        Task {
            for value in stride(from: 3, through: 1, by: -1) {
                countdown = value
                try? await Task.sleep(for: .seconds(1))
            }
            countdown = nil
            do {
                let attempt = (attemptsByPlan[plan.id] ?? 0) + 1
                let captureID = String(format: "%@-a%02d", plan.id, attempt)
                let temporary = FileManager.default.temporaryDirectory
                    .appendingPathComponent("\(captureID)-\(UUID().uuidString).mov")
                try camera.startRecording(to: temporary) { [weak self] result in
                    Task { @MainActor in
                        await self?.finishRecording(result: result, plan: plan, attempt: attempt)
                    }
                }
                statusMessage = "Recording. Include the complete sign and then stop."
            } catch {
                promptHidden = false
                errorMessage = error.localizedDescription
            }
        }
    }

    func stop(camera: CameraRecorder) {
        camera.stopRecording()
        statusMessage = "Finalizing video and immutable metadata…"
    }

    func captureAnotherAttempt() {
        promptHidden = false
        statusMessage = "A new numbered attempt will preserve the previous file."
    }

    private func finishRecording(
        result: Result<(URL, String), Error>, plan: CapturePlan, attempt: Int
    ) async {
        do {
            let (temporaryURL, orientation) = try result.get()
            let captureID = String(format: "%@-a%02d", plan.id, attempt)
            let initialPath = plan.fields["video_path"] ?? ""
            guard !initialPath.isEmpty else { throw CaptureAppError.missingField("video_path") }
            let suffixPattern = try NSRegularExpression(pattern: "-a\\d{2}\\.mov$")
            let range = NSRange(initialPath.startIndex..<initialPath.endIndex, in: initialPath)
            let relativePath = suffixPattern.stringByReplacingMatches(
                in: initialPath, range: range, withTemplate: "-a\(String(format: "%02d", attempt)).mov"
            )
            let destination = try safeDocumentURL(relativePath: relativePath)
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            guard !FileManager.default.fileExists(atPath: destination.path) else {
                throw CaptureAppError.capture("Refusing to overwrite \(relativePath)")
            }
            try FileManager.default.moveItem(at: temporaryURL, to: destination)
            let metadata = try await videoMetadata(destination, orientation: orientation)
            let hash = try sha256(destination)
            var row = plan.fields
            row["capture_id"] = captureID
            row["attempt"] = String(attempt)
            row["performed_gloss"] = plan.isOOV
                ? oovDescription.trimmingCharacters(in: .whitespacesAndNewlines)
                : plan.expectedRawGloss
            row["prompt_hidden_before_capture"] = "true"
            row["video_path"] = relativePath
            row["video_sha256"] = hash
            row["recorded_utc"] = ISO8601DateFormatter().string(from: Date())
            row["device_model"] = deviceModel()
            row["ios_version"] = UIDevice.current.systemVersion
            row["camera"] = "front"
            row["width"] = String(metadata.width)
            row["height"] = String(metadata.height)
            row["fps"] = String(format: "%.6f", metadata.fps)
            row["orientation"] = orientation
            row["mirrored"] = "true"
            row["lighting"] = lighting.trimmingCharacters(in: .whitespacesAndNewlines)
            row["background"] = background.trimmingCharacters(in: .whitespacesAndNewlines)
            row["objective_qc_status"] = "pending"
            row["objective_qc_reason"] = ""
            try appendUpdate(row)
            attemptsByPlan[plan.id] = attempt
            capturedPlans.insert(plan.id)
            oovDescription = ""
            promptHidden = false
            statusMessage = "Saved \(captureID). Desktop QC must accept or reject it before inference."
            advance()
        } catch {
            promptHidden = false
            errorMessage = error.localizedDescription
        }
    }

    private func appendUpdate(_ row: [String: String]) throws {
        let url = documentsURL.appendingPathComponent("capture_updates.csv")
        if !FileManager.default.fileExists(atPath: url.path) {
            let header = Self.ledgerFields.joined(separator: ",") + "\n"
            try header.write(to: url, atomically: true, encoding: .utf8)
        }
        let handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(CSVCodec.line(fields: Self.ledgerFields, row: row).utf8))
        try handle.close()
    }

    private func loadExistingUpdates() throws {
        let url = documentsURL.appendingPathComponent("capture_updates.csv")
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        let rows = try CSVCodec.rows(from: String(contentsOf: url, encoding: .utf8))
        for row in rows {
            guard let plannedID = row["planned_id"], let attempt = Int(row["attempt"] ?? "") else {
                throw CaptureAppError.invalidCSV("capture update lacks planned_id/attempt")
            }
            attemptsByPlan[plannedID] = max(attemptsByPlan[plannedID] ?? 0, attempt)
            capturedPlans.insert(plannedID)
        }
    }

    private func safeDocumentURL(relativePath: String) throws -> URL {
        let relative = URL(fileURLWithPath: relativePath)
        guard !relativePath.hasPrefix("/"), !relativePath.split(separator: "/").contains("..") else {
            throw CaptureAppError.unsafePath(relativePath)
        }
        let result = documentsURL.appendingPathComponent(relativePath).standardizedFileURL
        guard result.path.hasPrefix(documentsURL.standardizedFileURL.path + "/") else {
            throw CaptureAppError.unsafePath(relativePath)
        }
        return result
    }

    private func videoMetadata(
        _ url: URL, orientation: String
    ) async throws -> RecordedVideoMetadata {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw CaptureAppError.capture("Recorded file has no video track")
        }
        let size = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let oriented = size.applying(transform)
        let fps = Double(try await track.load(.nominalFrameRate))
        let width = Int(round(abs(oriented.width)))
        let height = Int(round(abs(oriented.height)))
        guard width > 0, height > 0, fps > 0 else {
            throw CaptureAppError.capture("Recorded file is not valid video")
        }
        guard
            (orientation.hasPrefix("portrait") && height > width)
                || (orientation.hasPrefix("landscape") && width > height)
        else {
            throw CaptureAppError.capture(
                "Recorded dimensions disagree with the saved \(orientation) metadata"
            )
        }
        return RecordedVideoMetadata(width: width, height: height, fps: fps)
    }

    private func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty {
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func deviceModel() -> String {
        var info = utsname()
        uname(&info)
        let mirror = Mirror(reflecting: info.machine)
        return mirror.children.reduce(into: "") { value, element in
            guard let byte = element.value as? Int8, byte != 0 else { return }
            value.append(Character(UnicodeScalar(UInt8(byte))))
        }
    }

    private func selectFirstSession() {
        if let first = sessions.first { selectedSession = first }
    }

    private func selectFirstPending() {
        let values = sessionPlans
        currentIndex = values.firstIndex { !capturedPlans.contains($0.id) } ?? max(0, values.count - 1)
        promptHidden = false
        oovDescription = ""
    }

    private func advance() {
        let values = sessionPlans
        if let next = values.indices.dropFirst(currentIndex + 1).first(where: {
            !capturedPlans.contains(values[$0].id)
        }) {
            currentIndex = next
        }
    }

    private func sessionRank(_ value: String) -> Int {
        if value.hasSuffix("_r1") { return 1 }
        if value.hasSuffix("_r2") { return 2 }
        return 3
    }
}
