import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var runner: BenchmarkRunner
    @State private var importing = false

    var body: some View {
        NavigationStack {
            Form {
#if targetEnvironment(simulator)
                Section("Simulator evidence only") {
                    Text("This run validates the iOS, Vision, and Core ML integration on Mac hardware. Its latency, memory, thermal state, and compute units are not physical-iPhone measurements.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
#endif
                Section("Video") {
                    Button("Choose video") { importing = true }
                    Text(runner.selectedURL?.lastPathComponent ?? "No video selected")
                        .font(.footnote)
                    Toggle("Pixels are mirrored", isOn: $runner.inputMirrored)
                    Text("Container metadata is applied first. A face/body probe then selects the nearest upright quadrant; the trained model handles the remaining continuous roll without changing aspect ratio.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Section("Ground truth") {
                    Picker("Expected label", selection: $runner.expectedLabel) {
                        ForEach(runner.labels, id: \.self) { Text($0).tag($0) }
                    }
                }
                Section("Sustained run") {
                    Stepper("\(runner.iterations) iterations", value: $runner.iterations, in: 20...1_000, step: 20)
                    Button(runner.isRunning ? "Running…" : "Extract and benchmark") {
                        runner.run()
                    }
                    .disabled(runner.isRunning || runner.selectedURL == nil)
                    if runner.isRunning { ProgressView() }
                    Text(runner.status).font(.footnote).foregroundStyle(.secondary)
                }
                if let report = runner.report {
                    Section("Result") {
                        LabeledContent("Prediction", value: report.predictedLabel)
                        LabeledContent("English", value: report.stage3NaturalEnglish)
                        LabeledContent("Literal", value: report.stage3LiteralEnglish)
                        LabeledContent("Stage 3 mode", value: report.stage3RenderingMode)
                        LabeledContent("Correct", value: report.correct ? "yes" : "no")
                        LabeledContent("Extraction", value: String(format: "%.1f ms", report.diagnostics.extractionMilliseconds))
                        LabeledContent("Vision correction", value: "\(report.diagnostics.visionCoarseRotationClockwise)°")
                        LabeledContent("Core ML median", value: String(format: "%.3f ms", report.medianInferenceMilliseconds))
                        LabeledContent("Core ML p90", value: String(format: "%.3f ms", report.p90InferenceMilliseconds))
                        LabeledContent("Thermal", value: "\(report.thermalBefore) → \(report.thermalAfter)")
                        LabeledContent("Resident memory", value: String(format: "%.1f MiB", Double(report.residentMemoryAfterBytes) / 1_048_576))
                        LabeledContent("Memory delta", value: String(format: "%.1f MiB", Double(report.residentMemoryAfterBytes - min(report.residentMemoryBeforeBytes, report.residentMemoryAfterBytes)) / 1_048_576))
                        LabeledContent("Compiled model", value: String(format: "%.1f MiB", Double(report.modelBytes) / 1_048_576))
                        if let reportURL = runner.reportURL {
                            ShareLink(item: reportURL) { Label("Export JSON", systemImage: "square.and.arrow.up") }
                        }
                    }
                }
            }
            .navigationTitle("Orientation v17 Benchmark")
            .fileImporter(
                isPresented: $importing,
                allowedContentTypes: [.movie],
                allowsMultipleSelection: false
            ) { result in
                do { runner.selectedURL = try result.get().first }
                catch { runner.errorMessage = error.localizedDescription }
            }
            .alert("Benchmark blocked", isPresented: Binding(
                get: { runner.errorMessage != nil },
                set: { if !$0 { runner.errorMessage = nil } }
            )) { Button("OK", role: .cancel) {} } message: {
                Text(runner.errorMessage ?? "Unknown error")
            }
        }
    }
}
