import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var store: CaptureStore
    @EnvironmentObject private var camera: CameraRecorder
    @Environment(\.openURL) private var openURL

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                selectors
                progress
                ZStack {
                    CameraPreview(session: camera.session)
                        .clipShape(RoundedRectangle(cornerRadius: 18))
                    if !store.promptHidden {
                        promptCard
                    } else if let count = store.countdown {
                        Text("\(count)")
                            .font(.system(size: 96, weight: .bold, design: .rounded))
                            .foregroundStyle(.white)
                            .shadow(radius: 8)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(.black)
                .clipShape(RoundedRectangle(cornerRadius: 18))

                metadataFields
                controls
                Text(store.statusMessage)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding()
            .navigationTitle("Portrait v17 Capture")
            .navigationBarTitleDisplayMode(.inline)
            .alert("Capture blocked", isPresented: Binding(
                get: { store.errorMessage != nil || camera.errorMessage != nil },
                set: { if !$0 { store.errorMessage = nil; camera.errorMessage = nil } }
            )) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(store.errorMessage ?? camera.errorMessage ?? "Unknown error")
            }
        }
    }

    private var selectors: some View {
        HStack {
            Picker("Signer", selection: $store.selectedSigner) {
                ForEach(store.signers, id: \.self) { Text($0).tag($0) }
            }
            Picker("Session", selection: $store.selectedSession) {
                ForEach(store.sessions, id: \.self) { Text($0).tag($0) }
            }
        }
        .pickerStyle(.menu)
        .disabled(camera.isRecording || store.countdown != nil)
    }

    private var progress: some View {
        HStack {
            Text("\(store.completedInSession)/\(store.sessionPlans.count) captured")
            Spacer()
            if let plan = store.currentPlan {
                Text("Prompt \(plan.promptOrder)")
            }
        }
        .font(.subheadline.monospacedDigit())
    }

    @ViewBuilder
    private var promptCard: some View {
        if let plan = store.currentPlan {
            VStack(spacing: 10) {
                Text(plan.label)
                    .font(.system(size: 42, weight: .bold, design: .rounded))
                if !plan.expectedRawGloss.isEmpty {
                    Text("Pinned variant: \(plan.expectedRawGloss) · \(plan.aslLexCode)")
                        .font(.callout.monospaced())
                }
                if let referenceURL = plan.referenceURL {
                    Button("Open approved reference") { openURL(referenceURL) }
                        .buttonStyle(.bordered)
                }
                Text("The prompt will disappear for three seconds before recording starts.")
                    .font(.footnote)
                    .multilineTextAlignment(.center)
                Text("Any phone orientation is allowed. Native aspect ratio is preserved.")
                    .font(.footnote.bold())
                    .multilineTextAlignment(.center)
            }
            .padding(24)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
            .padding()
        } else {
            Text("No plan loaded")
                .foregroundStyle(.white)
        }
    }

    private var metadataFields: some View {
        VStack(spacing: 8) {
            HStack {
                TextField("Lighting", text: $store.lighting)
                TextField("Background", text: $store.background)
            }
            .textFieldStyle(.roundedBorder)
            if store.currentPlan?.isOOV == true {
                TextField("Describe the non-target sign or gesture", text: $store.oovDescription)
                    .textFieldStyle(.roundedBorder)
            }
        }
        .disabled(camera.isRecording || store.countdown != nil)
    }

    private var controls: some View {
        HStack(spacing: 16) {
            if camera.isRecording {
                Button(role: .destructive) { store.stop(camera: camera) } label: {
                    Label("Stop & save", systemImage: "stop.circle.fill")
                }
                .buttonStyle(.borderedProminent)
            } else {
                Button { store.beginCountdown(camera: camera) } label: {
                    Label("Hide prompt & record", systemImage: "record.circle")
                }
                .buttonStyle(.borderedProminent)
                .disabled(!camera.isConfigured || store.countdown != nil || store.currentPlan == nil)
                Button("New attempt") { store.captureAnotherAttempt() }
                    .buttonStyle(.bordered)
                    .disabled(store.currentPlan == nil)
            }
        }
    }
}
