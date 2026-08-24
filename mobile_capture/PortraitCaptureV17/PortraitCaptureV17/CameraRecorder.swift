import AVFoundation
import SwiftUI
import UIKit

@MainActor
final class CameraRecorder: NSObject, ObservableObject, AVCaptureFileOutputRecordingDelegate {
    let session = AVCaptureSession()
    @Published private(set) var isConfigured = false
    @Published private(set) var isRecording = false
    @Published var errorMessage: String?

    private let output = AVCaptureMovieFileOutput()
    private let queue = DispatchQueue(label: "slt.portrait.capture.camera")
    private var completion: ((Result<(URL, String), Error>) -> Void)?
    private var recordingOrientation = ""

    func requestAndConfigure() async {
        let allowed = await AVCaptureDevice.requestAccess(for: .video)
        guard allowed else {
            errorMessage = "Camera access is required. No recording was made."
            return
        }
        do {
            try await withCheckedThrowingContinuation { continuation in
                queue.async {
                    do {
                        try self.configureOnQueue()
                        continuation.resume()
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
            isConfigured = true
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func startRecording(
        to url: URL,
        completion: @escaping (Result<(URL, String), Error>) -> Void
    ) throws {
        guard isConfigured, session.isRunning, !output.isRecording else {
            throw CaptureAppError.capture("Camera is not ready or is already recording")
        }
        let orientation = try currentCaptureOrientation()
        self.completion = completion
        recordingOrientation = orientation.label
        if let connection = output.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(orientation.angle) {
                connection.videoRotationAngle = orientation.angle
            } else {
                throw CaptureAppError.capture("The camera cannot record the current orientation")
            }
            if connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = true
            }
        }
        output.startRecording(to: url, recordingDelegate: self)
        isRecording = true
    }

    func stopRecording() {
        guard output.isRecording else { return }
        output.stopRecording()
    }

    nonisolated func fileOutput(
        _ output: AVCaptureFileOutput,
        didFinishRecordingTo outputFileURL: URL,
        from connections: [AVCaptureConnection],
        error: Error?
    ) {
        Task { @MainActor in
            isRecording = false
            let callback = completion
            completion = nil
            if let error {
                callback?(.failure(error))
            } else {
                callback?(.success((outputFileURL, recordingOrientation)))
            }
        }
    }

    func currentOrientationFamily() -> String? {
        try? currentCaptureOrientation().family
    }

    private func currentCaptureOrientation() throws -> (label: String, family: String, angle: CGFloat) {
        let interfaceOrientation = UIApplication.shared.connectedScenes
            .compactMap { ($0 as? UIWindowScene)?.interfaceOrientation }
            .first { $0 != .unknown }
        switch interfaceOrientation {
        case .portrait:
            return ("portrait", "portrait", 90)
        case .portraitUpsideDown:
            return ("portrait_upside_down", "portrait", 270)
        case .landscapeLeft:
            return ("landscape_left", "landscape", 180)
        case .landscapeRight:
            return ("landscape_right", "landscape", 0)
        default:
            throw CaptureAppError.capture("Hold the phone upright in portrait or landscape")
        }
    }

    private nonisolated func configureOnQueue() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .hd1920x1080
        guard let camera = AVCaptureDevice.default(
            .builtInWideAngleCamera, for: .video, position: .front
        ) else {
            throw CaptureAppError.capture("Front camera is unavailable")
        }
        let input = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(input), session.canAddOutput(output) else {
            throw CaptureAppError.capture("Cannot configure front-camera recording")
        }
        session.addInput(input)
        session.addOutput(output)
        session.startRunning()
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.previewLayer.session = session
    }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}
