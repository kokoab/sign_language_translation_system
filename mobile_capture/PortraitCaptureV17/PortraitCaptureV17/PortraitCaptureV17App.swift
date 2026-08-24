import SwiftUI

@main
struct PortraitCaptureV17App: App {
    @StateObject private var store = CaptureStore()
    @StateObject private var camera = CameraRecorder()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(store)
                .environmentObject(camera)
                .task {
                    store.loadFrozenPack()
                    await camera.requestAndConfigure()
                }
        }
    }
}
