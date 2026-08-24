import SwiftUI

@main
struct OrientationBenchmarkV17App: App {
    @StateObject private var runner = BenchmarkRunner()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(runner)
                .task {
                    runner.loadResources()
                    await runner.runAutomationIfRequested()
                }
        }
    }
}
