import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    GeneratedPluginRegistrant.register(with: self)
    // Build the Reel models behind the home page (desktop shell warm()).
    LiveReelShared.warm()
    guard let controller = window?.rootViewController as? FlutterViewController else {
      return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }
    let channel = FlutterMethodChannel(
      name: SLTInferenceBridge.channelName,
      binaryMessenger: controller.binaryMessenger
    )
    channel.setMethodCallHandler { [weak controller] call, result in
      let arguments = call.arguments as? [String: Any]
      switch call.method {
      case "getBenchmarkEnabled":
        result(UserDefaults.standard.bool(
          forKey: SLTInferenceBridge.benchmarkDefaultsKey
        ))
      case "setBenchmarkEnabled":
        guard let enabled = arguments?["enabled"] as? Bool else {
          result(FlutterError(
            code: "invalid_arguments",
            message: "Missing benchmark setting",
            details: nil
          ))
          return
        }
        UserDefaults.standard.set(
          enabled,
          forKey: SLTInferenceBridge.benchmarkDefaultsKey
        )
        result(nil)
      case "inferVideo":
        guard let path = arguments?["path"] as? String,
              let mirrored = arguments?["mirrored"] as? Bool,
              let benchmarkEnabled = arguments?["benchmarkEnabled"] as? Bool else {
          result(FlutterError(
            code: "invalid_arguments",
            message: "Missing video inference arguments",
            details: nil
          ))
          return
        }
        Task {
          do {
            let value = try await Task.detached(priority: .userInitiated) {
              try await SLTInferenceBridge.inferVideo(
                path: path,
                mirrored: mirrored,
                benchmarkEnabled: benchmarkEnabled
              )
            }.value
            await MainActor.run { result(value) }
          } catch {
            await MainActor.run {
              result(FlutterError(
                code: "inference_failed",
                message: error.localizedDescription,
                details: nil
              ))
            }
          }
        }
      case "openLiveReel", "openPractice":
        guard let controller else {
          result(FlutterError(code: "unavailable", message: "No root controller", details: nil))
          return
        }
        let mode: LiveReelViewController.Mode = call.method == "openPractice"
          ? .practice(arguments?["glosses"] as? [String] ?? [])
          : .live
        let live = LiveReelViewController(mode: mode)
        live.onClose = { value in result(value) }
        (controller.presentedViewController ?? controller).present(live, animated: true)
      case "liveReelStatus":
        result(LiveReelShared.status)
      case "getLiveSettings":
        result([
          "nameSpelling": UserDefaults.standard.bool(forKey: LiveReelShared.nameSpellingKey),
          "saveVideos": UserDefaults.standard.bool(forKey: LiveReelShared.saveVideosKey),
        ])
      case "setLiveSetting":
        let keys = ["nameSpelling": LiveReelShared.nameSpellingKey, "saveVideos": LiveReelShared.saveVideosKey]
        guard let key = (arguments?["key"] as? String).flatMap({ keys[$0] }),
              let value = arguments?["value"] as? Bool else {
          result(FlutterError(code: "invalid_arguments", message: "Unknown Live setting", details: nil))
          return
        }
        UserDefaults.standard.set(value, forKey: key)
        result(nil)
      case "resetLiveTips":
        UserDefaults.standard.removeObject(forKey: LiveReelShared.liveTipsSeenKey)
        result(nil)
      case "speak":
        guard let text = arguments?["text"] as? String, !text.isEmpty else {
          result(FlutterError(code: "invalid_arguments", message: "Nothing to speak", details: nil))
          return
        }
        LiveReelShared.speak(text)
        result(nil)
      case "deleteSession", "clearSessions":
        let stamp = arguments?["stamp"] as? String
        LiveReelShared.languageQueue.async {
          if call.method == "clearSessions" { LiveReelSessions.deleteAll() }
          else if let stamp { LiveReelSessions.delete(stamp: stamp) }
          DispatchQueue.main.async { result(nil) }
        }
      case "listSessions":
        LiveReelShared.languageQueue.async {
          let rows = LiveReelSessions.list()
          DispatchQueue.main.async { result(rows) }
        }
      case "shareReport":
        guard let path = arguments?["path"] as? String,
              let controller else {
          result(FlutterError(
            code: "invalid_arguments",
            message: "Missing report path",
            details: nil
          ))
          return
        }
        do {
          let url = try SLTInferenceBridge.validatedReportURL(path: path)
          let activity = UIActivityViewController(
            activityItems: [url],
            applicationActivities: nil
          )
          activity.popoverPresentationController?.sourceView = controller.view
          (controller.presentedViewController ?? controller).present(
            activity,
            animated: true
          )
          result(nil)
        } catch {
          result(FlutterError(
            code: "report_failed",
            message: error.localizedDescription,
            details: nil
          ))
        }
      default:
        result(FlutterMethodNotImplemented)
      }
    }
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
}
