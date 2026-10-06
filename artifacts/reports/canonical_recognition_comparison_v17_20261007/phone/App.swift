import UIKit
@main final class AppDelegate: UIResponder, UIApplicationDelegate {
  var window: UIWindow?
  func application(_ application: UIApplication, didFinishLaunchingWithOptions options: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
    let window = UIWindow(frame: UIScreen.main.bounds)
    let controller = UIViewController()
    controller.view.backgroundColor = .systemBackground
    let label = UILabel(frame: CGRect(x: 24, y: 120, width: 340, height: 100))
    label.text = "ATLAS classifier benchmark"; label.textAlignment = .center
    controller.view.addSubview(label)
    window.rootViewController = controller; window.makeKeyAndVisible(); self.window = window
    return true
  }
}
