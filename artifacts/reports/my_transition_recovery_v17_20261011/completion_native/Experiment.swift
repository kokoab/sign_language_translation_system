import Foundation
// Report-local ablation controls. No production source imports this file.
enum Experiment {
    static var pre = 0.1, post = 0.1
    static var gap=0.4, duration=0.6, tail=0.0
    static var audit = true
    static var events: [[String:Any]] = []
}
