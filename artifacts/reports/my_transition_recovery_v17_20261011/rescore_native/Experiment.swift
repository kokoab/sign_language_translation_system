import Foundation
// Report-local ablation controls. No production source imports this file.
enum Experiment {
    static var pre = 0.1, post = 0.1
    static var policy = "none"
    static var altPre = 0.1
    static var expert: LiveSpanRecognizer?
    static var audit = true
    static var events: [[String:Any]] = []
}
