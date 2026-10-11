import Foundation
// Report-local ablation controls. No production source imports this file.
enum Experiment {
    static var pre = 0.1, post = 0.1
    static var dumpPath = ""
    static var annotations: [[String:Any]] = []
    static var dumped: [[String:Any]] = []
    static var dumpKeys: Set<String> = []
    static func save(_ x: LiveSpanInput, start:Double, end:Double) throws {
        guard let a=annotations.first(where: { start<=($0["lo"] as! Double) && end>=($0["hi"] as! Double) && start>=($0["lo"] as! Double)-0.3 && end<=($0["hi"] as! Double)+0.10 }) else { return }
        let key="\(start):\(end)";guard !dumpKeys.contains(key) else {return};dumpKeys.insert(key)
        let values=x.landmarks+x.embeddings+x.valid+x.boxes
        let h=try FileHandle(forWritingTo:URL(fileURLWithPath:dumpPath));try h.seekToEnd()
        let data=values.withUnsafeBufferPointer { Data(buffer:$0) };try h.write(contentsOf:data);try h.close()
        dumped.append(["index":dumped.count,"event":a["id"]!,"label":"MY","start":start,"end":end,"floats":values.count])
    }
    static var audit = true
    static var events: [[String:Any]] = []
}
