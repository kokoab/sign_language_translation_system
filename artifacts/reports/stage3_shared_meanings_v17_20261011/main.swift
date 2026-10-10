import Foundation
import CoreML
let args=CommandLine.arguments
LiveModelLocator.packageDirectory=URL(fileURLWithPath:args[1])
let stage3=try LiveStage3(units:.cpuAndGPU)
let cases=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:args[2]))) as! [[String:Any]]
FileManager.default.createFile(atPath:args[3],contents:nil)
let output=try FileHandle(forWritingTo:URL(fileURLWithPath:args[3]))
let started=Date()
for (index,row) in cases.enumerated() {
    try autoreleasepool {
        let glosses=row["glosses"] as! [String], scores=row["scores"] as! [Double]
        let words=glosses.enumerated().map { i,g in LiveWord(gloss:g,startFrame:i*20,endFrame:i*20+10,commitFrame:i*20+12,startSeconds:Double(i),endSeconds:Double(i)+0.5,commitSeconds:Double(i)+0.6,score:scores[i],early:false) }
        let t=Date()
        let raw=stage3.renderSentence(glosses,scores)
        let full=stage3.renderUtterance(words).sentence
        var result:[String:Any]=["id":row["id"]!,"full":full,"raw":raw.sentence,"mode":raw.mode,"full_ms":1000*Date().timeIntervalSince(t)]
        if row["incremental"] as? Bool == true {
            let translator=LiveIncrementalTranslator()
            var prefixes:[[String:Any]]=[]
            for (i,w) in words.enumerated() {
                let lock=translator.append([w],using:stage3)
                var trace:[String:Any]=["n":i+1,"locked":translator.locked.map(\.text),"open":translator.buffer.map(\.gloss)]
                if let lock=lock {trace["new_lock"]=lock.text}
                prefixes.append(trace)
            }
            let finish=translator.finish([],using:stage3)
            result["incremental"]=finish.sentence; result["locked"]=finish.locked
            result["tail_words"]=finish.tailWords; result["prefixes"]=prefixes
        }
        var data=try JSONSerialization.data(withJSONObject:result,options:[.sortedKeys]);data.append(10)
        try output.write(contentsOf:data)
    }
    if (index+1)%25==0 {print("completed \(index+1)/\(cases.count), seconds \(Int(Date().timeIntervalSince(started)))");fflush(stdout)}
}
try output.close()
print("DONE \(cases.count) cases, seconds \(Date().timeIntervalSince(started))")
