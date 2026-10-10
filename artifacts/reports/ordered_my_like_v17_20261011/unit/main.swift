import Foundation
let args=CommandLine.arguments
let list=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:args[1]))) as! [[String:Any]]
var results:[[String:Any]]=[]
for item in list {
 let rows=(item["rows"] as! [[String:Any]]).map{r in LiveMyLikeMotion.Row(t:r["t"] as! Double,slot:(r["slot"] as! String)=="l" ? 0 : 1,pinch:r["pinch"] as! Double,straight:r["straight"] as! Double,size:r["size"] as! Double,px:r["px"] as! Double,py:r["py"] as! Double,flatValid:r["flat_valid"] as! Bool)}
 let (d,r)=LiveMyLikeMotion.assess(rows,start:item["start"] as! Double,end:item["end"] as! Double,available:item["available"] as! Double,nextStart:item["next_start"] as? Double)
 precondition(d == item["expected"] as! String,"Rule disagreement: \(item["id"]!) \(d)")
 results.append(["id":item["id"]!,"decision":d,"reason":r])
}
try JSONSerialization.data(withJSONObject:["passed":true,"cases":results],options:[.sortedKeys]).write(to:URL(fileURLWithPath:args[2]))
print("Swift ordered-rule tests passed",results.count)
