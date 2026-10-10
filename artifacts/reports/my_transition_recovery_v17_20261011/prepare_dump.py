from pathlib import Path
import shutil,json,subprocess
D=Path(__file__).resolve().parent;d=D/'dump_native';shutil.copytree(D/'native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('    static var audit', '''    static var dumpPath = ""
    static var annotations: [[String:Any]] = []
    static var dumped: [[String:Any]] = []
    static var dumpKeys: Set<String> = []
    static func save(_ x: LiveSpanInput, start:Double, end:Double) throws {
        guard let a=annotations.first(where: { start<=($0["lo"] as! Double) && end>=($0["hi"] as! Double) && start>=($0["lo"] as! Double)-0.3 && end<=($0["hi"] as! Double)+0.10 }) else { return }
        let key="\\(start):\\(end)";guard !dumpKeys.contains(key) else {return};dumpKeys.insert(key)
        let values=x.landmarks+x.embeddings+x.valid+x.boxes
        let h=try FileHandle(forWritingTo:URL(fileURLWithPath:dumpPath));try h.seekToEnd()
        let data=values.withUnsafeBufferPointer { Data(buffer:$0) };try h.write(contentsOf:data);try h.close()
        dumped.append(["index":dumped.count,"event":a["id"]!,"label":"MY","start":start,"end":end,"floats":values.count])
    }
    static var audit''');p.write_text(s)
p=d/'main.swift';s=p.read_text().replace('        Experiment.events = []','''        Experiment.events = []
        Experiment.dumped=[];Experiment.dumpKeys=[]
        Experiment.annotations=item["annotations"] as? [[String:Any]] ?? []
        Experiment.dumpPath=item["dump_path"] as! String
        FileManager.default.createFile(atPath:Experiment.dumpPath,contents:Data())''').replace('"score_events": Experiment.events,','"dumped":Experiment.dumped, "score_events": Experiment.events,');p.write_text(s)
p=d/'LiveReelDecoder.swift';s=p.read_text().replace('            batch.append(x); keys.append(key)','            try Experiment.save(x,start:seconds(key.s),end:seconds(key.e))\n            batch.append(x); keys.append(key)');p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
with (D/'dump_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/'dump_probe')]+files+[str(d/'main.swift')],stdout=f,stderr=subprocess.STDOUT,check=True)
N=D.parent/'sept30_fix_port_v17_20261011';base=json.loads((N/'new_video_cases.json').read_text());raw=D/'actual_windows';raw.mkdir(exist_ok=True)
cores=[('MY_FRIEND_1',1.02,1.13),('MY_FRIEND_2',4.76,4.85),('MY_NAME_1',7.32,7.45),('MY_NAME_2',9.61,9.72),('MY_FRIEND_3',14.92,15.02),('MY_EAT',16.73,16.83),('MY_FRIEND_4',19.28,19.36),('MY_NAME_3',21.20,21.30)]
ann=[dict(id=n,lo=lo,hi=hi) for n,lo,hi in cores]
(D/'dump_cases.json').write_text(json.dumps([dict(r,annotations=ann,dump_path=str(raw/(r['id']+'.bin'))) for r in base if r['phase'] in [0,4]],indent=2))
print('Built actual runtime window exporter')
