from pathlib import Path
import shutil,subprocess,json
D=Path(__file__).resolve().parent;d=D/'scoped_native';shutil.copytree(D/'veto_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('    static var mix:Float=1','    static var pairFloor=0.0\n    static var mix:Float=1');p.write_text(s)
p=d/'LiveReelModels.swift';s=p.read_text().replace('    var lastAdapted:', '    var lastOriginal: [[Float]] = []\n    var lastAdapted:').replace('        lastAdapted=[]','        lastAdapted=[];lastOriginal=[]').replace('                    let adapted=chosen;','                    lastOriginal.append(base)\n                    let adapted=chosen;').replace('p[my]>q[my] {','p[my]>q[my], let like=labels.firstIndex(of:"LIKE"),q[my]+q[like]>=Experiment.pairFloor {');p.write_text(s)
p=d/'LiveReelDecoder.swift';s=p.read_text().replace('                guard ap[my]>q else {continue}','                guard ap[my]>q, let like=recognizer.labels.firstIndex(of:"LIKE"), q+exp(lp[like])>=Experiment.pairFloor else {continue}')
needle='        _ = tick("recognizer", t)';s=s.replace(needle,'''        if Experiment.audit {
            for (index,key) in keys.enumerated() where index<recognizer.lastAdapted.count && index<recognizer.lastOriginal.count {
                let p=LiveSegmental.logSoftmax(recognizer.lastAdapted[index]).map {exp($0)}
                let q=LiveSegmental.logSoftmax(recognizer.lastOriginal[index]).map {exp($0)}
                let my=recognizer.labels.firstIndex(of:"MY")!, like=recognizer.labels.firstIndex(of:"LIKE")!
                if p[my]>0.3 {
                    let order=q.indices.sorted {q[$0]>q[$1]}.prefix(4)
                    Experiment.events.append(["adaptation":true,"s":seconds(key.s),"e":seconds(key.e),"available":obs.last!.seconds,"adapted_my":p[my],"adapted_top":recognizer.labels[p.indices.max(by:{p[$0]<p[$1]})!],"original_pair":q[my]+q[like],"original_top":order.map {recognizer.labels[$0]},"original_p":order.map {q[$0]},"eligible":Experiment.eligible(obs,start:seconds(key.s),end:seconds(key.e))])
                }
            }
        }
'''+needle);p.write_text(s)
main=D/'scoped_validation';main.mkdir(exist_ok=True);shutil.copy(D/'confirm_validation/main.swift',main/'main.swift')
for p in [d/'main.swift',main/'main.swift']:
 s=p.read_text().replace('Experiment.events=[]','Experiment.events=[]\n    Experiment.pairFloor=item["pair_floor"] as? Double ?? 0.0').replace('Experiment.events = []','Experiment.events = []\n        Experiment.pairFloor=item["pair_floor"] as? Double ?? 0.0')
 if p.parent==main:s=s.replace('r["id"]=item["id"]','r["score_events"]=Experiment.events\n        r["id"]=item["id"]')
 p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
for kind,mainfile in [('video',d/'main.swift'),('validation',main/'main.swift')]:
 with (D/('scoped_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/('scoped_'+kind+'_probe'))]+files+[str(mainfile)],stdout=f,stderr=subprocess.STDOUT,check=True)
new=json.loads((D/'veto40_new_cases.json').read_text()) if (D/'veto40_new_cases.json').exists() else json.loads((D/'hybrid40_new_cases.json').read_text())
(D/'scoped_diagnostic_cases.json').write_text(json.dumps([dict(r,pair_floor=0) for r in new if r['phase']==1]))
base=json.loads((D/'focused_veto_cases.json').read_text()) if (D/'focused_veto_cases.json').exists() else [r for r in json.loads((D/'hybrid40_validation_cases.json').read_text()) if r['id']=='DRINK__7071450710791192-DRINK']
(D/'scoped_drink_cases.json').write_text(json.dumps([dict(r,pair_floor=v,id=f'p{v}__'+r['id']) for r in base for v in [0,.05,.1,.2,.4]]))
print('Compiled diagnostic and scoped pair evidence experiment')
