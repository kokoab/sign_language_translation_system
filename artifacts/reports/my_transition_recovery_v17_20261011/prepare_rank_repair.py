from pathlib import Path
import shutil,json,subprocess
D=Path(__file__).resolve().parent;d=D/'rank_native';shutil.copytree(D/'scoped_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text().replace('    static var policy="pair"','    static var policy="swap"\n    static func refine(_ base:[Float], adapted:[Float], labels:[String])->[Float]? {\n        guard let my=labels.firstIndex(of:"MY"),let like=labels.firstIndex(of:"LIKE") else {return nil}\n        let q=LiveSegmental.logSoftmax(base).map {exp($0)},p=LiveSegmental.logSoftmax(adapted).map {exp($0)}\n        let top=q.indices.max(by:{q[$0]<q[$1]})!\n        guard top != my, p.indices.max(by:{p[$0]<p[$1]})==my,p[my]>=floor,q[my]+q[like]>=pairFloor else{return nil}\n        if policy == "swap" || (policy == "hybrid" && top == like) {\n            guard top == like else{return nil}\n            var row=base;row.swapAt(my,like);return row\n        }\n        let target=min(p[my],0.999999),lp=LiveSegmental.logSoftmax(base)\n        let mass=Double(base[my])-lp[my],adjustment=log((1-target)/max(1-q[my],1e-9))\n        return base.enumerated().map {index,value in index==my ? Float(mass+log(target)) : value+Float(adjustment)}\n    }')
p.write_text(s)
p=d/'LiveReelModels.swift';s=p.read_text();a=s.index('                    if let my=labels.firstIndex(of:"MY"),repairEligible?');b=s.index('\n                }\n                out.append',a)
s=s[:a]+'''                    if repairEligible?[i+slot] == true, let refined=Experiment.refine(base,adapted:adapted,labels:labels) {chosen=refined}
'''+s[b:];p.write_text(s)
p=d/'LiveReelDecoder.swift';s=p.read_text();a=s.index('                let word=Array(row.prefix(wordCount));');b=s.index('                local[key]=row;refined=true',a)
s=s[:a]+'''                let word=Array(row.prefix(wordCount))
                guard let updated=Experiment.refine(word,adapted:adapted,labels:recognizer.labels) else {continue}
                row=updated+Array(row.dropFirst(wordCount))
'''+s[b:];p.write_text(s)
validation=D/'rank_validation';validation.mkdir(exist_ok=True);shutil.copy(D/'scoped_validation/main.swift',validation/'main.swift')
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
for kind,main in [('video',d/'main.swift'),('validation',validation/'main.swift')]:
 with (D/('rank_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/('rank_'+kind+'_probe'))]+files+[str(main)],stdout=f,stderr=subprocess.STDOUT,check=True)
base=json.loads((D/'hybrid40_new_cases.json').read_text());arms=[dict(name=f'{p}_{f}',policy=p,floor=f) for p in ['swap','hybrid','boost'] for f in [.5,.6]]
(D/'rank_arms.json').write_text(json.dumps(arms,indent=2));(D/'rank_screen_cases.json').write_text(json.dumps([dict(r,**{k:v for k,v in a.items() if k!='name'},id=a['name']+'__'+r['id']) for r in base if r['phase'] in [1,4] for a in arms]))
drink=json.loads((D/'scoped_drink_cases.json').read_text())[0];(D/'rank_drink_cases.json').write_text(json.dumps([dict(drink,**{k:v for k,v in a.items() if k!='name'},id=a['name']+'__DRINK') for a in arms]))
print('Compiled six confidence-preserving correction arms')
