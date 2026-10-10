from pathlib import Path
import shutil,json,subprocess
D=Path(__file__).resolve().parent;d=D/'positive_native';shutil.copytree(D/'mix_native',d,dirs_exist_ok=True)
p=d/'LiveReelModels.swift';s=p.read_text();old='''                    chosen=zip(base,chosen).map { a,b in a+Experiment.mix*(b-a) }''';new='''                    let adapted=chosen;chosen=base
                    if let my=labels.firstIndex(of:"MY"), repairEligible?[i+slot] == true {
                        let p=LiveSegmental.logSoftmax(adapted).map {exp($0)}
                        let q=LiveSegmental.logSoftmax(base).map {exp($0)}
                        if p.indices.max(by:{p[$0]<p[$1]})==my && p[my]>=Experiment.floor && p[my]>q[my] {
                            let target=min(p[my],0.999999)
                            chosen=q.enumerated().map { index,value in Float(log(max(1e-9,index==my ? target : value*(1-target)/max(1e-9,1-q[my])))) }
                        }
                    }''';assert old in s;s=s.replace(old,new);p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
with (D/'positive_build.log').open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/'positive_probe')]+files+[str(d/'main.swift')],stdout=f,stderr=subprocess.STDOUT,check=True)
base=json.loads((D/'new_video_cases.json').read_text());g=json.loads((D/'genuine_cases.json').read_text());floors=[.6,.75,.9,.95]
for kind,rr in [('screen',[r for r in base if r['phase'] in [0,4]]),('genuine',g)]:
 (D/('positive_'+kind+'_cases.json')).write_text(json.dumps([dict(r,floor=f,max_duration=100,flat=False,id=f'floor{f}__'+r['id']) for r in rr for f in floors]))
print('Built four MY-only positive correction confidence arms')
