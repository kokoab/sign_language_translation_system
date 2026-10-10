from pathlib import Path
import shutil,subprocess,json
D=Path(__file__).resolve().parent;d=D/'preserved_native';shutil.copytree(D/'ambiguous_native',d,dirs_exist_ok=True)
p=d/'Experiment.swift';s=p.read_text();a=s.index('        if top==my');b=s.index('        let mass=',a)
s=s[:a]+'''        let order=q.indices.sorted {q[$0]>q[$1]}
        // Already-MY spans need either MY/LIKE ambiguity or strong independent MY confirmation.
        guard top != my || order.prefix(3).contains(like) || p[my]>=0.9 else{return nil}
        let target=min(max(p[my],top==like ? q[like] : 0),0.999999)
        guard target>q[my] else{return nil}
        let lp=LiveSegmental.logSoftmax(base)
'''+s[b:];p.write_text(s)
validation=D/'preserved_validation';validation.mkdir(exist_ok=True);shutil.copy(D/'ambiguous_validation/main.swift',validation/'main.swift')
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift','Experiment.swift']]
for kind,main in [('video',d/'main.swift'),('validation',validation/'main.swift')]:
 with (D/('preserved_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(D/('preserved_'+kind+'_probe'))]+files+[str(main)],stdout=f,stderr=subprocess.STDOUT,check=True)
for kind in ['new','genuine','validation','phase','splice','drink']:
 base=json.load(open(D/f'ambiguous_{kind}_cases.json'));(D/f'preserved_{kind}_cases.json').write_text(json.dumps(base))
print('Compiled confidence-preserving one-way experiment')
