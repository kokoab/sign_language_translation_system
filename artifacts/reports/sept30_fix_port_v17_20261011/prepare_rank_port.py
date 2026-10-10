from pathlib import Path
import shutil,subprocess
N=Path(__file__).resolve().parent;d=N/'rank_native';shutil.copytree(N/'legacy_native',d,dirs_exist_ok=True)
p=d/'LiveReelDecoder.swift';s=p.read_text();start=s.index('        let mass = p[have] + p[angry]');end=s.index('\n    }',start)
s=s[:start]+"        var out = word; out.swapAt(have, angry); return out"+s[end:];p.write_text(s)
files=[str(d/n) for n in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelEngine.swift','LiveMyLikeMotion.swift']]
G=N.parent/'ordered_my_like_v17_20261011/gap175'
for kind,main in [('video',d/'main.swift'),('validation',G/'validation/main.swift')]:
 with (N/('rank_'+kind+'_build.log')).open('w') as f:subprocess.run(['xcrun','swiftc','-O','-o',str(N/('rank_'+kind+'_probe'))]+files+[str(main)],stdout=f,stderr=subprocess.STDOUT,check=True)
