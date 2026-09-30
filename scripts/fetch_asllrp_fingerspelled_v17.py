"""Download ASLLRP Sign Bank clips of fingerspelled words (fs-...) signed inside sentences.

Source: the local Sign Bank sentence metadata (data/local/dataset_metadata/asllrp_signbank/*sentence*.csv);
each fingerspelled token has its own sign clip at https://dai.cs.rutgers.edu/ss3front/<file>.
Evaluation material for spell mode (other signers, natural sentences). Sequential, resume-safe.
"""
import csv, glob, json, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/local/asllrp_fingerspelled_v17'
BASE = 'https://dai.cs.rutgers.edu/ss3front/'
rows = []
for f in sorted(glob.glob(str(ROOT / 'data/local/dataset_metadata/asllrp_signbank/*sentence*.csv'))):
    with open(f, encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            k = {x.strip().lower(): v for x, v in r.items() if x}
            if k.get('main entry gloss label', '').startswith('fs-'):
                rows.append(dict(source=Path(f).name, gloss=k['main entry gloss label'], sign_video=k['sign video filename'],
                                 utterance_video=k.get('utterance video filename', ''),
                                 collection=k.get('source collection', ''),
                                 sign_start=k.get('start frame of the sign video', ''), sign_end=k.get('end frame of the sign video', ''),
                                 utterance_start=k.get('start frame of the containing utterance', ''),
                                 utterance_end=k.get('end frame of the containing utterance', '')))
(OUT / 'clips').mkdir(parents=True, exist_ok=True)
ok = failed = 0
for r in rows:
    path = OUT / 'clips' / r['sign_video']
    r['path'] = str(path.relative_to(ROOT))
    if path.exists() and path.stat().st_size > 0:
        ok += 1; continue
    try:
        req = urllib.request.Request(BASE + r['sign_video'], headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        path.write_bytes(data); ok += 1
    except Exception as e:
        r['error'] = str(e); failed += 1
    time.sleep(.2)
    if (ok + failed) % 100 == 0:
        print(ok, failed, len(rows), flush=True)
(OUT / 'manifest.json').write_text(json.dumps(dict(source=BASE, rows=rows, downloaded=ok, failed=failed), indent=1))
print('done', ok, failed, len(rows))
