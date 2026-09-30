"""Download the ASLLRP Sign Bank utterance videos that contain fingerspelled words (fs-...).

Companion to fetch_asllrp_fingerspelled_v17.py (sign clips). Each utterance is written once to
data/local/asllrp_fingerspelled_v17/utterances/<file>; the manifest lists, per utterance, every
fingerspelled word with its span in seconds from the utterance start (frames at 29.97 fps, from the
Sign Bank sentence metadata). In-context evaluation material for spell mode. Sequential, resume-safe.
"""
import csv, glob, json, time, urllib.request
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/local/asllrp_fingerspelled_v17'
BASE = 'https://dai.cs.rutgers.edu/ss3front/'
FPS = 29.97
utterances = defaultdict(lambda: dict(words=[]))
for f in sorted(glob.glob(str(ROOT / 'data/local/dataset_metadata/asllrp_signbank/*sentence*.csv'))):
    with open(f, encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            k = {x.strip().lower(): v for x, v in r.items() if x}
            gloss, video = k.get('main entry gloss label', ''), k.get('utterance video filename', '')
            if not gloss.startswith('fs-') or not video:
                continue
            try:
                u0, a, b = (int(k[c]) for c in ('start frame of the containing utterance',
                                                 'start frame of the sign video', 'end frame of the sign video'))
            except (KeyError, ValueError):
                continue
            u = utterances[video]
            u.update(source=Path(f).name, collection=k.get('source collection', ''))
            u['words'].append(dict(gloss=gloss, start=round((a - u0) / FPS, 3), end=round((b - u0) / FPS, 3)))
(OUT / 'utterances').mkdir(parents=True, exist_ok=True)
ok = failed = 0
for video, u in sorted(utterances.items()):
    path = OUT / 'utterances' / video
    u['path'] = str(path.relative_to(ROOT))
    if path.exists() and path.stat().st_size > 0:
        ok += 1; continue
    try:
        req = urllib.request.Request(BASE + video, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            path.write_bytes(resp.read())
        ok += 1
    except Exception as e:
        u['error'] = str(e); failed += 1
    time.sleep(.2)
    if (ok + failed) % 50 == 0:
        print(ok, failed, len(utterances), flush=True)
(OUT / 'utterances_manifest.json').write_text(json.dumps(dict(source=BASE, fps=FPS, ok=ok, failed=failed,
    utterances=[dict(video=v, **u) for v, u in sorted(utterances.items())]), indent=1))
print('done', ok, failed, len(utterances), flush=True)
