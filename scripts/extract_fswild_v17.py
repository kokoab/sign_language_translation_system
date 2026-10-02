"""Extract ChicagoFSWild (natural, in-the-wild fingerspelling) through the live Apple Vision contract.

Each sequence is a folder of JPEG frames (rate not stored; treated as 30 fps). Output per sequence:
data/local/chicago_fswild/features/<partition>/<source>__<sequence>.npz, same fields as FSboard
(scripts/extract_fsboard_v17.py); the annotation is the whole sequence. The official signer-disjoint
partitions are kept (train 87 / dev 37 / test 36 signers). The test partition is FINAL-TEST ONLY.
"""
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
D = ROOT / 'data/local/chicago_fswild'
SRC = D / 'ChicagoFSWild'


def manifest():
    clips = []
    for r in csv.DictReader(open(SRC / 'ChicagoFSWild.csv')):
        n = int(r['number_of_frames'])
        clips.append(dict(clipFilename=r['filename'], phrase=r['label_proc'], signerId='fswild_' + r['signer'],
                          split=r['partition'], url=r['url'], width=int(r['width']), height=int(r['height']),
                          clipStartTimeS=0., annotationStartTimeS=0., annotationEndTimeS=n / 30.))
    return clips


def main():
    from scripts.extract_fsboard_v17 import Extractor
    clips = manifest()
    (D / 'manifest.json').write_text(json.dumps(dict(
        source='https://home.ttic.edu/~klivescu/ChicagoFSWild.htm (Shi et al., SLT 2018)',
        note='official signer-disjoint partitions; test is FINAL-TEST ONLY', clips=clips), indent=1))
    ex = Extractor('ffmpeg', letters=False)
    started, done, failed = time.time(), 0, []
    for c in clips:
        out = D / 'features' / c['split']
        out.mkdir(parents=True, exist_ok=True)
        target = out / (c['clipFilename'].replace('/', '__') + '.npz')
        if target.exists():
            continue
        folder = SRC / c['clipFilename']
        try:
            ex.clip(folder, c, target)
            done += 1
        except Exception as e:
            failed.append((c['clipFilename'], repr(e)[:120]))
        if done and done % 250 == 0:
            print(f'{done} done, {len(failed)} failed, {(time.time() - started) / 60:.1f} min', flush=True)
    (D / 'extract_failed.json').write_text(json.dumps(failed, indent=1))
    print(f'finished: {done} extracted, {len(failed)} failed, {(time.time() - started) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
