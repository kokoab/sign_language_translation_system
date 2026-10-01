"""Select, download and extract an FSboard batch, streaming: download -> extract -> delete the video.

Selection (fixed seed): non-sensitive clips whose non-space characters are mostly letters (>= 70%,
>= 4 letters); train = up to --per-signer clips per train signer, alternating three strata when the
signer has them: person names and addresses/URLs (daun_v3) and English sentences (dmk_v3); validation = --per-val-signer clips
per daun_v3 validation signer. The official FSboard test split is never selected. Pilot clips are
excluded (their features already exist). Videos are MD5-checked, extracted with the ffmpeg decoder
(scripts/extract_fsboard_v17.py) and deleted; at most --buffer videos are on disk at once.
Re-running resumes: clips whose .npz exists are skipped.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import queue
import random
import re
import sys
import threading
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'data/local/fsboard_v17'
URL = 'https://www.kaggle.com/api/v1/datasets/download/googleai/fsboard/'


def usable(r):
    chars = re.sub(r'\s', '', r['phrase'])
    letters = len(re.findall(r'[A-Za-z]', chars))
    return not r.get('sensitiveContent', False) and letters >= 4 and letters >= .7 * max(1, len(chars))


def kind(r):
    """Stratum: person names ('first last', letters only), other daun_v3 (addresses/URLs), dmk_v3 sentences."""
    if r['collectionId'] == 'dpan-mackenzie':
        return 'sentence'
    return 'name' if re.fullmatch(r'[a-z]+ [a-z]+', r['phrase']) else 'address_url'


def take_alternating(groups, n, rng):
    groups = [sorted(g, key=lambda r: r['clipFilename']) for g in groups]
    for g in groups:
        rng.shuffle(g)
    take = []
    while len(take) < n and any(groups):                     # alternate strata
        for g in groups:
            if g and len(take) < n:
                take.append(g.pop())
    return take


def select(per_signer, per_val_signer, seed=0):
    rng = random.Random(seed)
    pilot = {r['clipFilename'] for r in json.loads((DATA / 'pilot/manifest.json').read_text())['clips']}
    meta = lambda name: json.loads((DATA / f'metadata/{name}.json').read_text())
    picked = []
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for subset in ('daun_v3', 'dmk_v3'):
        for r in meta(f'{subset}-train'):
            if usable(r) and r['clipFilename'] not in pilot:
                by[r['signerId']][kind(r)].append(r)
    for signer in sorted(by):
        take = take_alternating([by[signer][k] for k in sorted(by[signer])], per_signer, rng)
        picked += [dict(r, split='train', subset=sub_of(r), kind=kind(r)) for r in take]
    val = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in meta('daun_v3-validation'):
        if usable(r):
            val[r['signerId']][kind(r)].append(r)
    for signer in sorted(val):
        take = take_alternating([val[signer][k] for k in sorted(val[signer])], per_val_signer, rng)
        picked += [dict(r, split='validation', subset='daun_v3', kind=kind(r)) for r in take]
    return picked


def sub_of(r):
    return 'dmk_v3' if r['collectionId'] == 'dpan-mackenzie' else 'daun_v3'


def remote_path(r):
    return f"{r['subset']}/video_clips/{r['subset']}-{r['split']}/{r['clipFilename']}"


def download(r, folder, tries=4):
    target = folder / r['clipFilename']
    if target.exists() and hashlib.md5(target.read_bytes()).hexdigest() == r['clipFileMd5']:
        return target                                         # left over from an interrupted run
    for attempt in range(tries):
        try:
            urllib.request.urlretrieve(URL + urllib.parse.quote(remote_path(r), safe=''), target)
            if hashlib.md5(target.read_bytes()).hexdigest() == r['clipFileMd5']:
                return target
        except Exception as e:                                # network hiccup: retry with backoff
            print('download error', r['clipFilename'], repr(e), flush=True)
        target.unlink(missing_ok=True)
        time.sleep(5 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='batch1')
    ap.add_argument('--per-signer', type=int, default=25)
    ap.add_argument('--per-val-signer', type=int, default=50)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--buffer', type=int, default=24)
    ap.add_argument('--select-only', action='store_true')
    a = ap.parse_args()
    out = DATA / a.name
    manifest = out / 'manifest.json'
    if manifest.exists():
        clips = json.loads(manifest.read_text())['clips']
    else:
        clips = select(a.per_signer, a.per_val_signer)
        out.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(dict(
            source='https://www.kaggle.com/datasets/googleai/fsboard (CC BY 4.0)', per_signer=a.per_signer,
            per_val_signer=a.per_val_signer, seed=0, clips=clips), indent=1))
    count = collections.Counter((r['split'], r['kind']) for r in clips)
    print('selected', len(clips), dict(count), 'train signers', len({r['signerId'] for r in clips if r['split'] == 'train'}),
          'GB', round(sum(r['clipFileSize'] for r in clips) / 1e9, 1), flush=True)
    if a.select_only:
        return
    from scripts.extract_fsboard_v17 import Extractor
    videos = out / 'videos_tmp'
    videos.mkdir(exist_ok=True)
    todo = [r for r in clips if not (out / 'features' / r['split'] / (Path(r['clipFilename']).stem + '.npz')).exists()]
    ready, slots = queue.Queue(), threading.Semaphore(a.buffer)
    work = queue.Queue()
    for r in todo:
        work.put(r)

    def fetcher():
        while True:
            try:
                r = work.get_nowait()
            except queue.Empty:
                return
            slots.acquire()
            ready.put((r, download(r, videos)))

    threads = [threading.Thread(target=fetcher, daemon=True) for _ in range(a.workers)]
    for t in threads:
        t.start()
    ex = Extractor('ffmpeg', letters=False)
    started, failed, done = time.time(), [], 0
    for n in range(len(todo)):
        r, path = ready.get()
        if path is None:
            failed.append(r['clipFilename']); slots.release()
            continue
        folder = out / 'features' / r['split']
        folder.mkdir(parents=True, exist_ok=True)
        try:
            ex.clip(path, r, folder / (Path(r['clipFilename']).stem + '.npz'))
            done += 1
        except Exception as e:
            print('extract error', r['clipFilename'], repr(e), flush=True)
            failed.append(r['clipFilename'])
        path.unlink(missing_ok=True)
        slots.release()
        if done % 25 == 0:
            rate = done / (time.time() - started)
            print(f'{done}/{len(todo)} done, {len(failed)} failed, {rate * 3600:.0f} clips/h, '
                  f'eta {(len(todo) - done) / max(rate, 1e-6) / 3600:.1f} h', flush=True)
    (out / 'failed.json').write_text(json.dumps(failed, indent=1))
    print(f'finished: {done} extracted, {len(failed)} failed, {(time.time() - started) / 3600:.2f} h', flush=True)


if __name__ == '__main__':
    main()
