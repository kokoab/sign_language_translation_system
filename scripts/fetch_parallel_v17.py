"""Download one large file over N parallel HTTP range requests, resumable per part, then join.

Used for ChicagoFSWild (dl.ttic.edu caps one connection at ~0.45 MB/s). Parts are <out>.partK; re-running
resumes each part from its current size. The joined file is checked against Content-Length.
"""
import argparse, os, sys, threading, time, urllib.request
from pathlib import Path


def size_of(url):
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers['Content-Length'])


def fetch(url, part, start, end):
    while True:
        have = part.stat().st_size if part.exists() else 0
        if start + have > end:
            return
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={start + have}-{end}'})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, 'ab') as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as e:
            print('part', part.name, 'retry after', repr(e)[:80], flush=True)
            time.sleep(20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('url'); ap.add_argument('out', type=Path); ap.add_argument('--parts', type=int, default=4)
    a = ap.parse_args()
    total = size_of(a.url)
    if a.out.exists() and a.out.stat().st_size == total:
        print('already complete'); return
    step = -(-total // a.parts)
    bounds = [(k * step, min(total, (k + 1) * step) - 1) for k in range(a.parts)]
    parts = [a.out.with_name(a.out.name + f'.part{k}') for k in range(a.parts)]
    threads = [threading.Thread(target=fetch, args=(a.url, p, s, e), daemon=True) for p, (s, e) in zip(parts, bounds)]
    for t in threads: t.start()
    while any(t.is_alive() for t in threads):
        done = sum(p.stat().st_size for p in parts if p.exists())
        print(f'{time.strftime("%H:%M:%S")} {done / 1e9:.2f} / {total / 1e9:.2f} GB', flush=True)
        time.sleep(300)
    with open(a.out, 'wb') as f:
        for p in parts:
            with open(p, 'rb') as src:
                while chunk := src.read(1 << 24):
                    f.write(chunk)
    if a.out.stat().st_size != total:
        sys.exit(f'size mismatch {a.out.stat().st_size} != {total}')
    for p in parts: p.unlink()
    print('complete', total, flush=True)


if __name__ == '__main__':
    main()
