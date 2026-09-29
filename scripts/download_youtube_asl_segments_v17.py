"""Download one bounded video-only segment per YouTube-ASL channel (official ID list).

Used only as unlabeled continuous signing for boundary distillation (teacher targets need
no human labels). Channels discovered by acquire_youtube_asl_transition_voices_v17.py are
reused from its state file; one video per channel, first success wins.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT / 'data/local/youtube_asl_boundary_distill_v17')
    ap.add_argument('--target', type=int, default=300)
    ap.add_argument('--start', type=float, default=30.)
    ap.add_argument('--end', type=float, default=60.)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--yt-dlp', type=Path, default=ROOT / 'data/local/tools/yt-dlp_macos')
    args = ap.parse_args()
    state = json.loads((args.root / 'acquisition_state.json').read_text())
    by_channel = {}
    for vid, row in sorted(state['attempts'].items()):
        channel = row.get('channel_id') or row.get('channel') or vid
        by_channel.setdefault(channel, []).append(vid)
    clips = args.root / 'clips'
    clips.mkdir(exist_ok=True)
    status_path = args.root / 'segments_status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    lock = threading.Lock()
    done_channels = {v['channel'] for v in status.values() if v['status'] == 'ok'}

    def fetch(channel, vids):
        for vid in vids[:3]:
            out = clips / f'{vid}.mp4'
            if out.exists() and out.stat().st_size > 10000:
                return channel, vid, 'ok'
            cmd = [str(args.yt_dlp), '-q', '--no-warnings', '-f', 'bv*[height<=480][ext=mp4]/bv*[height<=480]',
                   '--download-sections', f'*{args.start:g}-{args.end:g}', '-o', str(clips / f'{vid}.%(ext)s'),
                   '--', vid]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
            except subprocess.TimeoutExpired:
                continue
            got = sorted(clips.glob(f'{vid}.*'))
            if r.returncode == 0 and got and got[0].stat().st_size > 10000:
                if got[0].suffix != '.mp4':
                    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(got[0]), '-an', '-c:v', 'libx264',
                                    '-preset', 'veryfast', str(out)], check=False)
                    got[0].unlink(missing_ok=True)
                if out.exists():
                    return channel, vid, 'ok'
            for g in got:
                g.unlink(missing_ok=True)
        return channel, vids[0], 'failed'

    pending = [(c, v) for c, v in by_channel.items() if c not in done_channels]
    started = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {}
        it = iter(pending)
        for _ in range(args.workers * 2):
            c = next(it, None)
            if c:
                futures[pool.submit(fetch, *c)] = c
        while futures:
            for f in as_completed(list(futures)):
                futures.pop(f)
                channel, vid, result = f.result()
                with lock:
                    status[vid] = dict(channel=channel, status=result, start=args.start, end=args.end)
                    ok = sum(v['status'] == 'ok' for v in status.values())
                    tmp = status_path.with_suffix('.tmp')
                    tmp.write_text(json.dumps(status, indent=1))
                    tmp.replace(status_path)
                if ok % 10 == 0 and result == 'ok':
                    print(f'{ok} ok / {len(status)} tried ({time.time() - started:.0f}s)', flush=True)
                if ok < args.target:
                    c = next(it, None)
                    if c:
                        futures[pool.submit(fetch, *c)] = c
                break
    print(f'final {sum(v["status"] == "ok" for v in status.values())} ok', flush=True)


if __name__ == '__main__':
    main()
