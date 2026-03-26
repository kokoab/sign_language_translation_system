"""
Unified ASL video downloader — downloads from WLASL, MSASL, SignASL.org, and YouTube.

Prerequisites:
    pip install yt-dlp requests beautifulsoup4

Usage:
    python3 src/download_asl.py --dry-run                     # preview all sources
    python3 src/download_asl.py --source wlasl --priority p0  # WLASL, high-fail classes only
    python3 src/download_asl.py --source msasl --priority all # MSASL, all matched classes
    python3 src/download_asl.py --source signasl              # SignASL.org clips
    python3 src/download_asl.py --source youtube              # YouTube search for missing classes
    python3 src/download_asl.py --source all                  # run all sources in order
"""

import os
import json
import re
import time
import random
import argparse
import subprocess
import urllib.request
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────────────────────

OUTPUT_DIR = "data/raw_videos/ASL VIDEOS"
STATS_FILE = "ASL_landmarks_float16/extraction_stats.json"
INVENTORY_PATH = "md files/ASL_VIDEOS_Class_Inventory.md"
CACHE_DIR = "/tmp/asl_download_cache"

COMPOSITES = {
    'ALSO_SAME': ['ALSO', 'SAME'], 'FEW_SEVERAL': ['FEW', 'SEVERAL'],
    'HE_SHE': ['HE', 'SHE'], 'HIS_HER': ['HIS', 'HER'],
    'I_ME': ['I', 'ME'], 'MARKET_STORE': ['MARKET', 'STORE'],
    'US_WE': ['US', 'WE'],
}

WLASL_JSON_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
MSASL_CLASSES_URL = "https://raw.githubusercontent.com/iamgarcia/msasl-video-downloader/master/MSASL_classes.json"
MSASL_TEST_URL = "https://raw.githubusercontent.com/iamgarcia/msasl-video-downloader/master/MSASL_test.json"
SIGNASL_BASE = "https://www.signasl.org/sign"

WLASL_GLOSS_MAP = {"THANKYOU": "thank you", "GOODBYE": "goodbye", "EXCUSE": "excuse", "DONT": "don't"}

# Classes not in WLASL/MSASL — need YouTube search
YOUTUBE_CLASSES = {
    "DONT": ["ASL sign don't", "how to sign don't ASL"],
    "FALL": ["ASL sign fall", "how to sign fall ASL"],
    "LOW": ["ASL sign low"], "LOOK": ["ASL sign look"],
    "EXAM": ["ASL sign exam test"],
    "HAND": ["ASL sign hand"], "DO": ["ASL sign do"],
    "CONFUSE": ["ASL sign confused"], "SO": ["ASL sign so"], "AT": ["ASL sign at"],
    "CODE": ["ASL sign code programming"], "DATA": ["ASL sign data"],
    "DELETE": ["ASL sign delete"], "DOWNLOAD": ["ASL sign download"],
    "UPLOAD": ["ASL sign upload"], "LOGIN": ["ASL sign login"],
    "VIDEO": ["ASL sign video"], "SYSTEM": ["ASL sign system"],
    "MODEL": ["ASL sign model"], "PASSWORD": ["ASL sign password"],
    "SOLUTION": ["ASL sign solution"],
    "ELEVEN": ["ASL number 11"], "TWELVE": ["ASL number 12"],
    "THIRTEEN": ["ASL number 13"], "FOURTEEN": ["ASL number 14"],
    "FIFTEEN": ["ASL number 15"], "SEVENTEEN": ["ASL number 17"],
    "TWENTY": ["ASL number 20"],
    "C": ["ASL fingerspelling C"], "L": ["ASL fingerspelling L"],
    "X": ["ASL fingerspelling X"], "Y": ["ASL fingerspelling Y"],
    "Z": ["ASL fingerspelling Z"],
}

USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]


# ─── Utilities ────────────────────────────────────────────────────────────────

def load_our_classes():
    """Load class inventory and extraction fail rates."""
    our_classes = set()
    if os.path.exists(INVENTORY_PATH):
        with open(INVENTORY_PATH) as f:
            for line in f:
                if line.startswith('|') and '|' in line[1:]:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 4 and parts[1].isdigit():
                        raw = parts[2].upper().strip()
                        if raw in COMPOSITES:
                            our_classes.update(COMPOSITES[raw])
                        elif raw == 'EXCUSE ME':
                            our_classes.add('EXCUSE')
                            our_classes.add('EXCUSE ME')
                        elif raw.startswith('RIGHT'):
                            our_classes.add('RIGHT')
                        else:
                            our_classes.add(raw)

    fail_rates = {}
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE) as f:
            stats = json.load(f)
        for label, s in stats.items():
            fail_rates[label.upper()] = s.get('fail_rate', 0.0)

    return our_classes, fail_rates


def download_json_cached(url, filename):
    """Download JSON with local cache."""
    cache_path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': random.choice(USER_AGENTS)})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        return data
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def download_video_yt(url, output_path, start_time=None, end_time=None,
                      max_retries=3, timeout=90):
    """Download video via yt-dlp with retry + exponential backoff."""
    if os.path.exists(output_path):
        return True, "exists"

    cmd = ["yt-dlp", "--quiet", "--no-warnings",
           "-f", "best[height<=720]", "-o", output_path]
    if start_time is not None and end_time is not None:
        cmd.extend(["--download-sections", f"*{start_time}-{end_time}"])
    cmd.append(url)

    for attempt in range(max_retries):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0 and os.path.exists(output_path):
                return True, "ok"
            err = result.stderr[:100] if result.stderr else "unknown"
            if "Video unavailable" in err or "Private video" in err:
                return False, "unavailable"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return False, err
        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return False, "timeout"
        except FileNotFoundError:
            return False, "yt-dlp not installed (pip install yt-dlp)"
    return False, "max retries"


def download_direct(url, output_path, timeout=30):
    """Download a direct video URL."""
    if os.path.exists(output_path):
        return True, "exists"
    try:
        import requests as req_lib
        resp = req_lib.get(url, timeout=timeout, stream=True,
                           headers={'User-Agent': random.choice(USER_AGENTS)})
        if resp.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            if os.path.getsize(output_path) < 1024:
                os.remove(output_path)
                return False, "too small"
            return True, "ok"
        return False, f"HTTP {resp.status_code}"
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        return False, str(e)[:60]


# ─── Source: WLASL ────────────────────────────────────────────────────────────

def run_wlasl(our_classes, fail_rates, priority, max_per_class, dry_run):
    """Download from WLASL dataset."""
    print("\n" + "=" * 60)
    print("SOURCE: WLASL (Word-Level ASL)")
    print("=" * 60)

    data = download_json_cached(WLASL_JSON_URL, "wlasl_v0.3.json")
    if not data:
        print("Failed to load WLASL JSON")
        return {}

    lookup = {entry['gloss'].upper(): entry for entry in data}

    matched = {}
    for cls in our_classes:
        if cls in lookup:
            matched[cls] = lookup[cls]
        else:
            mapped = WLASL_GLOSS_MAP.get(cls, '').upper()
            if mapped and mapped in lookup:
                matched[cls] = lookup[mapped]
            else:
                for v in [cls.replace('_', ' '), cls.replace('_', '')]:
                    if v.upper() in lookup:
                        matched[cls] = lookup[v.upper()]
                        break

    thresholds = {"p0": 0.5, "p1": 0.3, "all": 0.0}
    filtered = {c: (e, fail_rates.get(c, 0.0)) for c, e in matched.items()
                if fail_rates.get(c, 0.0) >= thresholds[priority]}

    total = sum(min(len(e['instances']), max_per_class) for e, _ in filtered.values())
    print(f"Matched: {len(matched)}, Priority {priority}: {len(filtered)}, Videos: {total}")

    if dry_run:
        for cls, (entry, fr) in sorted(filtered.items(), key=lambda x: -x[1][1]):
            n = min(len(entry['instances']), max_per_class)
            print(f"  {cls:<20} {fr*100:>5.1f}% fail  {n:>3} videos")
        return {}

    stats = defaultdict(lambda: {"ok": 0, "fail": 0, "skip": 0})
    for cls, (entry, fr) in sorted(filtered.items(), key=lambda x: -x[1][1]):
        class_dir = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(class_dir, exist_ok=True)
        instances = entry['instances'][:max_per_class]
        print(f"\n--- {cls} ({len(instances)} videos, {fr*100:.0f}% fail) ---")

        for i, inst in enumerate(instances):
            vid_id = inst.get('video_id', f"vid_{i}")
            url = inst.get('url', '') or f"https://www.youtube.com/watch?v={vid_id}"
            out_path = os.path.join(class_dir, f"wlasl_{vid_id}.mp4")

            start_time = end_time = None
            start, end, fps = inst.get('frame_start'), inst.get('frame_end'), inst.get('fps', 25)
            if start is not None and end is not None and fps > 0:
                start_time, end_time = start / fps, end / fps

            success, msg = download_video_yt(url, out_path, start_time, end_time)
            if success:
                stats[cls]["skip" if msg == "exists" else "ok"] += 1
            else:
                stats[cls]["fail"] += 1
            print(f"  [{i+1}/{len(instances)}] {msg}")
            time.sleep(0.3)

    return dict(stats)


# ─── Source: MSASL ────────────────────────────────────────────────────────────

def run_msasl(our_classes, fail_rates, priority, max_per_class, dry_run):
    """Download from MSASL dataset (test split only — train/val require Microsoft approval)."""
    print("\n" + "=" * 60)
    print("SOURCE: MSASL (Microsoft ASL) — test split")
    print("=" * 60)

    classes = download_json_cached(MSASL_CLASSES_URL, "msasl_classes.json")
    if not classes:
        print("Failed to load MSASL classes")
        return {}

    class_to_idx = {c.upper(): i for i, c in enumerate(classes)}

    all_samples = []
    test = download_json_cached(MSASL_TEST_URL, "msasl_test.json")
    if test:
        all_samples.extend(test)
        print(f"  test split: {len(test)} samples")

    # Check local train/val (require Microsoft approval)
    for fname in ["MSASL_train.json", "MSASL_val.json"]:
        path = os.path.join("/tmp/msasl-video-downloader", fname)
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            all_samples.extend(d)
            print(f"  {fname} (local): {len(d)} samples")

    matched = {}
    for cls in our_classes:
        if cls in class_to_idx:
            idx = class_to_idx[cls]
            samples = [s for s in all_samples if s.get('label', -1) == idx]
            if samples:
                matched[cls] = (samples, fail_rates.get(cls, 0.0))

    thresholds = {"p0": 0.5, "p1": 0.3, "all": 0.0}
    filtered = {c: v for c, v in matched.items() if v[1] >= thresholds[priority]}

    total = sum(min(len(s), max_per_class) for s, _ in filtered.values())
    print(f"Matched: {len(matched)}, Priority {priority}: {len(filtered)}, Videos: {total}")

    if dry_run:
        for cls, (samples, fr) in sorted(filtered.items(), key=lambda x: -x[1][1]):
            n = min(len(samples), max_per_class)
            print(f"  {cls:<20} {fr*100:>5.1f}% fail  {n:>3} videos")
        return {}

    stats = defaultdict(lambda: {"ok": 0, "fail": 0})
    for cls, (samples, fr) in sorted(filtered.items(), key=lambda x: -x[1][1]):
        class_dir = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(class_dir, exist_ok=True)
        subset = samples[:max_per_class]
        print(f"\n--- {cls} ({len(subset)} videos) ---")

        for i, s in enumerate(subset):
            url = s.get('url', '')
            if not url:
                continue
            if not url.startswith('http'):
                url = 'https://' + url
            start = s.get('start_time', s.get('start'))
            end = s.get('end_time', s.get('end'))
            vid_id = url.split('=')[-1] if 'youtube' in url else f"msasl_{i}"
            out_path = os.path.join(class_dir, f"msasl_{vid_id}_{i}.mp4")

            success, msg = download_video_yt(url, out_path, start, end)
            if success:
                stats[cls]["ok"] += 1
            else:
                stats[cls]["fail"] += 1
            print(f"  [{i+1}/{len(subset)}] {msg}")
            time.sleep(0.3)

    return dict(stats)


# ─── Source: SignASL.org ──────────────────────────────────────────────────────

def _signasl_get_videos(sign_name, max_retries=2):
    """Scrape video URLs from a SignASL.org sign page."""
    try:
        import requests as req_lib
        from bs4 import BeautifulSoup
    except ImportError:
        print("  Need: pip install requests beautifulsoup4")
        return []

    url_name = sign_name.lower().replace(' ', '-').replace("'", '')
    url = f"{SIGNASL_BASE}/{url_name}"

    for attempt in range(max_retries):
        try:
            resp = req_lib.get(url, timeout=15,
                               headers={'User-Agent': random.choice(USER_AGENTS)})
            if resp.status_code in (404, 403):
                if resp.status_code == 403 and attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return []
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, 'html.parser')
            videos = []
            for video in soup.find_all('video'):
                for source in video.find_all('source'):
                    src = source.get('src', '')
                    if src and src.endswith(('.mp4', '.webm')):
                        if not src.startswith('http'):
                            src = 'https://www.signasl.org' + src
                        videos.append(src)
            for iframe in soup.find_all('iframe'):
                src = iframe.get('src', '')
                if 'youtube' in src or 'youtu.be' in src:
                    match = re.search(r'(?:embed/|v=|youtu\.be/)([a-zA-Z0-9_-]{11})', src)
                    if match:
                        videos.append(f"https://www.youtube.com/watch?v={match.group(1)}")
            return videos
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
            continue
    return []


def run_signasl(our_classes, fail_rates, priority, dry_run):
    """Download from SignASL.org."""
    print("\n" + "=" * 60)
    print("SOURCE: SignASL.org")
    print("=" * 60)

    thresholds = {"p0": 0.5, "p1": 0.3, "all": 0.0}
    filtered = {c: fail_rates.get(c, 0.0) for c in our_classes
                if fail_rates.get(c, 0.0) >= thresholds[priority]}

    print(f"Checking {len(filtered)} classes...")

    if dry_run:
        found = 0
        for cls, fr in sorted(filtered.items(), key=lambda x: -x[1]):
            videos = _signasl_get_videos(cls)
            found += len(videos)
            print(f"  {cls:<20} {fr*100:>5.1f}% fail  {len(videos)} videos")
            time.sleep(0.5)
        print(f"Total found: {found}")
        return {}

    stats = defaultdict(lambda: {"ok": 0, "fail": 0})
    for cls, fr in sorted(filtered.items(), key=lambda x: -x[1]):
        class_dir = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(class_dir, exist_ok=True)
        videos = _signasl_get_videos(cls)
        if not videos:
            time.sleep(0.5)
            continue

        print(f"  {cls}: {len(videos)} videos")
        for i, url in enumerate(videos):
            out_path = os.path.join(class_dir, f"signasl_{cls.lower()}_{i}.mp4")
            if 'youtube.com' in url or 'youtu.be' in url:
                success, msg = download_video_yt(url, out_path)
            else:
                success, msg = download_direct(url, out_path)
            if success:
                stats[cls]["ok"] += 1
            else:
                stats[cls]["fail"] += 1
            print(f"    [{i+1}/{len(videos)}] {msg}")
        time.sleep(1)

    return dict(stats)


# ─── Source: YouTube search ───────────────────────────────────────────────────

def run_youtube(max_per_class, max_duration, dry_run, classes_filter=None):
    """YouTube search for classes not covered by other sources."""
    print("\n" + "=" * 60)
    print("SOURCE: YouTube search (missing classes)")
    print("=" * 60)

    targets = YOUTUBE_CLASSES.copy()
    if classes_filter:
        upper = {c.upper() for c in classes_filter}
        targets = {c: q for c, q in targets.items() if c in upper}

    print(f"Classes: {len(targets)}, max {max_duration}s per video")

    if dry_run:
        for cls, queries in sorted(targets.items()):
            print(f"  {cls:<20} {len(queries)} queries")
        return {}

    results = {}
    for cls, queries in sorted(targets.items()):
        class_dir = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(class_dir, exist_ok=True)
        total = 0

        for query in queries:
            if total >= max_per_class:
                break
            remaining = max_per_class - total
            print(f"  {cls}: searching '{query}' (max {remaining})...")

            cmd = [
                "yt-dlp", "--quiet", "--no-warnings",
                f"ytsearch{remaining}:{query}",
                "-f", "best[height<=720]",
                "-o", os.path.join(class_dir, f"yt_{cls}_%(id)s.%(ext)s"),
                "--max-downloads", str(remaining),
                "--match-filter", f"duration < {max_duration}",
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            # Count valid files
            for f in os.listdir(class_dir):
                if f.startswith(f"yt_{cls}_"):
                    fpath = os.path.join(class_dir, f)
                    if os.path.getsize(fpath) < 5000:
                        os.remove(fpath)
                    else:
                        total += 1
            time.sleep(1)

        results[cls] = total
        if total > 0:
            print(f"  {cls}: {total} videos")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified ASL video downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sources:
  wlasl     WLASL dataset (YouTube, best coverage)
  msasl     MSASL dataset (YouTube, test split only)
  signasl   SignASL.org (short clips, web scraping)
  youtube   YouTube search (for classes not in other datasets)
  all       Run all sources in order
        """)
    parser.add_argument("--source", choices=["wlasl", "msasl", "signasl", "youtube", "all"],
                        default="all")
    parser.add_argument("--priority", choices=["p0", "p1", "all"], default="p0",
                        help="p0: >50%% fail, p1: >30%% fail, all: everything")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=30)
    parser.add_argument("--max-duration", type=int, default=15,
                        help="Max video duration for YouTube search (default: 15s)")
    parser.add_argument("--classes", nargs="+",
                        help="YouTube source only: specific classes to download")
    args = parser.parse_args()

    our_classes, fail_rates = load_our_classes()
    print(f"Loaded {len(our_classes)} classes, {len(fail_rates)} with extraction stats")

    all_stats = {}
    sources = ["wlasl", "msasl", "signasl", "youtube"] if args.source == "all" else [args.source]

    for source in sources:
        if source == "wlasl":
            stats = run_wlasl(our_classes, fail_rates, args.priority,
                              args.max_per_class, args.dry_run)
        elif source == "msasl":
            stats = run_msasl(our_classes, fail_rates, args.priority,
                              args.max_per_class, args.dry_run)
        elif source == "signasl":
            stats = run_signasl(our_classes, fail_rates, args.priority, args.dry_run)
        elif source == "youtube":
            stats = run_youtube(args.max_per_class, args.max_duration,
                                args.dry_run, args.classes)

        if stats:
            all_stats[source] = stats

    if not args.dry_run and all_stats:
        # Summary
        print(f"\n{'='*60}")
        print("DOWNLOAD COMPLETE — ALL SOURCES")
        print(f"{'='*60}")
        for source, stats in all_stats.items():
            if isinstance(stats, dict):
                if any(isinstance(v, dict) for v in stats.values()):
                    ok = sum(s.get("ok", 0) for s in stats.values() if isinstance(s, dict))
                    fail = sum(s.get("fail", 0) for s in stats.values() if isinstance(s, dict))
                    print(f"  {source:<12} {ok} ok / {fail} fail")
                else:
                    total = sum(v for v in stats.values() if isinstance(v, int))
                    print(f"  {source:<12} {total} videos")

        with open("download_report.json", 'w') as f:
            json.dump(all_stats, f, indent=2, default=str)
        print(f"\nReport: download_report.json")


if __name__ == "__main__":
    main()
