#!/usr/bin/env python3
"""Download and verify the public RWTH-BOSTON-104 continuous-ASL corpus."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request


BASE = "https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-boston-104/"
ROOT = Path("data/local/open_asl_alternatives_20260913/rwth_boston_104")
REPORT = Path("artifacts/reports/open_asl_alternatives_20260913/rwth_boston104_audit.json")
CORPUS = (
    "all.sentences.corpus", "all.sentences.tracking-results.xml", "boston104.values",
    "devel-test.corpus", "devel-train.corpus", "man-1.corpus", "man-1.test.corpus",
    "man-1.test.recordings", "man-1.test.speaker", "mpg.all.sentences.corpus",
    "mpg.devel-test.corpus", "mpg.devel-train.corpus", "mpg.test.sentences.corpus",
    "mpg.train.sentences.pronunciations.corpus", "speaker.description",
    "test.sentences.corpus", "test.sentences.multi.translations.csv",
    "train.sentences.pronunciations.corpus",
    "train.sentences.pronunciations.multi.translations.csv", "woman-1.corpus",
    "woman-1.test.corpus", "woman-1.test.recordings", "woman-1.test.speaker",
    "woman-2.corpus", "woman-2.test.corpus", "woman-2.test.recordings",
    "woman-2.test.speaker",
)
LEXICON = ("devel-test_0.12.lexicon", "devel-train_0.12.lexicon",
           "test_0.12.lexicon", "train_0.12.lexicon")
LM = ("devel-lm-M1.sri.lm.gz", "devel-lm-M2.sri.lm.gz", "devel-lm-M3.sri.lm.gz",
      "devel-test.corpus.sentences.for-lm-perplexity",
      "devel-train.corpus.sentences-for-lm-training",
      "devel-train.corpus.vocabulary-for-lm-training",
      "train.sentences.pronunciations.corpus.sentences-for-lm-training",
      "ukn.1.lm.gz", "ukn.2.lm.gz", "ukn.3.lm.gz")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def acquire(item):
    remote, relative = item
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(BASE + remote, headers={"User-Agent": "SLT-v17-data-audit/1"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
            expected = int(response.headers.get("Content-Length", 0))
            size = 0
            for block in iter(lambda: response.read(1024 * 1024), b""):
                target.write(block)
                size += len(block)
            assert not expected or size == expected
        temporary.replace(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path),
            "source_url": BASE + remote}


def main():
    items = [(f"videoBank/camera0/{n:03d}_0.mpg", f"videoBank/camera0/{n:03d}_0.mpg")
             for n in range(1, 202)]
    items += [(f"corpus/{name}", f"corpus/{name}") for name in CORPUS]
    items += [(f"lexicon/{name}", f"lexicon/{name}") for name in LEXICON]
    items += [(f"lm/{name}", f"lm/{name}") for name in LM]
    items.append(("readme.info", "readme.info"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(acquire, items))
    videos = sorted((ROOT / "videoBank/camera0").glob("*.mpg"))
    assert len(videos) == 201
    decoded = []
    for video in videos:
        stream = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,duration",
            "-of", "json", str(video)]))["streams"][0]
        assert int(stream["nb_read_frames"]) > 0
        decoded.append({"path": str(video), **stream})
    result = {
        "source": BASE, "files": files, "videos": len(videos),
        "video_bytes": sum(v.stat().st_size for v in videos), "decoded": decoded,
        "training_eligible": False,
        "limitation": "Exact Citizen variant mapping and signer-role audit required before training.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"verified {len(videos)} videos and {len(files) - len(videos)} support files")


if __name__ == "__main__":
    main()
