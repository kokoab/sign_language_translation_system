# SLT Data Sources Guide

How to supplement your dataset with external ASL video sources.

---

## Prerequisites

```bash
pip install yt-dlp requests beautifulsoup4
```

---

## Step 1: WLASL (282 matching classes)

WLASL has 2,000 ASL signs with 21,000+ videos from YouTube. 282 of your 321 classes are covered.

```bash
# Make sure WLASL repo is cloned (one-time)
cd /tmp && git clone https://github.com/dxli94/WLASL.git

# Dry run first — see what would be downloaded
python3 src/download_wlasl.py --priority p0 --dry-run

# Download critical classes (>50% fail, ~25 classes, ~381 videos)
python3 src/download_wlasl.py --priority p0

# Later: download supplement too (>30% fail)
python3 src/download_wlasl.py --priority p1

# Download all matching classes
python3 src/download_wlasl.py --priority all
```

**Note:** WLASL videos come from YouTube. Many links are dead (~40-60% still work). The script handles failures gracefully.

---

## Step 2: Missing Classes (39 not in WLASL)

These classes aren't in WLASL: AT, C, CODE, CONFUSE, DATA, DELETE, DO, DONT, DOWNLOAD, ELEVEN, EXAM, FALL, FIFTEEN, FOURTEEN, GRADE, HAND, HE, L, LOGIN, LOOK, LOW, MARKET, MODEL, PASSWORD, SEVENTEEN, SO, SOLUTION, SYSTEM, THANKYOU, THIRTEEN, TWELVE, TWENTY, UPLOAD, US, VIDEO, WHOLE, X, Y, Z.

```bash
# See what's missing and where to get it
python3 src/download_other_sources.py --list

# Download from YouTube for critical missing classes
python3 src/download_other_sources.py --download --classes DONT FALL LOW LOOK EXAM

# Download all missing classes
python3 src/download_other_sources.py --download
```

---

## Step 3: MSASL (Microsoft ASL Dataset)

1,000 signs with 25,000+ videos. Excellent quality, multiple signers.

```bash
# Dry run
python3 src/download_msasl.py --dry-run

# Download only critical classes (>50% fail)
python3 src/download_msasl.py --priority p0

# Download all matching classes
python3 src/download_msasl.py
```

---

## Step 4: SignASL.org

10,000+ signs with short, clear, consistent video clips.

```bash
# Dry run — check which of your classes have videos on SignASL
python3 src/download_signasl.py --dry-run

# Download only critical classes
python3 src/download_signasl.py --priority p0

# Download all matching classes
python3 src/download_signasl.py
```

---

## Step 5: How2Sign (Continuous Signing — for Stage 2)

35,000+ sentences of continuous ASL. Great for training Stage 2 (CTC gloss sequences).

**How to get it:**
1. Go to https://how2sign.github.io/
2. Fill out the access request form
3. Download link arrives via email (~1-3 days)
4. Download the "pose" subset if available, otherwise full video

---

## All Sources Summary

| Source | Signs | Videos | Quality | Script |
|--------|-------|--------|---------|--------|
| **WLASL** | 2,000 | 21k | Good, multi-signer | `download_wlasl.py` |
| **YouTube ASL** | Any | Varies | Variable | `download_other_sources.py` |
| **MSASL** | 1,000 | 25k | Excellent | `download_msasl.py` |
| **SignASL.org** | 10,000+ | 10k+ | Excellent, consistent | `download_signasl.py` |
| **How2Sign** | Continuous | 35k sentences | Excellent | Manual request (Step 5) |
| **HandSpeak.com** | 5,000+ | 5k+ | Good, single signer | Manual (no bulk API) |
| **Gallaudet Video Library** | Academic | Comprehensive | Excellent | Manual request |
| **ASL Signbank** | Linguistic | Reference | One clip/sign | Manual |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `yt-dlp: command not found` | `pip install yt-dlp` |
| `ModuleNotFoundError: bs4` | `pip install beautifulsoup4` |
| Most WLASL downloads fail | Normal — ~40-60% of YouTube links are dead. Run MSASL and SignASL too |
| `HTTP 429 Too Many Requests` | You're being rate-limited. Wait 10 min and re-run (scripts skip already-downloaded) |
| Videos download but extraction still fails | Some downloaded videos may be low quality too. Run quality audit after extraction |
| MSASL classes.json download fails | Check internet. The JSON is cached after first download at `/tmp/msasl_cache/` |

---

## Quick Start (download everything automated)

```bash
# Install dependencies
pip install yt-dlp

# Clone external repos
cd /tmp && git clone https://github.com/dxli94/WLASL.git

# 1. WLASL — critical classes first, then all
python3 src/download_wlasl.py --priority p0
python3 src/download_wlasl.py --priority p1
python3 src/download_wlasl.py --priority all

# 2. MSASL — all matching classes
python3 src/download_msasl.py

# 3. SignASL.org — all matching classes
python3 src/download_signasl.py

# 4. YouTube — missing classes not in any dataset
python3 src/download_other_sources.py --download

# 5. Re-extract everything with improved pipeline
rm -rf ASL_landmarks_float16/
python3 src/extract.py
```
