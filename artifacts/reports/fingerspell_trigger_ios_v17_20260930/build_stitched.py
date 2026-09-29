"""Stitched spell-mode test videos (all held-out material; nothing here was used for training).

trigger_k: [sign A][FINGERSPELL][ASLLRP fingerspelled word][FINGERSPELL][sign B]   expect A fs-WORD B
name_k:    [MY][NAME][ASLLRP fingerspelled word][sign B]                           expect MY NAME fs-WORD B
control_k: [sign A][ASLLRP fingerspelled word][sign B]                             (no trigger: words only)
Citizen clips are from the official validation split; FINGERSPELL from Citizen val (held out of the
detector); fingerspelled words from ASLLRP Sign Bank sentence clips (other signers, natural speed).
"""
import csv, glob, json, subprocess
from pathlib import Path
HERE = Path(__file__).parent; ROOT = HERE.parents[2]
OUT = ROOT / 'data/local/spell_mode_test_v17'; SEG = OUT / 'segments'; SEG.mkdir(parents=True, exist_ok=True)
VAL = ROOT / 'data/local/citizen100_v17/raw/val'


def val_clip(gloss, k=0):
    clips = sorted(p for p in (VAL / gloss).glob('*.mp4') if not p.name.startswith('._'))
    return clips[k % len(clips)]


def norm(src):
    dst = SEG / (Path(src).parent.name + '__' + Path(src).stem + '.mp4')
    if not dst.exists():
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(src), '-an', '-vf',
                        'scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', str(dst)], check=True)
    return dst


def stitch(name, parts):
    listing = SEG / (name + '.txt')
    listing.write_text(''.join(f"file '{norm(p)}'\n" for p in parts))
    dst = OUT / (name + '.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(listing), '-c', 'copy', str(dst)], check=True)
    return dst


gloss_of = {}
for f in glob.glob(str(ROOT / 'data/local/dataset_metadata/asllrp_signbank/*sentence*.csv')):
    for r in csv.DictReader(open(f, encoding='utf-8-sig')):
        k = {x.strip().lower(): v for x, v in r.items() if x}
        if k.get('main entry gloss label', '').startswith('fs-'):
            gloss_of[k['sign video filename']] = k['main entry gloss label']
words = []
for p in sorted((ROOT / 'data/local/asllrp_fingerspelled_v17/clips').glob('*.mp4')):
    g = gloss_of.get(p.name, '')
    letters = g[3:]
    if 3 <= len(letters) <= 6 and letters.isalpha() and letters not in {w[1] for w in words}:
        words.append((p, letters))
fs_val = sorted(glob.glob(str(ROOT / 'data/local/fingerspell_trigger_v17/citizen/val/*.mp4')))
A = ['HELLO', 'I', 'WHAT', 'YOU', 'MY', 'WHERE', 'FRIEND', 'MOTHER']
B = ['GOOD', 'HAPPY', 'WORK', 'SCHOOL', 'HOME', 'TIRED', 'HUNGRY', 'SICK']
items = []
for k, fs in enumerate(fs_val):
    w, letters = words[k]
    items.append(dict(id=f'trigger_{k}', path=str(stitch(f'trigger_{k}', [val_clip(A[k]), fs, w, fs, val_clip(B[k])])),
                      reference=[A[k], 'fs-' + letters, B[k]], expect_events=['on:fingerspell', 'off:fingerspell']))
for k in range(4):
    w, letters = words[8 + k]
    items.append(dict(id=f'name_{k}', path=str(stitch(f'name_{k}', [val_clip('MY', k), val_clip('NAME', k), w, val_clip(B[k], 1)])),
                      reference=['MY', 'NAME', 'fs-' + letters, B[k]], expect_events=['on:name', 'off:name']))
    w, letters = words[12 + k]
    items.append(dict(id=f'control_{k}', path=str(stitch(f'control_{k}', [val_clip(A[k], 1), w, val_clip(B[k], 2)])),
                      reference=[A[k], 'fs-' + letters, B[k]], expect_events=[]))
(OUT / 'videos.json').write_text(json.dumps(items, indent=1))
print(len(items), 'videos'); [print(i['id'], i['reference']) for i in items]
