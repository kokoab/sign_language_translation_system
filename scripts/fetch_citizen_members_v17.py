"""Fetch selected members of the official ASL Citizen zip over HTTP range requests.

The 45.9 GB ASL_Citizen.zip is served with Accept-Ranges, so only the ZIP64 central directory (~10 MB)
and the requested videos are downloaded. Members are chosen by gloss and split from the official
metadata CSVs; the official test split is refused (project rule: the Citizen test gate is closed).

    fetch_citizen_members_v17.py --gloss FINGERSPELL --gloss FINGERSPELLING --out data/local/...
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import struct
import sys
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip'
SIZE = 45_924_134_223
META = ROOT / 'data/local/dataset_metadata'


def fetch(start, end):
    request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}', 'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if len(data) != end - start + 1:
        raise IOError(f'short range read {len(data)} != {end - start + 1}')
    return data


def central_directory():
    tail = fetch(SIZE - 65536, SIZE - 1)
    locator = tail.rfind(b'PK\x06\x07')
    if locator < 0:
        raise ValueError('no ZIP64 end-of-central-directory locator')
    (eocd64_offset,) = struct.unpack('<Q', tail[locator + 8:locator + 16])
    record = fetch(eocd64_offset, eocd64_offset + 55)
    if record[:4] != b'PK\x06\x06':
        raise ValueError('bad ZIP64 end-of-central-directory record')
    entries, cd_size, cd_offset = struct.unpack('<QQQ', record[32:56])
    return fetch(cd_offset, cd_offset + cd_size - 1), entries


def parse(cd):
    out, i = {}, 0
    while i + 46 <= len(cd) and cd[i:i + 4] == b'PK\x01\x02':
        method, = struct.unpack('<H', cd[i + 10:i + 12])
        csize, usize = struct.unpack('<II', cd[i + 20:i + 28])
        nlen, xlen, clen = struct.unpack('<HHH', cd[i + 28:i + 34])
        offset, = struct.unpack('<I', cd[i + 42:i + 46])
        name = cd[i + 46:i + 46 + nlen].decode('utf-8', 'replace')
        extra = cd[i + 46 + nlen:i + 46 + nlen + xlen]
        j = 0
        while j + 4 <= len(extra):
            tag, size = struct.unpack('<HH', extra[j:j + 4])
            if tag == 1:   # ZIP64: fields present only where the 32-bit value is 0xFFFFFFFF
                vals, k = extra[j + 4:j + 4 + size], 0
                if usize == 0xFFFFFFFF: usize, = struct.unpack('<Q', vals[k:k + 8]); k += 8
                if csize == 0xFFFFFFFF: csize, = struct.unpack('<Q', vals[k:k + 8]); k += 8
                if offset == 0xFFFFFFFF: offset, = struct.unpack('<Q', vals[k:k + 8]); k += 8
            j += 4 + size
        out[name] = dict(method=method, csize=csize, usize=usize, offset=offset)
        i += 46 + nlen + xlen + clen
    return out


def extract(entry):
    header = fetch(entry['offset'], entry['offset'] + 29)
    if header[:4] != b'PK\x03\x04':
        raise ValueError('bad local header')
    nlen, xlen = struct.unpack('<HH', header[26:30])
    start = entry['offset'] + 30 + nlen + xlen
    data = fetch(start, start + entry['csize'] - 1)
    if entry['method'] == 0:
        return data
    if entry['method'] == 8:
        return zlib.decompress(data, -15)
    raise ValueError(f"unsupported compression method {entry['method']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gloss', action='append', required=True)
    ap.add_argument('--split', action='append', default=['train', 'val'], choices=('train', 'val'))
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    wanted = {}
    for split in a.split:
        with open(META / f'asl_citizen_{split}.csv', encoding='utf-8-sig') as fh:
            for row in csv.DictReader(fh):
                if row['Gloss'].strip().upper() in {g.upper() for g in a.gloss}:
                    wanted[row['Video file']] = dict(split=split, gloss=row['Gloss'], participant=row['Participant ID'])
    print('wanted', len(wanted), flush=True)
    cd, entries = central_directory()
    members = parse(cd)
    print('central directory entries', len(members), 'of', entries, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for video, info in sorted(wanted.items()):
        name = next((n for n in members if n.endswith('/' + video)), None)
        if name is None:
            print('missing in zip', video); continue
        data = extract(members[name])
        path = a.out / info['split'] / video
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest.append(dict(member=name, path=str(path), bytes=len(data), **info))
        print('ok', info['split'], video, len(data), flush=True)
    (a.out / 'manifest.json').write_text(json.dumps(dict(source=URL, test_accessed=False, members=manifest), indent=1))


if __name__ == '__main__':
    main()
