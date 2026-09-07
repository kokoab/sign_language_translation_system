#!/usr/bin/env python3
"""Check every bank asset and render source/blue-avatar contact sheets for review."""
from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.avatar_rig_v17 import RetargetedAvatar, bone_length_metrics, mirror_avatar
from scripts.render_rigged_avatar_v17 import _load_makehuman, render, sha256
from scripts.compare_avatar_source_v17 import fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bank', type=Path)
    args = parser.parse_args()
    report = json.loads((args.bank/'report.json').read_text())
    manifest_path = Path('active/v17/citizen100_manifest.json')
    assert report['manifest_sha256'] == sha256(manifest_path)
    classes = json.loads(manifest_path.read_text())['classes']
    assert len(report['items']) == len(classes) == 100 and not report['failures']
    assert {r['entry']['gloss'] for r in report['items']} == {c['canonical_label'] for c in classes}
    asset = _load_makehuman(Path('artifacts/tools/makehuman_cc0'))
    tiles, checks = [], []
    for item in report['items']:
        gloss = item['entry']['gloss']
        cls = next(c for c in classes if c['canonical_label'] == gloss)
        assert item['raw_gloss'] == cls['citizen_raw_gloss'] and item['asl_lex_code'] == cls['citizen_asl_lex_code']
        path = args.bank/item['audit']
        assert sha256(path) == item['audit_sha256']
        assert sha256(Path(item['source'])) == item['source_sha256']
        with np.load(path) as data:
            lo, hi = item['core_start_frame'], item['core_stop_frame']
            hands = data['symbol_hands'][lo:hi]
            rig = RetargetedAvatar(data['shoulders'][lo:hi], data['elbows'][lo:hi], hands,
                                   data['hand_states'][lo:hi], np.zeros((hi-lo, 2), bool))
            indices = data['source_indices'][lo:hi]
        assert len(hands) >= 2 and np.isfinite(hands).all()
        metrics = bone_length_metrics(rig)
        assert max(metrics.values()) < 1e-4, (gloss, metrics)
        step = np.linalg.norm(np.diff(hands[:, :, 0], axis=0), axis=-1)
        checks.append(dict(gloss=gloss, frames=len(hands), **metrics,
                           max_wrist_step_metres=float(step.max()),
                           world_observed_fraction=item.get('world_observed_fraction')))
        mirrored = bool(np.all(np.char.startswith(rig.hand_states[:, 1], 'rest'))
                        and np.any(~np.char.startswith(rig.hand_states[:, 0], 'rest')))
        if mirrored:
            rig = mirror_avatar(rig)
        frame = len(hands)//2
        capture = cv2.VideoCapture(item['source'])
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(indices[frame]))
        ok, source = capture.read()
        capture.release()
        assert ok
        if mirrored:
            source = cv2.flip(source, 1)
        panel = np.concatenate((fit(source, 300, 360), fit(render(rig, frame, gloss, asset), 300, 360)), axis=1)
        panel[:28] = 25
        label = gloss + (' (source mirrored)' if mirrored else '')
        cv2.putText(panel, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 230, 255), 1, cv2.LINE_AA)
        tiles.append(panel)
        if len(tiles) == 8 or len(checks) == 100:
            page = (len(checks)-1)//8
            tiles += [np.zeros_like(tiles[0])] * (8-len(tiles))
            sheet = np.concatenate([np.concatenate(tiles[i:i+2], axis=1) for i in range(0, 8, 2)])
            cv2.imwrite(str(args.bank/f'coverage_{page:02d}.jpg'), sheet)
            tiles = []
        print(gloss, flush=True)
    (args.bank/'geometry_audit.json').write_text(json.dumps(dict(classes_checked=100, items=checks,
        coverage_only=True, human_accepted=False, limitations=['Contact sheets show one middle frame per gloss',
        'Fixed anatomy does not validate handshape, contact, timing or nonmanual grammar']), indent=2)+'\n')


if __name__ == '__main__':
    main()
