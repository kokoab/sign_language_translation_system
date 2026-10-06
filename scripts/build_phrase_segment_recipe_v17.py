#!/usr/bin/env python3
"""Build the recipe-scoped phrase-segment manifest (2026-10-07) and its symlink view.

Scope: the two downstream stages of the matched 96.83 chain only —
active/v17/train_unified_phrase_adapt_v17.py (frozen-encoder phrase-segment head adaptation)
and scripts/train_span_recognizer_v17.py (interval recognizer). It derives every split from
the canonical approved phrase manifest, which must verify first, plus the segmental lab's
tune/test identities so the lab's held-out test is never trained on or used for selection:

  train      = approved local_phrases train  ∩ lab local_train        (both stages train here)
  validation = approved local_phrases validation − lab test           (phrase-stage selection)
  lab tune   = recognizer selection only; lab test = untouched.

Only positive labelled segments are used (equal-width transcript segments; forced-aligned
decoder spans whose gloss is in the locked vocabulary). No blank/rest/OTHER supervision, no
NCSLGR/Flores/ASLLRP-OTHER inputs or metrics, and no unseen-OOV claim. The global canonical
manifest's training_ready flag is not modified. Refuses to overwrite an existing version.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import segmental_lab_v17 as lab  # noqa: E402

CANONICAL = ROOT / 'active/v17/approved_phrase_manifest_20260921_v2.json'
OUTPUT = ROOT / 'active/v17/phrase_segment_recipe_manifest_20261007.json'
VIEW = ROOT / 'data/local/phrase_segment_recipe_v17_20261007'
SOURCE_RGB = ROOT / 'data/local/stage2_v17_multimodal'
SOURCE_HAND = ROOT / 'data/local/stage2_v17_hand_mobileclip2'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists() or VIEW.exists():
        raise FileExistsError('recipe manifest version already exists; create a new version instead')
    verify = subprocess.run([sys.executable, '-m', 'active.v17.approved_phrase_data_v17'], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    verified = json.loads(verify.stdout[verify.stdout.index('{'):])
    canonical = json.loads(CANONICAL.read_text())
    assert verified['sha256'] == sha256(CANONICAL)
    role = {e['video_sha256']: e['role'] for e in canonical['admitted'] if e['source'] == 'local_phrases'}
    lab_ids = {name: {r['video_sha256'] for r in lab.rows_for(name)} for name in ('local_train', 'tune', 'test')}
    files = {}
    for split in ('train', 'validation'):
        for rgb in sorted((SOURCE_RGB / split / 'local_phrases').glob('*.npz')):
            with np.load(rgb, allow_pickle=False) as payload:
                metadata = json.loads(str(payload['metadata_json'].item()))
            hand = SOURCE_HAND / split / 'local_phrases' / rgb.name.replace(
                '.stage2_rgb_v17.npz', '.stage2_hand_mobileclip2_v17.npz')
            files[metadata['video_sha256']] = dict(rgb=rgb, hand=hand, target=list(metadata['target_sequence']),
                                                   source_item_id=metadata['source_item_id'])
    assign = {}
    for video, approved in role.items():
        if video in lab_ids['test']:
            assert approved == 'validation'
            continue
        if approved == 'train' and video in lab_ids['local_train']:
            assign[video] = 'train'
        elif approved == 'validation':
            assign[video] = 'validation'
    missing = [v for v in assign if v not in files]
    if missing:
        raise ValueError(f'{len(missing)} assigned clips lack multimodal/hand archives')
    for split in ('train', 'validation'):
        assert not ({v for v, s in assign.items() if s == split} & lab_ids['test'])
    assert not ({v for v, s in assign.items() if s == 'train'} & (lab_ids['tune'] | lab_ids['test']))
    entries = []
    for video, split in sorted(assign.items(), key=lambda item: (item[1], item[0])):
        item = files[video]
        for kind, base in (('rgb', VIEW / 'multimodal'), ('hand', VIEW / 'hand')):
            link = base / split / 'local_phrases' / item[kind].name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(item[kind].resolve())
        entries.append(dict(video_sha256=video, split=split, approved_role=role[video],
                            source_item_id=item['source_item_id'], target_sequence=item['target'],
                            rgb=str(item['rgb'].relative_to(ROOT)), rgb_sha256=sha256(item['rgb']),
                            hand=str(item['hand'].relative_to(ROOT)), hand_sha256=sha256(item['hand'])))
    counts = {s: sum(e['split'] == s for e in entries) for s in ('train', 'validation')}
    manifest = dict(
        format='slt_v17_phrase_segment_recipe_manifest', version='20261007',
        created_utc=datetime.now(timezone.utc).isoformat(),
        canonical_manifest=str(CANONICAL.relative_to(ROOT)), canonical_sha256=verified['sha256'],
        canonical_verification=verified,
        scope=['active/v17/train_unified_phrase_adapt_v17.py', 'scripts/train_span_recognizer_v17.py'],
        recipe_training_ready=True,
        review='Prepared by Claude under the user\'s overnight authorization (2026-10-07) to unblock '
               'these two stages properly; user review pending. Global canonical training_ready unchanged.',
        blocker_resolution={
            'excluded_validation_sources': 'No NCSLGR/Flores/ASLLRP-OTHER inputs or metrics. Selection uses '
                                           'approved local validation (minus lab test), isolated validation and the lab tune pool.',
            'blank_rest_oov': 'No blank/rest/OTHER targets; out-of-vocabulary spans are dropped; no unseen-OOV claim.',
            'auxiliary_inputs': 'Isolated feature caches, span labels and captured span inputs are pinned by the '
                                'launching driver (hashes in its recipe.json).'},
        splits=dict(train='approved train ∩ lab local_train', validation='approved validation − lab test',
                    lab_tune='recognizer selection only', lab_test='never trained on, selected on or evaluated'),
        counts=counts, lab_counts={k: len(v) for k, v in lab_ids.items()},
        view_root=str(VIEW.relative_to(ROOT)), entries=entries,
        citizen_test_accessed=False, semlex_test_accessed=False, local_test_accessed=False)
    OUTPUT.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(output=str(OUTPUT.relative_to(ROOT)), sha256=sha256(OUTPUT), counts=counts,
                          lab=manifest['lab_counts']), indent=2))


if __name__ == '__main__':
    main()
