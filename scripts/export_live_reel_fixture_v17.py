"""Parity fixtures for the Swift port of the live segmental Reel (mobile app, LiveReel/*.swift).

For each video: the Python live path (Core ML backend, v3 final config) is run frame by frame and
everything the Swift code must reproduce is recorded:
  - per-frame Apple Vision observations and per-frame hand-crop evidence (inputs to the decoders),
  - per-frame boundary BIO rows from the word and letter decoders,
  - recognizer inputs and logits for sampled spans,
  - committed words after the spelling buffer.
The Swift harness replays the observations and hand evidence (so Vision/crop pixel differences
are excluded) and must match the BIO rows, span inputs/logits and committed words.
Also writes crop fixtures (frame pixels + boxes -> Python crops) and Stage 3 rendering cases.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()


def hand_json(h):
    if h is None:
        return None
    return dict(xy=h.xy.astype(float).tolist(), c=h.confidence.astype(float).tolist(), chirality=h.chirality,
                score=float(h.score))


def obs_json(o):
    d = o.detection
    return dict(seconds=float(o.seconds), width=int(o.frame.shape[1]), height=int(o.frame.shape[0]),
                face_for_features=bool(o.face_for_features), left=hand_json(o.assigned['left']),
                right=hand_json(o.assigned['right']),
                body=dict(xy=d.body_xy.astype(float).tolist(), c=d.body_confidence.astype(float).tolist()),
                face=dict(xy=d.face_xy.astype(float).tolist(), c=d.face_confidence.astype(float).tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='tune')
    ap.add_argument('--videos', type=int, default=4)
    ap.add_argument('--spans', type=int, default=24)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--longest', action='store_true', help='take the longest videos of the set')
    ap.add_argument('--session', type=Path, action='append', default=[], help='recorded app session directory')
    ap.add_argument('--session-seconds', type=float, default=40., help='first N seconds of each session')
    a = ap.parse_args()
    from scripts import segmental_lab_v17 as lab
    from scripts.evaluate_temporal_boundary_v17 import arguments
    from scripts.live_stage2_ctc_v17 import observe_stage2_frame
    from scripts.live_isolated_v17 import crop_square
    from active.v17.extract_v17 import AppleVisionDetector
    from active.v17.segmental_runtime_v17 import build_runtime, SpellingBuffer
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rt = build_runtime()
    args = arguments('data', lab.REPORT / 'sessions')
    det = AppleVisionDetector(args.minimum_point_confidence)
    rng = np.random.default_rng(0)
    captured = {}
    for name, sub in (('words', rt.words), ('letters', rt.letter_rt)):
        original = sub.boundary.update

        def update(raw, seconds, _orig=original, _name=name):
            out = _orig(raw, seconds)
            if out is not None and out[0] is not None:
                captured[_name].append(np.asarray(out[1], np.float32).tolist())
            return out
        sub.boundary.update = update
    index_rows = []
    rows = lab.rows_for(a.set)
    if a.longest:
        def duration(r):
            c = cv2.VideoCapture(str(ROOT / r['video_path']))
            value = c.get(cv2.CAP_PROP_FRAME_COUNT) / max(c.get(cv2.CAP_PROP_FPS), 1)
            c.release()
            return -value
        rows = sorted(rows, key=duration)
    rows = rows[:a.videos]
    for session in a.session:
        history = json.loads((session / 'history.json').read_text())
        rows.append(dict(source_item_id=f'session:{session.name}', video_path=str(session / 'session_lowres.mp4'),
                         target_sequence=[], stamps=history.get('video_source_timestamps_seconds') or []))
    for row in rows:
        rt.reset()
        captured.update(words=[], letters=[])
        cap = cv2.VideoCapture(str(ROOT / row['video_path']))
        fps = cap.get(cv2.CAP_PROP_FPS)
        wrists = {'left': None, 'right': None}
        i = processed = 0
        deadline = 0.
        frames, observations, hands, words = [], [], [], []
        crop_cases = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            stamps = row.get('stamps') or []
            t = stamps[i] if i < len(stamps) else i / fps
            i += 1
            if 'stamps' in row and t > a.session_seconds:
                break
            if t + 1e-6 < deadline:
                continue
            deadline = max(deadline + 1 / 20, t)
            o = observe_stage2_frame(frame, t, processed, det, wrists, args)
            processed += 1
            words += rt.observe(o)
            h = rt.words.hands[-1]
            observations.append(o)
            hands.append(h)
            frames.append(dict(obs_json(o), hand=dict(emb=b64(h[0]), valid=h[1].astype(float).tolist(),
                                                      boxes=h[2].astype(float).tolist())))
            # Crop fixtures: a few frames with hands, with the processed frame's pixels.
            if len(crop_cases) < 3 and (h[1] > 0).any() and processed % 7 == 0:
                from scripts.live_isolated_v17 import hand_box, union_box
                from active.v17.schema_hand_rgb_v17 import HandRGBV17Config
                config = HandRGBV17Config()
                img = o.frame
                height, width = img.shape[:2]
                seen, cases = [], []
                for side in ('left', 'right'):
                    hand = o.assigned[side]
                    box = None if hand is None else hand_box(hand, width, height, config)
                    if box is None:
                        continue
                    seen.append(box)
                    cases.append(box)
                u = union_box(seen, width, height, config.union_box_scale)
                if u is not None:
                    cases.append(u)
                stem = f"{len(index_rows)}_{len(crop_cases)}"
                cv2.imwrite(str(a.output_dir / f'crop_frame_{stem}.png'), img)
                crops = [crop_square(img, b, config.crop_size) for b in cases]
                for k, c in enumerate(crops):
                    cv2.imwrite(str(a.output_dir / f'crop_{stem}_{k}.png'), c)
                crop_cases.append(dict(frame=f'crop_frame_{stem}.png', boxes=[b.astype(float).tolist() for b in cases],
                                       crops=[f'crop_{stem}_{k}.png' for k in range(len(crops))],
                                       embeddings=[b64(rt.recognizer._encode(c)) for c in crops]))
        cap.release()
        words += rt.finish()
        speller = SpellingBuffer()
        final = []
        for w in words:
            final += speller.push(w)
        final += speller.flush()
        # Sampled spans: recognizer inputs and logits over explicit observation indices.
        spans = []
        n = len(observations)
        for _ in range(a.spans):
            s = int(rng.integers(0, max(1, n - 8)))
            e = min(n - 1, s + int(rng.integers(3, 26)))
            ids = list(range(s, e + 1))
            x = rt.recognizer.inputs([observations[k] for k in ids], [hands[k] for k in ids])
            if x is None:
                spans.append(dict(ids=ids, landmarks=None, logits=None))
                continue
            logits = rt.recognizer.logits([x])[0]
            spans.append(dict(ids=ids, landmarks=b64(x[0]), embeddings=b64(x[1]), valid=x[2].astype(float).tolist(),
                              boxes=b64(x[3]), logits=np.asarray(logits, float).tolist()))
        name = f"fixture_{len(index_rows)}.json"
        (a.output_dir / name).write_text(json.dumps(dict(
            id=row['source_item_id'], video=str(ROOT / row['video_path']), reference=row['target_sequence'],
            frames=frames, bio_words=captured['words'], bio_letters=captured['letters'], spans=spans,
            crops=crop_cases,
            words=[{k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in w.items()} for w in final])))
        index_rows.append(dict(file=name, id=row['source_item_id'], frames=len(frames),
                               hypothesis=[w['gloss'] for w in final]))
        print(row['source_item_id'], [w['gloss'] for w in final], flush=True)
    # Stage 3 cases: rendered by the Python naturalizer (with slots) for the Swift renderer.
    from scripts.live_isolated_v17 import make_naturalizer
    from scripts.live_segmental_v17 import render_sentence
    nat_args = arguments('data', lab.REPORT / 'sessions')
    nat_args.naturalizer = 'tiny'
    nat_args.stage3_checkpoint = ROOT / rt.config['stage3_checkpoint']
    naturalizer = make_naturalizer(nat_args)
    corpus = [json.loads(l) for l in open(ROOT / 'data/local/stage3_asl_corpus_v17/corpus_with_fs_slots.jsonl')]
    held = [r for r in corpus if r['split'] == 'test'][:40]
    cases = [dict(glosses=r['glosses'], scores=r['confidences']) for r in held]
    cases += [dict(glosses=['HELLO', 'HOW', 'YOU'], scores=[.9, .8, .7]),
              dict(glosses=['MY', 'NAME', 'fs-GELO'], scores=[.9, .9, .8]),
              dict(glosses=['I', 'DIFFERENT', 'fs-CAT', 'GO'], scores=[.5, .45, .9, .3]),
              dict(glosses=['fs-JOHN'], scores=[.7])]
    for c in cases:
        v = render_sentence(naturalizer, c['glosses'], c['scores'])
        c['sentence'] = v['sentence']
        c['mode'] = v.get('rendering_mode')
    (a.output_dir / 'stage3_cases.json').write_text(json.dumps(cases, indent=1))
    (a.output_dir / 'index.json').write_text(json.dumps(index_rows, indent=1))


if __name__ == '__main__':
    main()
