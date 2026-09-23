"""Forced alignment of the local phrase clips, verified by the isolated recognizers.

The boundary model is trained on 1121 records of which ZERO are local phrase clips
(asllrp_other_ctc 1077, asllrp_contiguous 38, o5s5 6). The local corpus holds 431 clips
with gloss sequences but no interval annotation of any kind, so it has never contributed a
single supervised frame - while the held-out gain of the chain sits almost entirely in the
local subset (61.73% -> 54.94% WER). This script produces the missing intervals.

Equal division of a clip among its glosses is NOT used. Median sign duration is 0.33s and
the median inter-sign gap is also 0.33s, so roughly half of continuous signing is
transition; uniform intervals would label coarticulation as sign interior, which is the
confusion already identified as the root cause.

Three independent systems must agree before a clip is accepted:

  PROPOSE   the frozen DGS BIO teacher segments the clip. It is a different pipeline
            (MediaPipe holistic, noncausal) from anything downstream. It over-segments, so
            its segments are treated as candidate boundaries, not as answers.
  ALIGN     dynamic programming assigns the N expected glosses, in order and without
            overlap, to contiguous merges of those segments, scoring each candidate span by
            the Reel proposal classifier's probability for the EXPECTED gloss. A span whose
            expected gloss is outside the classifier's top-3 scores zero.
  VERIFY    the full visual verifier - a separate model from the proposal classifier -
            must independently rank the expected gloss top-1 on every assigned interval.

A clip is accepted only if all N intervals clear the gate. Distillation is not involved:
the teacher only proposes offline boundaries that become ordinary hard intervals, which is
why round 4's causality failure does not apply here.

Read-only. Nothing is promoted and no model is trained.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
UPSTREAM = ROOT / 'artifacts/reports/pose_boundary_transfer_v17_20260922'
sys.path[:0] = [str(ROOT), str(UPSTREAM), str(UPSTREAM / 'dependencies')]
from scripts.evaluate_temporal_boundary_v17 import arguments, observations
from scripts.live_reel_stage1_v17 import build_components
from active.v17.extract_v17 import AppleVisionDetector

COMBINED = ROOT / 'data/local/combined_dataset_v17_20260922/manifest.json'
REPORT = ROOT / 'artifacts/reports/local_phrase_alignment_v17_20260923'
CACHE = ROOT / 'artifacts/cache/local_phrase_teacher_segments_v17'
MIN_SECONDS, MAX_SECONDS = .15, 4.
CONTEXT = .1                      # the live path's own interval padding


def teacher_segments(rows, limit=None):
    """Cache the frozen teacher's candidate segments for every clip, in seconds."""
    from pose_format.utils.holistic import load_holistic
    from safetensors.torch import load_file
    from check import model_from_upstream
    from probe import upstream_helpers
    CACHE.mkdir(parents=True, exist_ok=True)
    helpers = upstream_helpers()
    weights = ROOT / 'artifacts/models/pose_boundary_dgs_2026'
    model = model_from_upstream(json.loads((weights / 'config.json').read_text())).float().eval()
    model.load_state_dict(load_file(str(weights / 'model.safetensors')), strict=True)
    model.to('mps')
    tick, made = time.perf_counter(), 0
    for i, row in enumerate(rows[:limit]):
        out = CACHE / (row['video_sha256'] + '.json')
        if out.exists():
            continue
        capture = cv2.VideoCapture(str(ROOT / row['video_path']))
        fps, frames = capture.get(cv2.CAP_PROP_FPS), []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        if not frames or not np.isfinite(fps) or fps <= 1:
            out.write_text(json.dumps(dict(segments=[], fps=0, error='unreadable')) + '\n')
            continue
        height, width = frames[0].shape[:2]
        pose = load_holistic(frames, fps=fps, width=width, height=height, pose_workers=1,
                             reuse=False, additional_holistic_config={
                                 'static_image_mode': False, 'model_complexity': 1})
        processed = helpers['preprocess_pose'](pose)
        data = processed.body.data.filled(0)[:, 0, :, :3].astype('float32')
        times = np.arange(len(data), dtype='float32') / fps
        features = np.concatenate([data, helpers['compute_velocity'](data, times)], axis=-1)
        with torch.inference_mode():
            logits = model(torch.from_numpy(features)[None].to('mps'),
                           timestamps=torch.from_numpy(times)[None].to('mps'))['sign'][0].cpu()
        out.write_text(json.dumps(dict(fps=float(fps), frames=len(frames),
            labels=logits.argmax(-1).numpy().astype(int).tolist())) + '\n')
        made += 1
        if (i + 1) % 25 == 0:
            print('  teacher %d/%d  %.0fs' % (i + 1, len(rows[:limit]), time.perf_counter() - tick), flush=True)
    print('teacher segments ready for %d clips (%d new) in %.0fs'
          % (len(rows[:limit]), made, time.perf_counter() - tick), flush=True)


def decode_segments(labels, fps, split_on_b=True, min_frames=3):
    """Upstream grouping, but a new B starts a new sign.

    Upstream `likeliest_probs_to_segments` groups contiguous B/I runs and never splits on a
    later B, so two adjacent signs with no O frame between them come out as one segment.
    That is what under-segments fast local signing: 1-3 segments for 3 glosses. B marks a
    sign beginning, so splitting there recovers the per-sign boundaries.
    """
    O, B, I = 1, 2, 3
    runs, start = [], None
    for i, p in enumerate(labels):
        if p in (B, I):
            if start is None:
                start = i
            elif split_on_b and p == B:
                runs.append((start, i - 1))
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(labels) - 1))
    return [dict(start=a / fps, end=b / fps) for a, b in runs if b - a + 1 >= min_frames]


def span_score(reel, obs, start, end, gloss, cache):
    """Both isolated models' probability for the EXPECTED gloss on this span.

    The proposal classifier and the full visual verifier are different models; an interval
    only counts as real when both see the expected gloss, and the verifier must rank it
    first. Anything outside a model's top-3 scores zero for that model.
    """
    key = (round(start, 4), round(end, 4))
    if key not in cache:
        selected = [o for o in obs if start - CONTEXT <= o.seconds <= end + CONTEXT]
        if len(selected) < 4:
            cache[key] = None
        else:
            cache[key] = (reel.classify(selected), reel.verify(selected))
    entry = cache[key]
    if entry is None:
        return dict(usable=False, score=0., proposal=0., verifier=0.,
                    verifier_gloss=None, verified=False)
    proposal, verifier = entry
    pick = lambda r: next((float(t['model_score']) for t in r['top3'] if t['gloss'] == gloss), 0.)
    p, v = pick(proposal), pick(verifier)
    verified = bool(verifier.get('candidate_gloss') == gloss and p > 0)
    return dict(usable=True, score=p * v, proposal=p, verifier=v,
                verifier_gloss=verifier.get('candidate_gloss'), verified=verified)


def align(reel, obs, segments, glosses, cache):
    """Assign glosses in order to non-overlapping merges of segments; a gloss may be SKIPPED.

    Skipping matters: a clip where two of three signs verify still contributes those two,
    and the unverified region is simply left unsupervised - exactly the contract the pinned
    recipe already uses ("negatives only inside accepted intervals; all other ignored").
    Only verified spans may be assigned, so a wrong label is never forced onto a clip.
    """
    n, m = len(glosses), len(segments)
    if n == 0 or m == 0:
        return []
    spans = {}
    for i in range(m):
        for j in range(i, m):
            start, end = segments[i]['start'], segments[j]['end']
            if MIN_SECONDS - 1e-8 <= end - start <= MAX_SECONDS:
                spans[(i, j)] = (start, end)
    memo = {}

    def solve(k, t):
        """Best (score, assignments) for glosses k.. using segments from index t on."""
        if k == n:
            return 0., []
        if (k, t) in memo:
            return memo[(k, t)]
        best = solve(k + 1, t)                      # skip this gloss
        best = (best[0], list(best[1]))
        for (i, j), (start, end) in spans.items():
            if i < t:
                continue
            info = span_score(reel, obs, start, end, glosses[k], cache)
            if not info['verified']:
                continue
            after = solve(k + 1, j + 1)
            total = info['score'] + after[0]
            if total > best[0]:
                best = (total, [dict(gloss=glosses[k], index=k, start=start, end=end,
                                     proposal_score=info['proposal'],
                                     verifier_score=info['verifier'])] + list(after[1]))
        memo[(k, t)] = best
        return best

    return solve(0, 0)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--role', default='train')
    parser.add_argument('--verify-score', type=float, default=.0,
                        help='extra minimum verifier probability on top of top-1 agreement')
    args_cli = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)

    combined = json.loads(COMBINED.read_text())
    rows = sorted([r for r in combined['records']
                   if r['source'] == 'local_phrases' and r['role'] == args_cli.role],
                  key=lambda r: r['source_item_id'])
    if args_cli.limit:
        rows = rows[:args_cli.limit]
    print('local %s clips: %d, %d sign tokens'
          % (args_cli.role, len(rows), sum(len(r['target_sequence']) for r in rows)), flush=True)

    teacher_segments(rows)

    args = arguments(rows[0]['video_path'], REPORT / 'sessions')
    args.no_motion_trim = True
    reel = build_components(args)['classifier']
    reel.args.no_motion_trim = reel.full.args.no_motion_trim = True
    detector = AppleVisionDetector(args.minimum_point_confidence)

    accepted, tick = [], time.perf_counter()
    reasons, per_gloss = {}, {}
    total_tokens = covered = 0
    for i, row in enumerate(rows):
        def note(why):
            reasons[why] = reasons.get(why, 0) + 1
        cached = json.loads((CACHE / (row['video_sha256'] + '.json')).read_text())
        glosses = row['target_sequence']
        total_tokens += len(glosses)
        if not cached.get('labels'):
            note('teacher_unreadable'); continue
        segments = decode_segments(cached['labels'], cached['fps'])
        if not segments:
            note('teacher_found_no_segments'); continue
        try:
            obs = observations(row, args, detector)
        except Exception:
            note('observation_failure'); continue
        if not obs:
            note('no_observations'); continue
        chosen = align(reel, obs, segments, glosses, {})
        for g in glosses:
            per_gloss.setdefault(g, [0, 0])[1] += 1
        for c in chosen:
            per_gloss[c['gloss']][0] += 1
        if not chosen:
            note('nothing_verified'); continue
        covered += len(chosen)
        if len(chosen) < len(glosses):
            note('partial_%d_of_%d' % (len(chosen), len(glosses)))
        accepted.append(dict(id=row['source_item_id'], video_sha256=row['video_sha256'],
                             video_path=row['video_path'], signer=row['signer_id'],
                             target_sequence=glosses, teacher_segments=len(segments),
                             intervals=[[float(c['start']), float(c['end'])] for c in chosen],
                             aligned=chosen))
        if (i + 1) % 25 == 0:
            print('  aligned %d/%d  clips with >=1 verified interval %d  intervals %d  %.0fs'
                  % (i + 1, len(rows), len(accepted), covered, time.perf_counter() - tick), flush=True)

    print('\nCLIPS with at least one verified interval: %d/%d (%.1f%%)'
          % (len(accepted), len(rows), 100 * len(accepted) / max(len(rows), 1)))
    print('VERIFIED INTERVALS: %d of %d sign tokens (%.1f%%)'
          % (covered, total_tokens, 100 * covered / max(total_tokens, 1)))
    full = sum(1 for a in accepted if len(a['intervals']) == len(a['target_sequence']))
    print('clips aligned in FULL: %d (%.1f%%)' % (full, 100 * full / max(len(rows), 1)))
    print('notes:', json.dumps(reasons))
    print('\nper-gloss verified / occurrences:')
    for g, (ok, tot) in sorted(per_gloss.items(), key=lambda kv: -kv[1][1]):
        print('   %-10s %4d / %4d  %5.1f%%' % (g, ok, tot, 100 * ok / max(tot, 1)))
    if accepted:
        d = [b - a for x in accepted for a, b in x['intervals']]
        v = [c['verifier_score'] for x in accepted for c in x['aligned']]
        print('\naligned sign duration: median %.2fs  mean %.2fs  min %.2f  max %.2f'
              % (np.median(d), np.mean(d), np.min(d), np.max(d)))
        print('verifier confidence on accepted intervals: median %.2f  mean %.2f' % (np.median(v), np.mean(v)))
    (REPORT / 'alignment.json').write_text(json.dumps(dict(
        scope='forced alignment of local phrase clips; frozen DGS teacher proposes candidate '
              'boundaries, DP assigns glosses in order allowing skips, and an interval is kept '
              'only when the full visual verifier ranks the expected gloss top-1 and the '
              'separate proposal classifier also has it in top-3',
        role=args_cli.role, clips=len(rows), clips_with_intervals=len(accepted),
        fully_aligned=full, verified_intervals=covered, sign_tokens=total_tokens,
        per_gloss={g: dict(verified=o, occurrences=t) for g, (o, t) in per_gloss.items()},
        notes=reasons, reel=reel.provenance(), records=accepted), indent=1) + '\n')
    print('wrote', REPORT / 'alignment.json')


if __name__ == '__main__':
    main()
