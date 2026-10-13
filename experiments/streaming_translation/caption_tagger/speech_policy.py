"""How soon should captions be spoken? Sweep the settle pause on real timing.

Streams: the 20 timed phone traces and every desktop session with >= 2 committed words
(arrival = time the word was shown). For each settle pause: display erasure, speech
delay after the last sign of a turn, number of spoken chunks per turn, and how often a
spoken chunk is followed by more signing within 3.5 s (a mid-thought pause).
Development evidence (these streams were not held out).
"""
import glob
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts.live_captions_v17 import CaptionModels, LiveCaptions  # noqa: E402

streams = []
for c in json.loads((HERE.parent / 'cases.json').read_text()):
    if c['family'] == 'phone' and c.get('event_complete'):
        streams.append(('phone', c['glosses'], c['scores'], c['arrivals']))
for h in sorted(glob.glob(str(ROOT / 'artifacts/app_sessions/*/history.json'))):
    try:
        d = json.loads(Path(h).read_text())
    except Exception:
        continue
    w = [x for x in d.get('segmental_words', []) if 'shown_elapsed_seconds' in x]
    if len(w) >= 2:
        streams.append(('desktop', [x['gloss'] for x in w], [x['score'] for x in w],
                        [x['shown_elapsed_seconds'] for x in w]))

m = CaptionModels()
out = {}
for settle in (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.5):
    erased = total = 0
    delays, chunks, mid = [], [], 0
    for _, g, s, t in streams:
        c = LiveCaptions(m, end_seconds=settle)
        prev, n_chunks = [], 0
        for j in range(len(g)):
            u = c.push(g[j], s[j], t[j])
            for update in (u, c.tick(t[j] + settle) if (j + 1 == len(g) or t[j + 1] > t[j] + settle) else None):
                if update is None:
                    continue
                toks = c.tokens
                lcp = next((i for i, (a, b) in enumerate(zip(prev, toks)) if a != b), min(len(prev), len(toks)))
                erased += len(prev) - lcp; prev = list(toks)
                if update.speak:
                    n_chunks += 1
                    turn_end = j + 1 == len(g) or t[j + 1] - t[j] > 3.5
                    if update.kind == 'end_of_turn' and turn_end:
                        delays.append(settle)
                    elif update.kind == 'end_of_turn':
                        mid += 1
        total += len(prev); chunks.append(n_chunks)
    out[settle] = {'streams': len(streams), 'normalized_erasure': round(erased / max(1, total), 4),
                   'speech_delay_after_turn_s': settle, 'mean_spoken_chunks_per_stream': round(statistics.mean(chunks), 2),
                   'mid_thought_speech_chunks': mid}
    print(settle, out[settle], flush=True)
(HERE / 'speech_policy.json').write_text(json.dumps(out, indent=1))
