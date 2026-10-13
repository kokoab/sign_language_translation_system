"""Choose the live tagger's left-context size on the corpus validation split.

Runs the desktop caption runtime over all 1,100 validation sessions (signs in
order, end of utterance at the end) for several context sizes. Metrics: session
chrF++ against the corpus English and sentence-count agreement. Held-out and
stress sets are not used for this choice.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import sacrebleu  # noqa: E402
import scripts.live_captions_v17 as L  # noqa: E402
from align import corpus_rows  # noqa: E402
from forms import split_counted  # noqa: E402

rows = [r for r in corpus_rows('validation')]
m = L.CaptionModels()
out = {}
for ctx in (0, 4, 8, 16):
    L.CONTEXT = ctx
    hyps, refs, agree = [], [], 0
    for r in rows:
        c = L.LiveCaptions(m)
        for g, s in zip(r['glosses'], r['confidences']):
            c.push(g, s, 0.)
        text = c.finish().text
        ref = ' '.join(split_counted(r['english']))
        hyps.append(text); refs.append(ref)
        agree += len(re.findall(r'[.?!](\s|$)', text)) == len(r['sentence_counts'])
    out[ctx] = {'chrf++': round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
                'sentence_count_agreement': round(agree / len(rows), 4), 'n': len(rows)}
    print(ctx, out[ctx], flush=True)
(HERE / 'context_select.json').write_text(json.dumps(out, indent=1))
