"""Monotone gloss->English alignment and per-gloss edit labels.

Each gloss gets (pre, form, post): tokens inserted before it, its surface form
(or <drop> if unrealized), and tokens after it (only the sentence-final gloss
carries post, which includes the end punctuation). Unmatched English tokens
attach to the next realized gloss; trailing tokens attach to the last gloss.
"""
import json
from pathlib import Path
from forms import FORMS, tokenize, split_counted

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / 'data/local/stage3_multisentence_corpus_v17/corpus.jsonl'
EVAL = ROOT / 'artifacts/reports/stage3_multisentence_eval_v17_20260929/eval_set.jsonl'
DROP = '<drop>'

def align(glosses, toks):
    """Max-match monotone alignment. Returns list of (start, end) or None per gloss."""
    n, m = len(glosses), len(toks)
    NEG = -1
    # best[i][j]: max matches using glosses[:i] and toks[:j]
    best = [[NEG] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        best[0][j] = 0
    for i in range(1, n + 1):
        forms = FORMS[glosses[i - 1]]
        for j in range(m + 1):
            # gloss i unrealized
            cand = [(best[i - 1][j], ('drop', j))]
            if j > 0:  # token j is an insertion (keeps gloss i pending)
                cand.append((best[i][j - 1], ('ins', j - 1)))
            for f in forms:
                L = len(f)
                if j >= L and tuple(toks[j - L:j]) == f and best[i - 1][j - L] >= 0:
                    # small preference for matching later occurrence ties is avoided:
                    cand.append((best[i - 1][j - L] + 1 + 1e-3 * L, ('match', j - L)))
            score, op = max(cand, key=lambda c: c[0])
            best[i][j], back[i][j] = score, op
    # recover
    spans, i, j = [None] * n, n, m
    while i > 0:
        kind, k = back[i][j]
        if kind == 'drop':
            i -= 1
        elif kind == 'ins':
            j = k
        else:
            spans[i - 1] = (k, j); i -= 1; j = k
    return spans

def bag_matches(glosses, toks):
    """Glosses whose form occurs anywhere in the sentence (order-free upper bound)."""
    s = ' ' + ' '.join(toks) + ' '
    used, count = [], 0
    for g in glosses:
        if any((' ' + ' '.join(f) + ' ') in s for f in FORMS[g]):
            count += 1
    return count

def labels(glosses, toks):
    spans = align(glosses, toks)
    out, cursor = [], 0
    for g, sp in zip(glosses, spans):
        if sp is None:
            out.append({'pre': '', 'form': DROP, 'post': ''})
            continue
        out.append({'pre': ' '.join(toks[cursor:sp[0]]), 'form': ' '.join(toks[sp[0]:sp[1]]), 'post': ''})
        cursor = sp[1]
    tail = ' '.join(toks[cursor:])
    # attach tail to last gloss (realized or not); leading text before a dropped-only
    # sentence is impossible since every sentence has >= 1 gloss.
    out[-1]['post'] = tail
    matched = sum(sp is not None for sp in spans)
    return out, matched, spans

def corpus_rows(split=None):
    for line in CORPUS.open():
        r = json.loads(line)
        if split and r['split'] != split:
            continue
        yield r

def row_sentences(r):
    """Yield (gloss_span, english_sentence) using the corpus sentence counts."""
    sents = split_counted(r['english'])
    counts = r['sentence_counts']
    if len(sents) != len(counts) or sum(counts) != len(r['glosses']):
        return None
    out, k = [], 0
    for c, s in zip(counts, sents):
        out.append((r['glosses'][k:k + c], s)); k += c
    return out
