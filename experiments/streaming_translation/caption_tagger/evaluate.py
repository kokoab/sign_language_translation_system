"""Frozen evaluation of the caption tagger (run once after training).

1. Held-out multi-sentence set: final captions vs installed tiny T5 and literal.
2. Phase-1 timed cases: caption timelines under commit-k / draft policies.
No references, future words or evaluation outputs enter any decision.
"""
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
import sacrebleu  # noqa: E402
from forms import FORMS, tokenize, detokenize  # noqa: E402
from align import align, bag_matches, EVAL  # noqa: E402
from tagger import Captioner, render  # noqa: E402

torch.set_num_threads(4)
D = json.loads((HERE / 'data.json').read_text())
FUNCTION_WORDS = {w for p in D['pre_vocab'] + D['post_vocab'] for w in p.split()} - set('.,!?;:')
ALL_FORM_WORDS = {w for fs in FORMS.values() for f in fs for w in f}


class TinyT5:
    """Installed phone translator, decoded exactly as phase 1 (no shared cache writes)."""

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        path = str(ROOT / 'artifacts/models/stage3_multisentence_tiny_v17_20260929')
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True).eval()

    def __call__(self, glosses, scores):
        buckets = ['hi' if s >= .6 else 'mid' if s >= .4 else 'lo' for s in scores]
        slots, inputs = {}, []
        for g in glosses:
            if g.startswith('fs-'):
                k = f'FS{len(slots)}'; slots[k] = g[3:].title(); inputs.append(k)
            else:
                inputs.append(g)
        x = self.tok(' '.join(f'{b} {g.lower()}' for b, g in zip(buckets, inputs)), return_tensors='pt')
        if x.input_ids.shape[1] > 64:
            return ' '.join(inputs).capitalize() + '.'
        with torch.inference_mode():
            y = self.model.generate(**x, max_new_tokens=63, do_sample=False, num_beams=1)
        raw = self.tok.decode(y[0], skip_special_tokens=True)
        if slots:
            if all(len(re.findall(r'\b' + k + r'\b', raw, re.I)) == 1 for k in slots):
                for k, v in slots.items():
                    raw = re.sub(r'\b' + k + r'\b', v, raw, flags=re.I)
            else:
                raw = ' '.join(slots.get(g, g.lower()) for g in inputs).capitalize() + '.'
        parts = re.findall(r'(?:^|\s)(\d+):\s*(.*?)(?=\s+\d+:|$)', raw)
        return ' '.join(t.strip() for _, t in parts) if parts else raw


def slot_glosses(glosses):
    """fs-NAME -> FS0 token for the tagger; returns names to restore in order."""
    names = [g[3:].title() for g in glosses if g.startswith('fs-')]
    return ['FS0' if g.startswith('fs-') else g for g in glosses], names


def restore(text, names):
    for n in names:
        text = text.replace('FS0', n, 1)
    return text


def fidelity(glosses, text):
    """Order/realization/negation/invention checks of an output against its input."""
    g = [x for x in glosses if not x.startswith('fs-')]
    toks = [t for t in tokenize(text) if t not in '.,!?;:']
    spans = align(g, toks)
    in_order = sum(s is not None for s in spans)
    present = bag_matches(g, toks)
    input_words = {w for x in g for f in FORMS[x] for w in f}
    outside = [t for t in toks if t in ALL_FORM_WORDS and t not in input_words and t not in FUNCTION_WORDS]
    novel = [t for t in toks if t not in ALL_FORM_WORDS and t not in FUNCTION_WORDS and t != "'s"]
    pron = [t for t in toks if t in {'me', 'him', 'her', 'it', 'them', 'us', 'he', 'she', 'i', 'my'}
            and t not in input_words]
    neg_ok = ('NO' not in g) or any(t in ('no', 'not', 'never') for t in toks)
    return {'n': len(g), 'in_order': in_order, 'present': present, 'all_in_order': in_order == len(g),
            'all_present': present == len(g), 'negation_ok': neg_ok,
            'other_gloss_words': outside, 'novel_words': novel, 'inserted_pronouns': pron}


def summarize(name, hyps, refs, inputs):
    fid = [fidelity(g, h) for g, h in zip(inputs, hyps)]
    negs = [f for f, g in zip(fid, inputs) if 'NO' in g]
    return {
        'system': name, 'n': len(hyps),
        'bleu': round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
        'chrf++': round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
        'exact': sum(h.strip() == r.strip() for h, r in zip(hyps, refs)),
        'all_glosses_present': sum(f['all_present'] for f in fid),
        'all_glosses_in_order': sum(f['all_in_order'] for f in fid),
        'gloss_recall': round(sum(f['present'] for f in fid) / sum(f['n'] for f in fid), 4),
        'negation_kept': f"{sum(f['negation_ok'] for f in negs)}/{len(negs)}",
        'outputs_with_other_gloss_words': sum(bool(f['other_gloss_words']) for f in fid),
        'outputs_with_novel_words': sum(bool(f['novel_words']) for f in fid),
        'outputs_with_inserted_pronouns': sum(bool(f['inserted_pronouns']) for f in fid),
    }, fid


def heldout(caps, tiny, finalizer):
    # Phone rows have no reference; they are covered by the phase-1 timeline replay.
    rows = [json.loads(l) for l in EVAL.open()]
    rows = [r for r in rows if r['reference']]
    out, table = {}, []
    systems = {f'tagger_k{k}': None for k in caps} | {'tagger_k1_finalize': None, 'tiny_t5_full': None,
                                                       'literal': None}
    fin_stats = {'sentences': 0, 'replaced': 0}
    hyps = {s: [] for s in systems}
    sent_hyps = {s: [] for s in systems}
    sent_meta = []
    for r in rows:
        g = r['glosses']; sc = [0.9] * len(g)
        for k, cap in caps.items():
            hyps[f'tagger_k{k}'].append(render(cap.tags(g, sc, True)))
        toks, ns, nr, _ = finalized_tokens(caps[1].tags(g, sc, True), g, sc, finalizer, True)
        hyps['tagger_k1_finalize'].append(detokenize(toks))
        fin_stats['sentences'] += ns; fin_stats['replaced'] += nr
        hyps['tiny_t5_full'].append(tiny(g, sc))
        hyps['literal'].append(detokenize([x.lower() for x in g] + ['.']))
        for s in r['sentences'] or []:
            sg = s['gloss']; ssc = [0.9] * len(sg)
            for k, cap in caps.items():
                sent_hyps[f'tagger_k{k}'].append(render(cap.tags(sg, ssc, True)))
            toks, _, _, _ = finalized_tokens(caps[1].tags(sg, ssc, True), sg, ssc, finalizer, True)
            sent_hyps['tagger_k1_finalize'].append(detokenize(toks))
            sent_hyps['tiny_t5_full'].append(tiny(sg, ssc))
            sent_hyps['literal'].append(detokenize([x.lower() for x in sg] + ['.']))
            sent_meta.append({'type': s['type'], 'gloss': sg, 'ref': s['en']})
    refs = [r['reference'] for r in rows]
    inputs = [r['glosses'] for r in rows]
    clean = [i for i, r in enumerate(rows) if not any(r['sentence_in_training'] or [])]
    for s in systems:
        out[s], fid = summarize(s, hyps[s], refs, inputs)
        out[s]['subset_no_training_overlap'] = summarize(
            s, [hyps[s][i] for i in clean], [refs[i] for i in clean], [inputs[i] for i in clean])[0]
        by_type = {}
        for t in sorted({m['type'] for m in sent_meta}):
            idx = [i for i, m in enumerate(sent_meta) if m['type'] == t]
            by_type[t] = summarize(s, [sent_hyps[s][i] for i in idx], [sent_meta[i]['ref'] for i in idx],
                                   [sent_meta[i]['gloss'] for i in idx])[0]
        out[s]['by_sentence_type'] = by_type
    for i, r in enumerate(rows):
        table.append({'id': r['id'], 'source': r['source'], 'glosses': ' '.join(r['glosses']),
                      'reference': r['reference'], **{s: hyps[s][i] for s in systems}})
    sents = [{**m, 'gloss': ' '.join(m['gloss']), **{s: sent_hyps[s][i] for s in systems}}
             for i, m in enumerate(sent_meta)]
    out['tagger_k1_finalize']['finalizer'] = fin_stats
    return out, table, sents


def tokens_of(tags):
    toks = []
    for pre, form, post in tags:
        toks += pre.split() + ([] if form == '<drop>' else form.split()) + post.split()
    return toks


def sentence_spans(tags, closed_end):
    """Gloss index spans of sentences closed by end punctuation in the tags."""
    spans, start = [], 0
    for i, (pre, form, post) in enumerate(tags):
        if re.search(r'[.?!]', post):
            spans.append((start, i + 1)); start = i + 1
    if closed_end and start < len(tags):
        spans.append((start, len(tags))); start = len(tags)
    return spans, start


def gate(glosses, text):
    f = fidelity(glosses, text)
    return f['all_present'] and f['negation_ok'] and not f['other_gloss_words'] and not f['novel_words']


class Finalizer:
    """Sentence-level T5 rewrite, accepted only if it passes the fidelity gate."""

    def __init__(self, tiny):
        self.tiny, self.cache, self.ms = tiny, {}, {}

    def __call__(self, glosses, scores):
        key = json.dumps(glosses)
        if key not in self.cache:
            t0 = time.perf_counter()
            text = self.tiny(glosses, scores)
            self.ms[key] = (time.perf_counter() - t0) * 1000
            self.cache[key] = text if gate(glosses, text) else None
        return self.cache[key], self.ms[key]


def finalized_tokens(tags, glosses, scores, finalizer, closed_end):
    spans, open_from = sentence_spans(tags, closed_end)
    toks, replaced, cost = [], 0, 0.0
    for a, b in spans:
        text, ms = finalizer(glosses[a:b], scores[a:b]); cost += ms
        if text is None:
            toks += tokens_of(tags[a:b])
        else:
            toks += tokenize(text); replaced += 1
    return toks + tokens_of(tags[open_from:]), len(spans), replaced, cost


def caption_timeline(case, cap, k, draft=None, flush_delay=0.0, finalizer=None):
    """Commit-k: gloss i committed when gloss i+k has arrived; never revised.

    draft: optional k=0 Captioner showing uncommitted glosses provisionally.
    flush_delay: end-of-utterance signal after the last arrival (0 = oracle end).
    """
    g, names = slot_glosses(case['glosses'])
    sc, ts = case['scores'], [t - case['arrivals'][0] for t in case['arrivals']]
    events, prev, erased, shown_at = [], [], 0, {}
    busy = 0.0

    def emit(at, toks, n_committed):
        nonlocal prev, erased
        lcp = 0
        for a, b in zip(prev, toks):
            if a != b:
                break
            lcp += 1
        erased += len(prev) - lcp
        events.append({'at': round(at, 3), 'committed_glosses': n_committed,
                       'text': restore(detokenize(toks), names)})
        prev = toks

    for j in range(len(g)):
        t0 = time.perf_counter()
        tags = cap.tags(g[:j + 1], sc[:j + 1], False)
        n_commit = max(0, j - k + 1) if k else j + 1
        toks = tokens_of(tags[:n_commit])
        fin_ms = 0.0
        if finalizer is not None:
            toks, _, _, fin_ms = finalized_tokens(tags[:n_commit], g[:n_commit], sc[:n_commit], finalizer, False)
        if draft is not None:
            toks = toks + tokens_of(draft.tags(g[:j + 1], sc[:j + 1], False)[n_commit:])
        ms = (time.perf_counter() - t0) * 1000  # includes any first-time sentence finalization
        at = max(ts[j], busy) + ms / 1000; busy = at
        for i in range(n_commit):
            shown_at.setdefault(i, at)
        if draft is not None:
            for i in range(j + 1):
                shown_at.setdefault(('draft', i), at)
        emit(at, toks, n_commit)
    end = ts[-1] + flush_delay
    final_tags = cap.tags(g, sc, True)
    toks = tokens_of(final_tags)
    n_sent = n_rep = 0
    if finalizer is not None:
        toks, n_sent, n_rep, _ = finalized_tokens(final_tags, g, sc, finalizer, True)
    emit(max(end, busy), toks, len(g))
    for i in range(len(g)):
        shown_at.setdefault(i, max(end, busy))
    final = restore(render(final_tags) if finalizer is None else detokenize(toks), names)
    lags = [shown_at[i] - ts[i] for i in range(len(g))]
    first_any = min(v for v in shown_at.values())
    return {'id': case['id'], 'glosses': ' '.join(case['glosses']), 'final': final,
            'normalized_erasure': erased / max(1, len(toks)), 'first_text_s': round(first_any, 3),
            'median_commit_lag_s': round(statistics.median(lags), 3),
            'max_commit_lag_s': round(max(lags), 3), 'sentences': n_sent, 'finalized': n_rep,
            'events': events}


def streaming(caps, finalizer):
    cases = json.loads((HERE.parent / 'cases.json').read_text())
    timed = [c for c in cases if c.get('arrivals') and c.get('event_complete')]
    res = {}
    for name, (k, draft) in {'commit_k0': (0, None), 'commit_k1': (1, None), 'commit_k2': (2, None),
                             'draft_k0_commit_k1': (1, 'k0'), 'draft_k0_commit_k2': (2, 'k0'),
                             'commit_k1_finalize': (1, None)}.items():
        fin = finalizer if name.endswith('finalize') else None
        rows = [caption_timeline(c, caps[k], k, caps[0] if draft else None, finalizer=fin) for c in timed]
        phone = [r for r, c in zip(rows, timed) if c['family'] == 'phone']
        res[name] = {
            'cases': rows,
            'phone_n': len(phone),
            'phone_mean_normalized_erasure': round(statistics.mean(r['normalized_erasure'] for r in phone), 4),
            'phone_median_first_text_s': round(statistics.median(r['first_text_s'] for r in phone), 3),
            'phone_median_commit_lag_s': round(statistics.median(r['median_commit_lag_s'] for r in phone), 3),
            'phone_max_commit_lag_s': round(max(r['max_commit_lag_s'] for r in phone), 3),
        }
    # Invariant: committed final caption equals whole-input rendering (causality).
    for name, (k, _) in {'commit_k1': (1, 0), 'commit_k2': (2, 0)}.items():
        for r, c in zip(res[name]['cases'], timed):
            g, names = slot_glosses(c['glosses'])
            assert r['final'] == restore(render(caps[k].tags(g, c['scores'], True)), names), r['id']
    full = {c['id']: restore(render(caps['full'].tags(*slot_glosses(c['glosses'])[:1], c['scores'], True)),
                             slot_glosses(c['glosses'])[1]) for c in cases}
    finals = {}
    for c in cases:
        g, names = slot_glosses(c['glosses'])
        finals[c['id']] = {'glosses': ' '.join(c['glosses']),
                           **{f'k{k}': restore(render(caps[k].tags(g, c['scores'], True)), names) for k in (0, 1, 2)},
                           'full_context': full[c['id']], 'references': c.get('references'),
                           'phase1_tiny': c.get('historical_english')}
    return res, finals


def main():
    caps = {k: Captioner(k) for k in (0, 1, 2, 'full')}
    tiny = TinyT5()
    t0 = time.time()
    finalizer = Finalizer(tiny)
    held, table, sents = heldout(caps, tiny, finalizer)
    stream, finals = streaming(caps, finalizer)
    user = {}
    for g in ('I GO HOSPITAL TOMORROW MORNING', 'TOMORROW MORNING I GO HOSPITAL',
              'YESTERDAY I GO HOSPITAL', 'YOU GO HOSPITAL WHEN', 'I GO HOSPITAL NO'):
        gl = g.split(); case = {'id': g, 'glosses': gl, 'scores': [0.9] * len(gl),
                                'arrivals': [float(i) for i in range(len(gl))]}
        user[g] = {'tiny_t5_full': tiny(gl, case['scores']),
                   'commit_k1': caption_timeline(case, caps[1], 1)['events'],
                   'draft_k0_commit_k1': caption_timeline(case, caps[1], 1, caps[0])['events'],
                   'commit_k1_finalize': caption_timeline(case, caps[1], 1, finalizer=finalizer)['events']}
    (HERE / 'eval_heldout.json').write_text(json.dumps(held, indent=1))
    (HERE / 'eval_streaming.json').write_text(json.dumps({k: {kk: vv for kk, vv in v.items()}
                                                          for k, v in stream.items()}, indent=1))
    (HERE / 'eval_case_finals.json').write_text(json.dumps(finals, indent=1))
    (HERE / 'eval_user_examples.json').write_text(json.dumps(user, indent=1))
    import csv
    for name, rows in (('heldout_outputs.csv', table), ('heldout_sentence_outputs.csv', sents)):
        with (HERE / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({s: {k: v for k, v in h.items() if k != 'by_sentence_type'} for s, h in held.items()}, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'cases'} for k, v in stream.items()}, indent=1))
    print('eval seconds', round(time.time() - t0, 1))


if __name__ == '__main__':
    main()
