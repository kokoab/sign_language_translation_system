"""Evaluation of the fixed live engine (meaning-level, speed and stability).

Compares: v2 engine (draft k0 + commit k1 + learned clause finalizer + decoding
fixes), phase-2 plain k1 tagger, and the installed tiny T5 at Finish.
Meaning features per held-out sentence vs the reference: polarity, question vs
statement, tense, WH word, every sign represented. "Thought there" = all agree.
These are automatic proxies, not human judgement.
"""
import collections
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
import sacrebleu  # noqa: E402
import evaluate as E  # noqa: E402
import coverage as CV  # noqa: E402
from align import EVAL  # noqa: E402
from forms import tokenize  # noqa: E402
from tagger import Captioner, render  # noqa: E402
from live_caption import LiveCaptioner, tense_conflict  # noqa: E402
from tense_check import PAST, conflict, VERBS  # noqa: E402

torch.set_num_threads(1)  # closer to a phone CPU core than the 4-thread desktop runs
# Live k=1 tagger: average of the phase-2 model and the boundary-retrained one
# (chosen on corpus validation: boundary 93.8%, joint 82.6%).
LIVE = ['tagger_k1.pt', 'tagger_k1b.pt']
CONTEXT = 0  # chosen on corpus validation (context_select.py), as in the desktop runtime
WH_WORDS = {'what', 'where', 'when', 'who', 'why', 'how'}


def features(text):
    t = tokenize(text)
    words = set(t)
    future = 'will' in words or ('going' in words and 'to' in words and 'go' in words) or 'tomorrow' in words
    past = bool(words & (PAST - {'read'})) or 'yesterday' in words
    return {'negative': bool(words & {'not', 'never'}),
            'question': text.strip().endswith('?'),
            'tense': 'past' if past and not future else 'future' if future and not past else 'present/mixed',
            'wh': sorted(words & WH_WORDS)}


def agree(hyp, ref, glosses):
    a, b = features(hyp), features(ref)
    f = E.fidelity(glosses, hyp)
    out = {k: a[k] == b[k] for k in a}
    out['signs'] = f['all_present']
    out['all'] = all(out.values())
    return out


def heldout(systems):
    rows = [json.loads(l) for l in EVAL.open()]
    rows = [r for r in rows if r['reference']]
    sess = {name: [] for name in systems}
    sent = {name: [] for name in systems}
    meta = []
    for r in rows:
        g = r['glosses']; sc = [0.9] * len(g)
        for name, fn in systems.items():
            sess[name].append(fn(g, sc))
        for s in r['sentences']:
            sg = s['gloss']; ssc = [0.9] * len(sg)
            meta.append({'type': s['type'], 'gloss': ' '.join(sg), 'ref': s['en']})
            for name, fn in systems.items():
                sent[name].append(fn(sg, ssc))
    refs = [r['reference'] for r in rows]
    res = {}
    for name in systems:
        ag = [agree(h, m['ref'], m['gloss'].split()) for h, m in zip(sent[name], meta)]
        by_type = {}
        for t in sorted({m['type'] for m in meta}):
            idx = [i for i, m in enumerate(meta) if m['type'] == t]
            by_type[t] = {'n': len(idx), 'thought_there': sum(ag[i]['all'] for i in idx),
                          'chrf++': round(sacrebleu.corpus_chrf([sent[name][i] for i in idx],
                                                                [[meta[i]['ref'] for i in idx]], word_order=2).score, 2)}
        res[name] = {
            'session_bleu': round(sacrebleu.corpus_bleu(sess[name], [refs]).score, 2),
            'session_chrf++': round(sacrebleu.corpus_chrf(sess[name], [refs], word_order=2).score, 2),
            'sentences': len(meta),
            'thought_there': sum(a['all'] for a in ag),
            **{f'agree_{k}': sum(a[k] for a in ag) for k in ('negative', 'question', 'tense', 'wh', 'signs')},
            'by_type': by_type,
        }
    with (HERE / 'v2_heldout_sentences.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['type', 'gloss', 'ref'] + list(systems))
        w.writeheader()
        for i, m in enumerate(meta):
            w.writerow({**m, **{n: sent[n][i] for n in systems}})
    return res


def stress(systems):
    random.seed(20261013)
    seqs = [CV.expand(CV.TEMPLATES[i % len(CV.TEMPLATES)]) for i in range(2400)]
    seqs += [random.choices(CV.GLOSSES, k=random.randint(3, 7)) for _ in range(800)]
    res = {n: collections.Counter() for n in systems}
    samples = []
    for i, g in enumerate(seqs):
        sc = [0.9] * len(g)
        outs = {n: fn(g, sc) for n, fn in systems.items()}
        has_time = {'TOMORROW', 'YESTERDAY'} & set(g) and VERBS & set(g)
        for n, o in outs.items():
            f = E.fidelity(g, o); r = res[n]
            r['n'] += 1; r['all_signs'] += f['all_present']
            r['invented_sign_words'] += bool(f['other_gloss_words'])
            r['inserted_pronouns'] += bool(f['inserted_pronouns'])
            if 'NO' in g:
                r['neg_n'] += 1; r['neg_kept'] += features(o)['negative'] or 'no' in tokenize(o)
            if set(g) & set(CV.C['WH']) and i < 2400:
                r['wh_n'] += 1; r['wh_question'] += o.strip().endswith('?')
            if has_time:
                r['time_n'] += 1; r['tense_conflict'] += conflict(g, o)
                r['tense_conflict_scoped'] += tense_conflict(g, tokenize(o))
        if i % 40 == 0:
            samples.append({'glosses': ' '.join(g), **outs})
    with (HERE / 'v2_stress_samples.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(samples[0])); w.writeheader(); w.writerows(samples)
    return {n: dict(c) for n, c in res.items()}


POLICIES = {  # tuned on the 20 phone traces (development evidence, not held out)
    'v2_final': dict(end_s=3.5, draft_mode='no_post'),
    'v2_final_end_2.5s': dict(end_s=2.5, draft_mode='no_post'),
    'v2_final_full_draft': dict(end_s=3.5, draft_mode='full'),
    'v2_final_no_draft': dict(end_s=3.5, draft_delay=None),
    'v2_no_finalizer': dict(end_s=3.5, draft_mode='no_post', finalize=False),
    'v2_revise_always': dict(end_s=3.5, draft_mode='no_post', revise='always'),
    'first v2 (pause 1.5, revise always, draft immediate)': dict(pause_s=1.5, end_s=None, revise='always'),
}


def streaming(eng_factory):
    cases = json.loads((HERE.parent / 'cases.json').read_text())
    timed = [c for c in cases if c.get('arrivals') and c.get('event_complete')]
    out = {}
    for name, cfg in POLICIES.items():
        eng = eng_factory(cfg)
        rows = []
        for c in timed:
            r = eng.run(c['glosses'], c['scores'], c['arrivals'])
            last = c['arrivals'][-1] - c['arrivals'][0]
            ev = r['events']
            settle_at = ev[-1]['at']
            for e in reversed(ev):
                if e['text'] != ev[-1]['text']:
                    break
                settle_at = e['at']
            rows.append({'id': c['id'], 'family': c['family'], 'glosses': ' '.join(c['glosses']),
                         'final': r['final'], 'ne': round(r['normalized_erasure'], 4),
                         'first_text_s': next(e['at'] for e in ev if e['text']),
                         'word_ne': round(r['normalized_word_erasure'], 4),
                         'settle_after_last_sign_s': round(max(0.0, settle_at - last), 3),
                         'revising_events': sum(1 for x, y in zip(ev, ev[1:]) if not y['text'].startswith(x['text'].rstrip('.?!,'))),
                         'compute_ms_max': round(max(r['compute_ms']), 2),
                         'timeline': [(e['at'], e['kind'], e['text']) for e in ev]})
        ph = [r for r in rows if r['family'] == 'phone']
        out[name] = {
            'phone_n': len(ph),
            'phone_mean_ne': round(statistics.mean(r['ne'] for r in ph), 4),
            'phone_max_ne': max(r['ne'] for r in ph),
            'phone_mean_word_ne': round(statistics.mean(r['word_ne'] for r in ph), 4),
            'phone_median_first_text_s': statistics.median(r['first_text_s'] for r in ph),
            'phone_median_settle_after_last_sign_s': statistics.median(r['settle_after_last_sign_s'] for r in ph),
            'phone_max_settle_after_last_sign_s': max(r['settle_after_last_sign_s'] for r in ph),
            'phone_utterances_with_any_rewrite': sum(r['revising_events'] > 0 for r in ph),
            'diagnostic_mean_ne': round(statistics.mean(r['ne'] for r in rows if r['family'] != 'phone'), 4),
            'cases': rows}
    return out


def speed(eng):
    """Per-update compute, single thread, including long conversations."""
    random.seed(1)
    ms = []
    for n in (5, 10, 20, 40, 80):
        for _ in range(5):
            g = [random.choice(CV.TEMPLATES) for _ in range(n)]
            seq = []
            for t in g:
                seq += CV.expand(t)
            seq = seq[:n]
            r = eng.run(seq, [0.9] * len(seq), [float(i) for i in range(len(seq))])
            ms += [(n, m) for m in r['compute_ms']]
    allm = sorted(m for _, m in ms)
    pct = lambda p: round(allm[int(p * (len(allm) - 1))], 2)
    by_len = {n: round(statistics.median(m for k, m in ms if k == n), 2) for n in (5, 10, 20, 40, 80)}
    return {'updates': len(allm), 'p50_ms': pct(.5), 'p95_ms': pct(.95), 'max_ms': round(allm[-1], 2),
            'median_ms_by_conversation_length': by_len}


def main():
    t0 = time.time()
    if len(sys.argv) > 1 and sys.argv[1] == 'streaming':
        stream = streaming(lambda cfg: LiveCaptioner(live_path=LIVE, context=CONTEXT, **cfg))
        for k, v in stream.items():
            print(k, {kk: vv for kk, vv in v.items() if kk != 'cases'})
        return
    eng = LiveCaptioner(end_s=3.5, draft_mode='no_post', live_path=LIVE, context=CONTEXT)
    plain = Captioner(1); tiny = E.TinyT5()
    systems = {
        'v2_engine': lambda g, s: eng.final_text(g, s),
        'phase2_k1': lambda g, s: render(plain.tags(*E.slot_glosses(g)[:1], s, True)),
        'tiny_t5_finish': lambda g, s: tiny(g, s),
    }
    held = heldout(systems)
    st = stress(systems)
    stream = streaming(lambda cfg: LiveCaptioner(live_path=LIVE, context=CONTEXT, **cfg))
    sp = speed(eng)
    # tiny T5 single-request cost for reference, same thread setting
    tms = []
    for g in (['HELLO', 'MY', 'FRIEND', 'HOW', 'YOU'], ['I', 'GO', 'HOSPITAL', 'TOMORROW', 'MORNING'] * 3):
        for _ in range(5):
            c = time.perf_counter(); tiny(g, [0.9] * len(g)); tms.append((time.perf_counter() - c) * 1000)
    sp['tiny_t5_single_request_median_ms'] = round(statistics.median(tms), 2)
    examples = {}
    for txt in ('I GO HOSPITAL TOMORROW MORNING', 'I GO DOCTOR TOMORROW MORNING', 'TOMORROW MORNING I GO HOSPITAL',
                'YOU GO HOSPITAL WHEN', 'YOU NAME WHAT', 'I WANT WATER NO', 'I GO HOSPITAL NO', 'NO I TIRED',
                'THEY HUNGRY WHO', 'HELLO FRIEND I HELP', 'MY NAME fs-GE fs-LO', 'I KNOW YOU LIKE NO',
                'HELLO MY FRIEND HOW YOU', 'YESTERDAY I WORK TOMORROW I SLEEP', 'PLEASE YOU GIVE I WATER'):
        g = txt.split(); r = eng.run(g, [0.9] * len(g), [float(i) for i in range(len(g))])
        examples[txt] = {'timeline': [f"{e['at']:.2f}s {e['kind']}: {e['text']}" for e in r['events']],
                         'tiny_t5_finish': tiny(g, [0.9] * len(g))}
    out = {'heldout': held, 'stress': st, 'streaming': {k: {kk: vv for kk, vv in v.items() if kk != 'cases'}
                                                        for k, v in stream.items()},
           'speed': sp, 'examples': examples, 'seconds': round(time.time() - t0, 1)}
    (HERE / 'eval_v2.json').write_text(json.dumps(out, indent=1))
    (HERE / 'eval_v2_streaming_cases.json').write_text(json.dumps(
        {k: v['cases'] for k, v in stream.items()}, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'examples'}, indent=1))
    for k, v in examples.items():
        print('\n' + k, '| T5:', v['tiny_t5_finish']); print('  ' + '\n  '.join(v['timeline']))


if __name__ == '__main__':
    main()
