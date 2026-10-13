"""Part C: the full 100-class combination space, not only the recorded sessions.

1. How much of the gloss-combination space the training corpus covers.
2. Held-out quality split by whether a sentence contains unseen gloss bigrams.
3. Reference-free stress test: thousands of new combinations that use every
   class many times (slot templates + unconstrained random orders). Measures
   per class whether each system drops the sign, invents a sign that was not
   signed, or loses negation. No references exist for these, so English quality
   is shown by samples only, not scored.
"""
import collections
import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sacrebleu  # noqa: E402
from align import corpus_rows, row_sentences, EVAL  # noqa: E402
from forms import GLOSSES, detokenize, tokenize  # noqa: E402
from tagger import Captioner, render  # noqa: E402
import evaluate as E  # noqa: E402

CLASSES = {
    'SUBJ': 'I YOU HE WE THEY', 'PERSON': 'MAN WOMAN CHILD MOTHER FATHER FRIEND DOCTOR FAMILY FS0',
    'POSS': 'MY YOUR OUR', 'THING': 'WATER HOME SCHOOL HOSPITAL LANGUAGE SIGN NAME TIME',
    'PLACE': 'HOME SCHOOL HOSPITAL', 'TIME': 'YESTERDAY TOMORROW NOW MORNING NIGHT DAY WEEK YEAR',
    'VT': 'GIVE HELP ASK TELL SEE KNOW LOVE LIKE WANT NEED FIND TAKE MAKE USE HAVE ANSWER HEAR READ '
          'WRITE LEARN UNDERSTAND EAT DRINK',
    'VI': 'GO COME SLEEP WORK WAIT STOP TALK LISTEN TRY THINK FEEL',
    'ADJ': 'ANGRY BAD BIG COLD DIFFERENT EASY EXCITED GOOD HAPPY HOT HUNGRY IMPORTANT READY SAD SICK '
           'SMALL TIRED SAME',
    'WH': 'WHAT WHERE WHEN WHO WHY HOW', 'DISC': 'HELLO GOODBYE THANKYOU PLEASE SORRY YES NO MAYBE',
    'QUANT': 'MORE LESS',
}
C = {k: v.split() for k, v in CLASSES.items()}
TEMPLATES = [
    'NP ADJ', 'TIME? SUBJ VI', 'TIME? NP VT POSS? THING', 'SUBJ GO PLACE TIME TIME?',
    'TIME SUBJ GO PLACE', 'NP VT THING WH', 'SUBJ ADJ WH', 'SUBJ NO VT THING', 'SUBJ VT THING NO',
    'NP NO ADJ', 'DISC SUBJ VT SUBJ', 'DISC NP', 'SUBJ VT QUANT THING', 'POSS PERSON VT SUBJ',
    'NP VT THING | NP ADJ', 'DISC | TIME? NP VI WH',
]


def np_():
    r = random.random()
    return random.choice(C['SUBJ']) if r < .4 else (
        [random.choice(C['POSS']), random.choice(C['PERSON'] + C['THING'])] if r < .7 else random.choice(C['PERSON']))


def expand(template):
    out = []
    for slot in template.split():
        if slot == '|':
            continue
        opt = slot.endswith('?'); slot = slot.rstrip('?')
        if opt and random.random() < .5:
            continue
        if slot == 'NP':
            x = np_(); out += x if isinstance(x, list) else [x]
        elif slot in C:
            out.append(random.choice(C[slot]))
        else:
            out.append(slot)
    return out


def corpus_ngrams():
    uni, bi, tri, sents = collections.Counter(), set(), set(), set()
    for r in corpus_rows('train'):
        g = r['glosses']; uni.update(g)
        bi.update(zip(g, g[1:])); tri.update(zip(g, g[1:], g[2:]))
        for sg, _ in row_sentences(r):
            sents.add(tuple(sg))
    return uni, bi, tri, sents


def main():
    random.seed(20261013)
    covered = set(sum(C.values(), []))
    assert covered == set(GLOSSES), set(GLOSSES) ^ covered
    uni, bi, tri, sents = corpus_ngrams()
    V = len(GLOSSES)
    space = {
        'classes': V, 'possible_ordered_bigrams': V * V, 'train_distinct_bigrams': len(bi),
        'possible_ordered_trigrams': V ** 3, 'train_distinct_trigrams': len(tri),
        'possible_5_sign_sequences': V ** 5, 'train_distinct_sentences': len(sents),
        'train_class_count_min': min(uni[g] for g in GLOSSES), 'train_class_count_max': max(uni.values()),
        'rarest_classes': sorted(GLOSSES, key=lambda g: uni[g])[:8],
    }
    space['bigram_coverage'] = space['train_distinct_bigrams'] / space['possible_ordered_bigrams']
    space['trigram_coverage'] = space['train_distinct_trigrams'] / space['possible_ordered_trigrams']

    # 2. Held-out split by bigram novelty (sentence level, references exist).
    rows = list(csv.DictReader((HERE / 'heldout_sentence_outputs.csv').open()))
    groups = {'all_bigrams_seen': [], 'has_unseen_bigram': []}
    for r in rows:
        g = r['gloss'].split()
        groups['has_unseen_bigram' if any(b not in bi for b in zip(g, g[1:])) else 'all_bigrams_seen'].append(r)
    heldout_cls = collections.Counter(g for r in rows for g in r['gloss'].split())
    novelty = {'heldout_classes_present': sum(1 for g in GLOSSES if heldout_cls[g]),
               'heldout_sentences_seen_verbatim_in_train': sum(tuple(r['gloss'].split()) in sents for r in rows)}
    for name, rs in groups.items():
        novelty[name] = {'n': len(rs)}
        for s in ('tagger_k1', 'tagger_k1_finalize', 'tiny_t5_full'):
            novelty[name][s] = round(sacrebleu.corpus_chrf([r[s] for r in rs], [[r['ref'] for r in rs]],
                                                           word_order=2).score, 2)

    # 3. Reference-free stress over all classes.
    seqs = []
    for i in range(2400):
        t = TEMPLATES[i % len(TEMPLATES)]
        seqs.append(('template', t, expand(t)))
    for i in range(800):
        seqs.append(('random', 'uniform', random.choices(GLOSSES, k=random.randint(3, 7))))
    cap = Captioner(1); tiny = E.TinyT5(); fin = E.Finalizer(tiny)
    per_class = {s: collections.defaultdict(lambda: [0, 0]) for s in ('tagger_k1', 'tagger_k1_finalize', 'tiny_t5_full')}
    agg = {s: collections.Counter() for s in per_class}
    samples, n_novel_seq, n_unseen_bi = [], 0, 0
    for kind, t, g in seqs:
        sc = [0.9] * len(g)
        tags = cap.tags(g, sc, True)
        outs = {'tagger_k1': render(tags),
                'tagger_k1_finalize': detokenize(E.finalized_tokens(tags, g, sc, fin, True)[0]),
                'tiny_t5_full': tiny(g, sc)}
        novel = tuple(g) not in sents; n_novel_seq += novel
        unseen = any(b not in bi for b in zip(g, g[1:])); n_unseen_bi += unseen
        for s, text in outs.items():
            f = E.fidelity(g, text)
            a = agg[s]; a['n'] += 1; a[f'{kind}_n'] += 1
            a['all_present'] += f['all_present']; a['all_in_order'] += f['all_in_order']
            a['invented_sign_words'] += bool(f['other_gloss_words'])
            a['novel_words'] += bool(f['novel_words']); a['inserted_pronouns'] += bool(f['inserted_pronouns'])
            if 'NO' in g:
                a['neg_n'] += 1; a['neg_kept'] += f['negation_ok']
            a[f'{kind}_all_present'] += f['all_present']
            toks = [x for x in tokenize(text)]
            from forms import FORMS
            for x in set(g):
                present = any(' '.join(fm) in ' ' + ' '.join(toks) + ' ' or
                              (' ' + ' '.join(fm) + ' ') in (' ' + ' '.join(toks) + ' ') for fm in FORMS[x])
                per_class[s][x][0] += 1; per_class[s][x][1] += not present
        if len(samples) < 80 and (len(samples) < 60 or kind == 'random'):
            samples.append({'kind': kind, 'template': t, 'glosses': ' '.join(g), 'unseen_bigram': unseen,
                            **outs, 'k1_timeline': [e['text'] for e in E.caption_timeline(
                                {'id': '', 'glosses': g, 'scores': sc,
                                 'arrivals': [float(i) for i in range(len(g))]}, cap, 1)['events']]})
    stress = {'sequences': len(seqs), 'not_in_training_verbatim': n_novel_seq,
              'with_unseen_bigram': n_unseen_bi, 'finalizer_rejected_sentences':
                  sum(v is None for v in fin.cache.values()), 'finalizer_sentences': len(fin.cache),
              'systems': {s: dict(a) for s, a in agg.items()}}
    worst = {}
    for s, d in per_class.items():
        rates = {g: d[g][1] / d[g][0] for g in GLOSSES if d[g][0]}
        worst[s] = {'classes_tested': len(rates), 'min_occurrences': min(d[g][0] for g in rates),
                    'classes_ever_dropped': sum(1 for r in rates.values() if r > 0),
                    'worst_drop_rates': sorted(((g, round(r, 3), d[g][0]) for g, r in rates.items()),
                                               key=lambda x: -x[1])[:12]}
    out = {'space': space, 'heldout_novelty': novelty, 'stress': stress, 'per_class_drop': worst}
    (HERE / 'coverage.json').write_text(json.dumps(out, indent=1))
    with (HERE / 'stress_samples.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(samples[0])); w.writeheader(); w.writerows(samples)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
