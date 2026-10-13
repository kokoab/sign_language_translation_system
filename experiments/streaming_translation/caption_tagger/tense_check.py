"""Part C2: committed verb tense vs. time signs over the regenerated stress set.

Conflict = a past verb form together with TOMORROW, or 'will' together with
YESTERDAY, in the final caption. Split by whether the time sign precedes the verb.
"""
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coverage as CV  # noqa: E402
import evaluate as E  # noqa: E402
from forms import tokenize  # noqa: E402
from tagger import Captioner, render  # noqa: E402

PAST = {'went', 'came', 'ate', 'drank', 'gave', 'saw', 'knew', 'took', 'made', 'told', 'thought', 'wrote',
        'found', 'felt', 'slept', 'heard', 'understood', 'was', 'were', 'did', 'had', 'loved', 'liked', 'wanted',
        'needed', 'helped', 'asked', 'worked', 'learned', 'talked', 'stopped', 'tried', 'used', 'waited',
        'listened', 'answered', 'signed', 'read'}
VERBS = set(CV.C['VT'] + CV.C['VI'])


def conflict(glosses, text):
    t = set(tokenize(text))
    if 'TOMORROW' in glosses and t & (PAST - {'read'}):
        return True
    if 'YESTERDAY' in glosses and 'will' in t:
        return True
    return False


def main():
    random.seed(20261013)
    seqs = [CV.expand(CV.TEMPLATES[i % len(CV.TEMPLATES)]) for i in range(2400)]
    seqs = [g for g in seqs if {'TOMORROW', 'YESTERDAY'} & set(g) and VERBS & set(g)]
    caps = {k: Captioner(k) for k in (0, 1, 2)}
    tiny = E.TinyT5(); fin = E.Finalizer(tiny)
    res = collections.defaultdict(collections.Counter)
    examples = []
    for g in seqs:
        ti = min(i for i, x in enumerate(g) if x in ('TOMORROW', 'YESTERDAY'))
        vi = min(i for i, x in enumerate(g) if x in VERBS)
        order = 'time_before_verb' if ti < vi else 'time_after_verb'
        sc = [0.9] * len(g)
        outs = {f'k{k}': render(c.tags(g, sc, True)) for k, c in caps.items()}
        outs['k1_finalize'] = E.detokenize(E.finalized_tokens(caps[1].tags(g, sc, True), g, sc, fin, True)[0])
        outs['tiny_t5_full'] = tiny(g, sc)
        for s, o in outs.items():
            res[(s, order)]['n'] += 1; res[(s, order)]['conflict'] += conflict(g, o)
        if order == 'time_after_verb' and len(examples) < 12:
            examples.append({'glosses': ' '.join(g), **outs})
    out = {f'{s}|{o}': dict(c) for (s, o), c in sorted(res.items())}
    (HERE / 'tense_check.json').write_text(json.dumps({'counts': out, 'examples': examples}, indent=1))
    for k, v in out.items():
        print(f"{k:32s} {v['conflict']}/{v['n']}")
    for e in examples[:8]:
        print(e)


if __name__ == '__main__':
    main()
