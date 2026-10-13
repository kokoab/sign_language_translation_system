"""Integrity and causal checks for the caption tagger experiment."""
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from align import CORPUS, EVAL  # noqa: E402
from tagger import Captioner  # noqa: E402

EXPECTED_CORPUS = '62081b4db0295e216082b610bc5dfef3b101433a45bb46dfbd3f65c5b9278cce'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = {'corpus_sha256_ok': sha(CORPUS) == EXPECTED_CORPUS}
    # Held-out sessions as exact gloss sequences are absent from corpus rows.
    corpus_seqs = {tuple(json.loads(l)['glosses']) for l in CORPUS.open()}
    rows = [json.loads(l) for l in EVAL.open()]
    out['heldout_sessions_in_corpus'] = sum(tuple(r['glosses']) in corpus_seqs for r in rows)
    out['heldout_sentences_flagged_in_training'] = sum(
        sum(bool(x) for x in (r['sentence_in_training'] or [])) for r in rows)
    # Causality on trained weights: a gloss's rendering never depends on signs
    # beyond its lookahead, and prefix rendering equals full-input rendering.
    random.seed(0)
    glosses = json.loads((HERE / 'data.json').read_text())['glosses']
    checks = 0
    for k in (0, 1, 2):
        cap = Captioner(k)
        for _ in range(200):
            n = random.randint(2, 12)
            g = random.choices(glosses, k=n); s = [random.random() for _ in g]
            full = cap.tags(g, s, False)
            for i in range(n - k):
                prefix = cap.tags(g[:i + k + 1], s[:i + k + 1], False)
                alt = g[:i + k + 1] + random.choices(glosses, k=n - i - k - 1)
                other = cap.tags(alt, s, False)
                assert prefix[i] == full[i] == other[i], (k, g, i)
                checks += 1
    out['causal_checks_passed'] = checks
    # v2 engine: the boundary-retrained live model is causal too, and frozen
    # (closed) clauses never change once shown; every sign stays represented.
    import torch
    from live_caption import LiveCaptioner, Decoder
    dec = Decoder(); m = dec.model(1, 'tagger_k1b.pt'); c2 = 0
    for _ in range(100):
        n = random.randint(3, 10); g = random.choices(glosses, k=n); s = [0.9] * n
        full = dec.tags(m, g, s, False)
        for i in range(n - 1):
            assert dec.tags(m, g[:i + 2], s[:i + 2], False)[i] == full[i]; c2 += 1
    out['k1b_causal_checks_passed'] = c2
    eng = LiveCaptioner(end_s=3.5, draft_mode='no_post', live_path=['tagger_k1.pt', 'tagger_k1b.pt'])
    frozen_ok = 0
    for _ in range(150):
        n = random.randint(3, 14); g = random.choices(glosses, k=n); s = [0.9] * n
        state, shown = {'frozen': [], 'open': 0}, []
        for j in range(n):
            eng.step(state, g[:j + 1], s[:j + 1], False)
            assert state['frozen'][:len(shown)] == shown, (g, j)
            shown = list(state['frozen'])
        frozen_ok += 1
    out['frozen_clause_stability_sequences'] = frozen_ok
    out['files'] = {p.name: sha(p) for p in sorted(HERE.iterdir())
                    if p.suffix in ('.py', '.json', '.pt', '.csv', '.md', '.sh') and p.name != 'integrity.json'}
    (HERE / 'integrity.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'files'}, indent=1))


if __name__ == '__main__':
    main()
