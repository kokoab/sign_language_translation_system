"""Evaluate letter_ctc_v17 checkpoints: FSboard validation, and ASLLRP in-sentence fingerspelling.

ASLLRP (data/local/asllrp_fingerspelled_v17/live_features) is natural-speed spelling inside signed
sentences from signers outside FSboard. Split by signer, fixed before any result: dev = Cory (used only
to choose between configurations and the lexicon acceptance margin), test = every other signer.
Each fingerspelled word is scored on the letters the model emits (token midpoint) within +-PAD seconds
of its annotated span (some annotated spans are implausibly short); a letter in overlapping windows goes
to the nearest word. False letters = letters emitted more than PAD from every spelled word, per minute of
that remaining (signing/rest) time. Baseline = the current phone letter decoder (spell mode), from the
same extraction (letter_times).

Lexicon (data/local/name_lists_v17/lexicon_v1.txt: macOS dictionary + propernames + FSboard train name
tokens): candidates within Levenshtein distance of the greedy word (rapidfuzz), rescored by exact CTC
likelihood on the word window; the best candidate replaces the greedy word when its negative
log-likelihood is at most margin above the greedy string's. The margin is chosen on dev.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.letter_ctc_v17 import (CLASSES, LETTERS, add_deltas, base_features, cer_counts, greedy,
                                       letters_only, load_checkpoint, model_features)

PAD = .5
FEAT = ROOT / 'data/local/asllrp_fingerspelled_v17/live_features'
LEXICON = ROOT / 'data/local/name_lists_v17/lexicon_v1.txt'
DEV_SIGNERS = {'Cory'}
MARGINS = [0, .5, 1, 2, 3, 5, 8, 12, float('inf')]


def run_model(model, z, device):
    base, pres = base_features(z['raw'])
    lm = torch.from_numpy(model_features(model.config, base, pres))[None].to(device)
    emb = torch.from_numpy(z['hand_embeddings'].astype(np.float32))[None].to(device)
    valid = torch.from_numpy(z['hand_valid'])[None].to(device)
    mask = torch.ones(lm.shape[:2], dtype=torch.bool, device=device)
    with torch.no_grad():
        return model(lm, emb, valid, mask)[0].float().cpu()


def assign(tokens, words):
    """tokens [(letter, time)] -> per-word letter strings (nearest word whose padded window holds it)."""
    out = [''] * len(words)
    outside = 0
    for ch, t in tokens:
        hits = [i for i, w in enumerate(words) if w['start'] - PAD <= t <= w['end'] + PAD]
        if not hits:
            outside += 1
            continue
        i = min(hits, key=lambda i: abs(t - (words[i]['start'] + words[i]['end']) / 2))
        out[i] += ch
    return out, outside


def nll(lp, strings):
    """CTC negative log-likelihood of each string on log-probs lp [T, C]."""
    ids = [[1 + LETTERS.index(c) for c in s] for s in strings]
    T = lp.shape[0]
    keep = [i for i, x in enumerate(ids) if x]
    res = [float('inf')] * len(strings)
    if not keep:
        return res
    x = lp[:, None].expand(T, len(keep), lp.shape[1])
    loss = F.ctc_loss(x, torch.cat([torch.tensor(ids[i]) for i in keep]), torch.full((len(keep),), T),
                      torch.tensor([len(ids[i]) for i in keep]), blank=0, reduction='none', zero_infinity=False)
    for j, i in enumerate(keep):
        res[i] = float(loss[j])
    return res


class Lexicon:
    def __init__(self, path=LEXICON):
        self.words = open(path).read().split()

    def candidates(self, s, k=40):
        from rapidfuzz import process
        from rapidfuzz.distance import Levenshtein
        if not s:
            return []
        limit = max(1, min(3, len(s) // 3 + 1))
        hits = process.extract(s, self.words, scorer=Levenshtein.distance, limit=k, score_cutoff=limit)
        return [h[0] for h in hits]


def asllrp_records(model, device, lexicon):
    """Per spelled word: signer, ref, greedy hyp, baseline hyp, and (best lexicon candidate, NLL gap)."""
    manifest = {Path(c['clipFilename']).stem: c for c in json.loads((FEAT / 'manifest.json').read_text())['clips']}
    records, other = [], dict(model=[0, 0.], baseline=[0, 0.])
    for f in sorted(glob.glob(str(FEAT / 'features/*.npz'))):
        z = np.load(f)
        clip = manifest[Path(f).stem]
        words, times = clip['words'], z['times']
        lp = run_model(model, z, device)
        _, toks = greedy(lp)
        mid = lambda a, b: times[min(len(times) - 1, (a + b) // 2)]
        model_tokens = [(CLASSES[c], mid(a, b)) for c, a, b in toks if CLASSES[c] in LETTERS]
        base_tokens = [(ch, (t0 + t1) / 2) for ch, (t0, t1) in zip(str(z['letter_output']), z['letter_times'])]
        hyp, out_m = assign(model_tokens, words)
        bhyp, out_b = assign(base_tokens, words)
        covered = sum(min(w['end'] + PAD, times[-1]) - max(w['start'] - PAD, times[0]) for w in words)
        rest_min = max(0., (times[-1] - times[0]) - covered) / 60
        other['model'][0] += out_m; other['baseline'][0] += out_b
        other['model'][1] += rest_min; other['baseline'][1] += rest_min
        for w, h, b in zip(words, hyp, bhyp):
            ref = re.sub('[^A-Z]', '', w['gloss'][3:].upper())
            rec = dict(signer=clip['signerId'], ref=ref, hyp=h, base=b, seconds=w['end'] - w['start'])
            if lexicon is not None and h:
                sel = (times >= w['start'] - PAD) & (times <= w['end'] + PAD)
                cands = [c for c in lexicon.candidates(h) if c != h]
                if cands:
                    scores = nll(lp[sel], [h] + cands)
                    j = int(np.argmin(scores[1:])) + 1
                    rec['lex'] = cands[j - 1]; rec['lex_gap'] = scores[j] - scores[0]
            records.append(rec)
    return records, other


def summarize(records, key, margin=None):
    edits = total = exact = 0
    for r in records:
        h = r[key]
        if margin is not None and 'lex' in r and r['lex_gap'] <= margin:
            h = r['lex']
        e, n = cer_counts(h, r['ref'])
        edits += e; total += n; exact += h == r['ref']
    return dict(words=len(records), cer=round(edits / max(1, total), 4), exact=round(exact / max(1, len(records)), 4))


def fsboard_val(model, device):
    from scripts.train_letter_ctc_v17 import load, plain, collate
    import json as _json
    items = load(sorted(glob.glob(str(ROOT / 'data/local/fsboard_v17/batch1/features/validation/*.npz'))))
    kind = {Path(r['clipFilename']).stem: r['kind'] for r in
            _json.loads((ROOT / 'data/local/fsboard_v17/batch1/manifest.json').read_text())['clips']}
    acc = {}
    for it in items:
        lp = run_model(model, np.load(it['file']), device)
        hyp, _ = greedy(lp)
        e, n = cer_counts(letters_only(hyp), letters_only(it['phrase']))
        for k in ('all', kind[Path(it['file']).stem]):
            a = acc.setdefault(k, [0, 0, 0, 0]); a[0] += e; a[1] += n; a[2] += letters_only(hyp) == letters_only(it['phrase']); a[3] += 1
    return {k: dict(clips=v[3], cer=round(v[0] / v[1], 4), exact=round(v[2] / v[3], 4)) for k, v in acc.items()}


def fswild(model, device, split):
    """ChicagoFSWild partition (natural in-the-wild spelling, signer-disjoint): letter CER and exact rate."""
    from scripts.train_letter_ctc_v17 import load
    items = load(sorted(glob.glob(str(ROOT / f'data/local/chicago_fswild/features/{split}/*.npz'))))
    e = n = exact = 0
    for it in items:
        hyp, _ = greedy(run_model(model, np.load(it['file']), device))
        h, r = letters_only(hyp), letters_only(it['phrase'])
        de, dn = cer_counts(h, r); e += de; n += dn; exact += h == r
    return dict(clips=len(items), cer=round(e / n, 4), exact=round(exact / len(items), 4))


def user_gelo(model, device):
    """The user's 5 desktop GELO sessions (artifacts/generated/letter_ctc_v17/user_gelo_sessions): letter runs
    split at 1 s gaps; counts of runs equal to GELO/ANGELO and within one edit. Qualitative: no attempt times."""
    out = []
    for f in sorted(glob.glob(str(ROOT / 'artifacts/generated/letter_ctc_v17/user_gelo_sessions/*.npz'))):
        z = np.load(f); t = z['times']
        grid = np.arange(t[0], t[-1], .05); idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
        zz = {k: z[k][idx] for k in ('raw', 'hand_embeddings', 'hand_valid')}
        _, toks = greedy(run_model(model, zz, device))
        runs = []
        for c, a, b in toks:
            if CLASSES[c] not in LETTERS:
                continue
            tt = grid[min(len(grid) - 1, (a + b) // 2)]
            if runs and tt - runs[-1][1] < 1.:
                runs[-1] = (runs[-1][0] + CLASSES[c], tt)
            else:
                runs.append((CLASSES[c], tt))
        out += [r for r, _ in runs]
    near = [r for r in out if min(cer_counts(r, 'GELO')[0], cer_counts(r, 'ANGELO')[0]) <= 1]
    return dict(runs=len(out), exact=sum(r in ('GELO', 'ANGELO') for r in out), within1=len(near))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoints', nargs='+', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='mps')
    a = ap.parse_args()
    lexicon = Lexicon()
    report = {}
    for ck in a.checkpoints:
        model, payload = load_checkpoint(ck, a.device)
        recs, other = asllrp_records(model, a.device, lexicon)
        dev = [r for r in recs if r['signer'] in DEV_SIGNERS]
        test = [r for r in recs if r['signer'] not in DEV_SIGNERS]
        dev_curve = {str(m): summarize(dev, 'hyp', m) for m in MARGINS}
        best_margin = min(MARGINS, key=lambda m: (dev_curve[str(m)]['cer'], m))
        name = ck.parent.name
        report[name] = dict(
            checkpoint=str(ck), epoch=payload['epoch'], train_clips=payload['train_clips'],
            fsboard_val=fsboard_val(model, a.device),
            fswild_dev=fswild(model, a.device, 'dev'), fswild_test=fswild(model, a.device, 'test'),
            user_gelo=user_gelo(model, a.device),
            asllrp_dev=dict(raw=summarize(dev, 'hyp'), baseline=summarize(dev, 'base'), lexicon_by_margin=dev_curve,
                            chosen_margin=best_margin),
            asllrp_test=dict(raw=summarize(test, 'hyp'), baseline=summarize(test, 'base'),
                             lexicon=summarize(test, 'hyp', best_margin),
                             multi_letter_raw=summarize([r for r in test if len(r['ref']) > 1], 'hyp'),
                             multi_letter_lexicon=summarize([r for r in test if len(r['ref']) > 1], 'hyp', best_margin),
                             multi_letter_baseline=summarize([r for r in test if len(r['ref']) > 1], 'base')),
            false_letters_per_min=dict(model=round(other['model'][0] / max(1e-6, other['model'][1]), 2),
                                       baseline=round(other['baseline'][0] / max(1e-6, other['baseline'][1]), 2),
                                       non_spelling_minutes=round(other['model'][1], 2)))
        (a.out.parent).mkdir(parents=True, exist_ok=True)
        (a.out.with_suffix(f'.{name}.records.json')).write_text(json.dumps(recs, indent=0))
        print(name, json.dumps(report[name]['asllrp_test']), flush=True)
    a.out.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
