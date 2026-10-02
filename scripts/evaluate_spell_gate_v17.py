"""End-to-end evaluation: letter reader (letter_ctc_v17) + spelling gate (spell_gate_v17), no trigger.

A letter token is kept when its peak posterior >= 0.5 and the gate's mean probability over the token's
frames >= theta (theta = 0 means ungated). theta is chosen on ASLLRP dev (signer Cory) by a rule fixed
before results: the fewest false letters per minute whose dev CER is at most ungated dev CER + 0.03.

Measured at every theta: ASLLRP dev/test spelled-word CER and false letters per minute outside spelled
words (+-0.5 s); ASL Citizen validation signers (isolated signs, no spelling): letters per minute;
O5S5 LG (evaluation-only signer): letters per minute inside annotated signs, and FS spans with >= 1
letter; ChicagoFSWild test CER; the user's GELO sessions (exact GELO/ANGELO runs).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.letter_ctc_v17 import CLASSES, LETTERS, add_deltas, base_features, cer_counts, greedy, letters_only, load_checkpoint
from active.v17.spell_gate_v17 import load_gate
from scripts.evaluate_letter_ctc_v17 import DEV_SIGNERS, FEAT, PAD, assign, run_model
from scripts.train_spell_gate_v17 import o5_items

THETAS = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
PEAK = .5


def gate_probs(gate, z, device):
    base, pres = base_features(z['raw'])
    lm = torch.from_numpy(add_deltas(base, pres))[None].to(device)
    with torch.no_grad():
        return torch.sigmoid(gate(lm, torch.ones(lm.shape[:2], dtype=torch.bool, device=device)))[0].cpu().numpy()


def tokens(model, gate, z, device):
    """[(letter, time, peak, gate_mean)] for every emitted letter."""
    lp = run_model(model, z, device); g = gate_probs(gate, z, device); t = z['times']
    _, toks = greedy(lp)
    return [(CLASSES[c], t[min(len(t) - 1, (a + b) // 2)], float(lp[a:b + 1, c].exp().max()), float(g[a:b + 1].mean()))
            for c, a, b in toks if CLASSES[c] in LETTERS]


MODE = 'token'


def kept(toks, theta):
    """MODE token: each letter's own gate mean >= theta. run_max / run_mean: letters are grouped into runs
    (gaps < 1 s) and a whole run is kept when the max / mean of its letters' gate means >= theta, so the
    gate's switch-on delay does not clip a word's first letter."""
    toks = [x for x in toks if x[2] >= PEAK]
    if theta == 0:
        return [(ch, t) for ch, t, p, g in toks]
    if MODE == 'token':
        return [(ch, t) for ch, t, p, g in toks if g >= theta]
    out, run = [], []
    for x in toks + [None]:
        if x is not None and run and x[1] - run[-1][1] < 1.:
            run.append(x); continue
        if run:
            score = max(r[3] for r in run) if MODE == 'run_max' else float(np.mean([r[3] for r in run]))
            if score >= theta:
                out += [(r[0], r[1]) for r in run]
        run = [x] if x is not None else []
    return out


def as_dict(z):
    return {k: z[k] for k in ('raw', 'hand_embeddings', 'hand_valid', 'times')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--letters', type=Path, default=ROOT / 'artifacts/generated/letter_ctc_v17/I_wild_noemb_speed_s0/best.pt')
    ap.add_argument('--gate', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--mode', choices=('token', 'run_max', 'run_mean'), default='token')
    a = ap.parse_args()
    global MODE
    MODE = a.mode
    model, _ = load_checkpoint(a.letters, a.device)
    gate, gp = load_gate(a.gate, a.device)
    res = {th: {} for th in THETAS}

    # ASLLRP (dev / test by signer).
    man = {Path(c['clipFilename']).stem: c for c in json.loads((FEAT / 'manifest.json').read_text())['clips']}
    cache = []
    for f in sorted(glob.glob(str(FEAT / 'features/*.npz'))):
        z = np.load(f); c = man[Path(f).stem]; t = z['times']
        covered = sum(min(w['end'] + PAD, t[-1]) - max(w['start'] - PAD, t[0]) for w in c['words'])
        cache.append((c, tokens(model, gate, as_dict(z), a.device), max(0., t[-1] - t[0] - covered) / 60))
    for th in THETAS:
        for name, sel in (('asllrp_dev', lambda c: c['signerId'] in DEV_SIGNERS), ('asllrp_test', lambda c: c['signerId'] not in DEV_SIGNERS)):
            e = n = exact = words = out = 0; mins = 0.
            for c, toks, m in cache:
                if not sel(c):
                    continue
                hyp, o = assign(kept(toks, th), c['words']); out += o; mins += m
                for w, h in zip(c['words'], hyp):
                    ref = re.sub('[^A-Z]', '', w['gloss'][3:].upper()); de, dn = cer_counts(h, ref)
                    e += de; n += dn; exact += h == ref; words += 1
            res[th][name] = dict(cer=round(e / n, 4), exact=round(exact / words, 4), false_per_min=round(out / mins, 2))

    # ASL Citizen validation signers: isolated signs, every letter is false.
    cit = []
    for f in sorted(glob.glob(str(ROOT / 'data/local/citizen100_v17/live_features/val/*.npz'))):
        z = np.load(f); cit.append((tokens(model, gate, as_dict(z), a.device), (z['times'][-1] - z['times'][0]) / 60))
    for th in THETAS:
        res[th]['citizen_val_letters_per_min'] = round(sum(len(kept(tk, th)) for tk, _ in cit) / sum(m for _, m in cit), 2)

    # O5S5 LG (evaluation-only signer).
    for it in o5_items(eval_only=True):
        z = np.load(it['file']); y = it['y']; t = z['times']; toks = tokens(model, gate, as_dict(z), a.device)
        sign_min = (y == 0).sum() * .05 / 60
        fs_runs = np.split(np.where(y == 1)[0], np.where(np.diff(np.where(y == 1)[0]) > 1)[0] + 1) if (y == 1).any() else []
        for th in THETAS:
            k = kept(toks, th)
            idx = [min(len(t) - 1, int(np.searchsorted(t, tt))) for _, tt in k]
            in_sign = sum(y[i] == 0 for i in idx)
            hit = sum(any(t[r[0]] - .3 <= tt <= t[r[-1]] + .3 for _, tt in k) for r in fs_runs if len(r))
            res[th]['o5s5_lg'] = dict(letters_per_min_in_signs=round(in_sign / max(1e-6, sign_min), 2),
                                      fs_spans_with_letters=f'{hit}/{len(fs_runs)}', sign_minutes=round(sign_min, 2))

    # ChicagoFSWild test (natural spelling, unseen signers).
    from scripts.train_letter_ctc_v17 import load
    wild = []
    for it in load(sorted(glob.glob(str(ROOT / 'data/local/chicago_fswild/features/test/*.npz')))):
        wild.append((tokens(model, gate, as_dict(np.load(it['file'])), a.device), letters_only(it['phrase'])))
    for th in THETAS:
        e = n = 0
        for tk, ref in wild:
            de, dn = cer_counts(''.join(ch for ch, _ in kept(tk, th)), ref); e += de; n += dn
        res[th]['fswild_test_cer'] = round(e / n, 4)

    # User GELO sessions (resampled to 20 Hz).
    sess = []
    for f in sorted(glob.glob(str(ROOT / 'artifacts/generated/letter_ctc_v17/user_gelo_sessions/*.npz'))):
        z = np.load(f); t = z['times']; grid = np.arange(t[0], t[-1], .05); idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
        zz = {k: z[k][idx] for k in ('raw', 'hand_embeddings', 'hand_valid')} | {'times': grid}
        sess.append(tokens(model, gate, zz, a.device))
    for th in THETAS:
        runs = []
        for tk in sess:
            r = []
            for ch, tt in kept(tk, th):
                if r and tt - r[-1][1] < 1.:
                    r[-1] = (r[-1][0] + ch, tt)
                else:
                    r.append((ch, tt))
            runs += [x for x, _ in r]
        res[th]['user_gelo'] = dict(runs=len(runs), exact=sum(x in ('GELO', 'ANGELO') for x in runs),
                                    within1=sum(min(cer_counts(x, 'GELO')[0], cer_counts(x, 'ANGELO')[0]) <= 1 for x in runs))

    base_cer = res[0]['asllrp_dev']['cer']
    ok = [th for th in THETAS if res[th]['asllrp_dev']['cer'] <= base_cer + .03]
    chosen = min(ok, key=lambda th: (res[th]['asllrp_dev']['false_per_min'], th))
    report = dict(mode=MODE, letters=str(a.letters), gate=str(a.gate), gate_val=gp['val'], peak=PEAK, chosen_theta=chosen,
                  rule='min ASLLRP-dev false letters/min with dev CER <= ungated + 0.03', by_theta={str(k): v for k, v in res.items()})
    a.out.write_text(json.dumps(report, indent=1))
    for th in THETAS:
        r = res[th]
        print(f"theta {th:.1f}{' *' if th == chosen else '  '} ASLLRP dev {r['asllrp_dev']['cer']:.3f}/{r['asllrp_dev']['false_per_min']:5.1f}  "
              f"test {r['asllrp_test']['cer']:.3f}/{r['asllrp_test']['false_per_min']:5.1f}  citizen {r['citizen_val_letters_per_min']:5.1f}/min  "
              f"LG signs {r['o5s5_lg']['letters_per_min_in_signs']:5.1f}/min FS {r['o5s5_lg']['fs_spans_with_letters']}  "
              f"FSWild {r['fswild_test_cer']:.3f}  GELO {r['user_gelo']['exact']}/{r['user_gelo']['within1']}", flush=True)


if __name__ == '__main__':
    main()
