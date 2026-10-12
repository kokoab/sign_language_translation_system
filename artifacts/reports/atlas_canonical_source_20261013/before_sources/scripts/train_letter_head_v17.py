"""Train the supplemental static-letter head on frozen span-recognizer features.

Positives: local A-Z clips (train = unknown-session clips; validation = DWIGHT + numbered sessions).
Negatives (NONE): real sign spans the decoder proposes (local + ASLLRP training spans, including
out-of-vocabulary ASLLRP signs) and isolated sign clips. Validation negatives: every scored span in
the decoder tuning pool plus isolated validation clips. No held-out test data is used.
The acceptance threshold is chosen for <= 1% of tuning-pool spans being called letters.
"""
from __future__ import annotations

import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.8'
os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.6'
import argparse
import glob
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.export_unified_multimodal_coreml_v17 import load_model
from active.v17.letter_head_v17 import CLASSES, FORMAT, LETTERS, NONE, LetterHead, unified_features
from scripts import segmental_lab_v17 as lab

LETTER_ROOT = ROOT / 'data/local/fingerspelling_letters_v17/spans'
ISO = ROOT / 'artifacts/generated/unfrozen_phrase_adapt_v17'


CORRECTIONS = LETTER_ROOT.parent / 'label_corrections.json'


def corrected_labels():
    """Audited label fixes, keyed '<role>/<letter>/<file>.npz' (videos stay where they were)."""
    if not CORRECTIONS.exists():
        return {}
    return {k: v['now'] for k, v in json.loads(CORRECTIONS.read_text())['corrections'].items()}


def letter_rows(role, corrections=None):
    corrections = corrected_labels() if corrections is None else corrections
    for path in sorted(glob.glob(str(LETTER_ROOT / role / '*' / '*.npz'))):
        d = np.load(path)
        if 'landmarks' not in d:
            continue
        key = str(Path(path).relative_to(LETTER_ROOT))
        letter = corrections.get(key, Path(path).parent.name)
        yield ((d['landmarks'], d['hand_embeddings'], d['hand_valid'], d['hand_boxes']),
               CLASSES.index('FS_' + letter))


def user_rows(root, holdout, repeat):
    """The user's own letters harvested from recorded app sessions (scripts/harvest_user_letters_v17.py)."""
    rows = []
    for path in sorted(glob.glob(str(Path(root) / '*' / '*' / '*.npz'))):
        session = Path(path).parents[1].name
        if session in holdout:
            continue
        d = np.load(path)
        rows.append(((d['landmarks'], d['hand_embeddings'], d['hand_valid'], d['hand_boxes']),
                     CLASSES.index('FS_' + Path(path).parent.name)))
    return rows * repeat


def span_negatives(names, keep, rng):
    """Yield a random `keep` fraction of labelled decoder spans (streamed, memo by memo)."""
    for name in names:
        for row in lab.rows_for(name):
            path = lab.CACHE / 'train_labels' / (lab.key(row['source_item_id']) + '.json')
            if not path.exists():
                continue
            memo = lab.load_memo(row)
            for item in json.loads(path.read_text())['spans']:
                r = memo.get(tuple(item['span']))
                if r is not None and r.get('inputs') is not None and rng.random() < keep:
                    x = r['inputs']
                    yield ((x['landmarks'], x['hand_embeddings'], x['hand_valid'], x['hand_boxes']), NONE)
            del memo


def memo_negatives(name, keep, rng):
    for row in lab.rows_for(name):
        memo = lab.load_memo(row)
        for r in memo.values():
            if r is not None and r.get('inputs') is not None and rng.random() < keep:
                x = r['inputs']
                yield ((x['landmarks'], x['hand_embeddings'], x['hand_valid'], x['hand_boxes']), NONE)
        del memo


def isolated(split, cap, rng):
    out = []
    for source in ('citizen', 'semlex', 'local'):
        with np.load(ISO / f'{source}_{split}.npz', allow_pickle=False) as d:
            # Each d[key] access decompresses the whole array: read every array once.
            arrays = [d[k] for k in ('landmarks', 'hand_embeddings', 'hand_valid', 'hand_boxes')]
        idx = list(range(len(arrays[0])))
        rng.shuffle(idx)
        for i in idx[:cap]:
            out.append((tuple(a[i] for a in arrays), NONE))
        del arrays
    return out


@torch.inference_mode()
def featurize(model, rows, device, batch=256):
    """Stream inputs through the frozen recognizer; only compact features are kept."""
    feats, labels, chunk = [], [], []

    def run():
        x = [torch.from_numpy(np.stack([c[0][k] for c in chunk]).astype(np.float32 if k != 2 else bool)).to(device)
             for k in range(4)]
        lf, hf, w = unified_features(model, *x)
        feats.append(torch.cat([lf, hf, w], -1).float().cpu())
        labels.extend(c[1] for c in chunk)
    for item in rows:
        chunk.append(item)
        if len(chunk) == batch:
            run(); chunk = []
    if chunk:
        run()
    return torch.cat(feats), torch.tensor(labels)


def split(feats):
    return feats[:, :256], feats[:, 256:512], feats[:, 512:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recognizer', default='artifacts/models/span_recognizer_v17_local_a/best_model.pth')
    ap.add_argument('--output', type=Path, default=ROOT / 'artifacts/models/letter_head_v17_a/model.pth')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=28928)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--user-letters', default=None, help='harvested user session letters root')
    ap.add_argument('--holdout-session', action='append', default=[], help='user session never used for training')
    ap.add_argument('--user-repeat', type=int, default=4, help='oversampling of the user letters')
    ap.add_argument('--no-corrections', action='store_true', help='ignore label_corrections.json')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    device = a.device
    torch.set_num_threads(6)
    model, _ = load_model(Path(a.recognizer))
    model.to(device).eval()
    import itertools
    Xl, yl = featurize(model, letter_rows('train', {} if a.no_corrections else None), device)
    Xs, ys = featurize(model, span_negatives(('local_train', 'asllrp_train'), .6, rng), device)
    Xi, yi = featurize(model, iter(isolated('train', 1500, rng)), device)
    parts = [(Xl, yl), (Xs, ys), (Xi, yi)]
    if a.user_letters:
        user = user_rows(a.user_letters, set(a.holdout_session), a.user_repeat)
        if user:
            parts.append(featurize(model, iter(user), device))
        print(json.dumps(dict(user_letters=len(user), holdout=a.holdout_session, repeat=a.user_repeat)), flush=True)
    Xtr, ytr = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
    Xvl, yvl = featurize(model, letter_rows('validation'), device)
    Xvt, _ = featurize(model, memo_negatives('tune', .25, rng), device)
    Xvi, _ = featurize(model, iter(isolated('val', 100000, rng)), device)
    print(json.dumps(dict(train=len(ytr), train_letters=int((ytr != NONE).sum()), val_letters=len(yvl),
                          val_none_tune_spans=len(Xvt), val_none_isolated=len(Xvi))), flush=True)
    head = LetterHead().to(device)
    counts = torch.bincount(ytr, minlength=len(CLASSES)).float()
    weights = (counts.sum() / (len(CLASSES) * counts.clamp_min(1))).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    def probs(X):
        head.eval()
        with torch.inference_mode():
            return torch.softmax(head(*[t.to(device) for t in split(X)]), -1).cpu()

    def evaluate():
        pl, pt, pi = probs(Xvl), probs(Xvt), probs(Xvi)
        letter_top1 = float((pl[:, :NONE].argmax(1) == yvl).float().mean())
        rows = []
        for tau in (.5, .6, .7, .8, .9, .95):
            acc_l = (pl[:, :NONE].max(1).values >= tau) & (pl[:, :NONE].argmax(1) == yvl)
            rows.append(dict(tau=tau, letter_recall=float(acc_l.float().mean()),
                             tune_span_false_letter=float((pt[:, :NONE].max(1).values >= tau).float().mean()),
                             isolated_sign_false_letter=float((pi[:, :NONE].max(1).values >= tau).float().mean())))
        return dict(letter_top1_26way=letter_top1, thresholds=rows)

    best, history = None, []
    for epoch in range(1, a.epochs + 1):
        head.train()
        perm = torch.randperm(len(ytr))
        total = 0.
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            lf, hf, w = (t[idx].to(device) for t in split(Xtr))
            loss = F.cross_entropy(head(lf, hf, w), ytr[idx].to(device), weight=weights)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(idx)
        sched.step()
        row = dict(epoch=epoch, loss=total / len(ytr), **evaluate())
        history.append(row)
        # Selection: best letter recall at the smallest tau keeping tuning-span false letters <= 1%.
        ok = [t for t in row['thresholds'] if t['tune_span_false_letter'] <= .01]
        key = max((t['letter_recall'] for t in ok), default=0.)
        if best is None or key > best[0]:
            best = (key, epoch, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, row)
        if epoch % 5 == 0:
            print(json.dumps(dict(epoch=epoch, loss=round(row['loss'], 4), top1=round(row['letter_top1_26way'], 4),
                                  selection_key=round(key, 4))), flush=True)
    key, epoch, state, row = best
    ok = [t for t in row['thresholds'] if t['tune_span_false_letter'] <= .01]
    tau = min((t['tau'] for t in ok if t['letter_recall'] == key), default=.95)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format=FORMAT, config=head.config, state_dict=state, classes=CLASSES, threshold=tau,
                    recognizer=a.recognizer, epoch=epoch, selected=row,
                    data='letters: local A-Z (train unknown sessions; val DWIGHT+numbered); NONE: decoder spans + isolated'
                         + ('' if a.no_corrections else f'; label corrections {CORRECTIONS.name}')
                         + (f'; user session letters {a.user_letters} x{a.user_repeat} (holdout {a.holdout_session})' if a.user_letters else ''),
                    test_accessed=False), a.output)
    (a.output.parent / 'history.json').write_text(json.dumps(history, indent=1))
    print(json.dumps(dict(selected_epoch=epoch, threshold=tau, selected=row), indent=1))


if __name__ == '__main__':
    main()
