"""FINGERSPELL-sign detector: window features on Apple Vision landmarks + a small classifier.

Train  positives: ASL Citizen FINGERSPELL train clips (27). Negatives: 6 Citizen train clips per sign of
       the 100 signs, and the letter train-split spans (fingerspelling itself must not trigger).
Held out (never used to fit or pick the threshold):
       positives: Citizen FINGERSPELL val (8) + ASLLRP in-sentence FINGERSPELL (2)
       negatives: all 378 Citizen *validation* clips of the 100 signs, the user's recorded sessions,
                  the letter validation spans.
Threshold: chosen by 5-fold cross-validation on the training clips for ~99% clip-level specificity.
"""
import gzip, json, pickle, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
WINDOW, STEP = 20, 2
TIPS, PIPS = [8, 12, 16, 20], [6, 10, 14, 18]


N_HAND = 18


def hand_features(raw, s):
    """N_HAND features of hand block s (0 or 21) over one window of raw [W,61,5] (isotropic xy)."""
    h = raw[:, s:s + 21]
    present = (h[:, :, 4] > 0).sum(1) >= 15
    f = np.zeros(N_HAND, np.float32)
    f[0] = present.mean()
    if present.sum() < 4:
        return f
    xy = h[present, :, :2]
    rel = xy - xy[:, :1]
    palm = np.linalg.norm(rel[:, 9], axis=1)
    palm_m = float(np.median(palm))
    good = palm > .6 * palm_m                                # drop collapsed / mis-tracked frames
    if good.sum() < 4 or palm_m < 1e-3:
        return f
    xy, rel, palm = xy[good], rel[good], palm[good]
    reln = rel / palm[:, None, None]
    tip_d = np.linalg.norm(reln[:, TIPS], axis=2); pip_d = np.linalg.norm(reln[:, PIPS], axis=2)
    ext = tip_d / np.maximum(pip_d, 1e-3)
    f[1] = ext.mean(); f[2] = ext.min(1).mean()                                     # finger extension
    v = np.diff(tip_d, axis=0)                                                      # per-finger length velocity
    f[3] = np.abs(v).mean()                                                         # wiggle amount
    f[4] = (np.diff(np.sign(v), axis=0) != 0).mean() if len(v) > 2 else 0          # wiggle reversals
    f[5] = v.std(1).mean()                                                          # asynchronous fingers
    wrist = xy[:, 0] / palm_m
    d = wrist[-1] - wrist[0]
    f[6], f[7] = d
    path = np.linalg.norm(np.diff(wrist, axis=0), axis=1).sum()
    f[8] = path; f[9] = np.linalg.norm(d) / max(path, 1e-3)                        # straightness
    f[10] = np.mean([np.linalg.norm(reln[:, a] - reln[:, b], axis=1).mean() for a, b in zip(TIPS, TIPS[1:])])
    direction = reln[:, 12] - reln[:, 9]
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
    f[11], f[12] = direction[:, 0].mean(), direction[:, 1].mean()
    f[13] = np.linalg.norm(reln[:, 4] - reln[:, 5], axis=1).mean()                  # thumb out
    body = raw[:, 57:59]
    ok = (body[:, :, 4] > 0).all(1)
    if ok.any():
        width = np.linalg.norm(body[ok, 1, :2] - body[ok, 0, :2], axis=1).mean()
        centre = body[ok, :, :2].mean(1).mean(0)
        if width > 1e-3:
            f[14] = (centre[1] - xy[:, 0, 1].mean()) / width
            f[15] = np.abs(xy[:, 0, 0].mean() - centre[0]) / width
            f[16] = palm_m / width
            f[17] = 1
    return np.clip(np.nan_to_num(f), -20, 20)


def window_features(raw):
    """[n_windows, N_HAND + 4]: the more present/moving hand, then a summary of the other hand."""
    raw = np.asarray(raw, np.float32)
    if len(raw) < WINDOW:
        raw = np.concatenate([raw, np.zeros((WINDOW - len(raw),) + raw.shape[1:], np.float32)])
    rows = []
    for start in range(0, len(raw) - WINDOW + 1, STEP):
        w = raw[start:start + WINDOW]
        a, b = hand_features(w, 0), hand_features(w, 21)
        if (b[0], b[8]) > (a[0], a[8]):
            a, b = b, a
        rows.append(np.concatenate([a, b[[0, 1, 3, 8]]]))
    return np.stack(rows)


def load_raw_dir():
    out = []
    for p in sorted((HERE / 'raw').glob('*.npz')):
        d = np.load(p)
        out.append((json.loads(str(d['meta'])), d['raw']))
    return out


def smooth_max(s):
    if len(s) >= 3:
        s = np.convolve(s, np.ones(3) / 3, mode='valid')                            # 3 consecutive windows (0.3 s)
    return float(s.max())


def main():
    import hashlib
    from active.v17.stage1_window_v17 import raw_observation_features
    clips = load_raw_dir()
    letters = {}
    for split in ('train', 'validation'):
        letters[split] = []
        for p in sorted((ROOT / 'data/local/fingerspelling_letters_v17/spans' / split).glob('*/*.npz')):
            with np.load(p) as z:
                if 'raw' in z.files:
                    letters[split].append(z['raw'])
    # training clips: (features, label, kind)
    train = [(window_features(r), 1, 'fs') for m, r in clips if m['kind'] == 'fingerspell_sign' and m['split'] == 'citizen_train']
    train += [(window_features(r), 0, 'sign') for m, r in clips if m['kind'] == 'sign']
    train += [(window_features(r), 0, 'letter') for r in letters['train']]
    held_pos = [(m['video'], window_features(r)) for m, r in clips if m['kind'] == 'fingerspell_sign' and m['split'] != 'citizen_train']
    val_neg = []
    for p in sorted((ROOT / 'data/local/citizen100_v17/raw/val').glob('*/*.mp4')):
        if p.name.startswith('._'):
            continue
        key = ROOT / 'artifacts/reports/letter_theft_v17_20260930/cache' / (hashlib.sha256(str(p).encode()).hexdigest()[:16] + '.pkl.gz')
        with gzip.open(key, 'rb') as fh:
            obs, _ = pickle.load(fh)
        if obs:
            val_neg.append((p.parent.name, p.name, window_features(raw_observation_features(obs)[0])))
    sessions = []
    for p in sorted((ROOT / 'artifacts/reports/letter_arbitration_v17_20260929/session_cache').glob('*.pkl.gz')):
        with gzip.open(p, 'rb') as fh:
            obs, _ = pickle.load(fh)
        raw, times = raw_observation_features(obs)
        sessions.append((p.name.split('.')[0], window_features(raw), times))
    val_letters = [window_features(r) for r in letters['validation']]
    print(dict(train_fs=sum(k == 'fs' for _, _, k in train), train_sign=sum(k == 'sign' for _, _, k in train),
               train_letter=sum(k == 'letter' for _, _, k in train), held_pos=len(held_pos), val_signs=len(val_neg),
               val_letters=len(val_letters), sessions=len(sessions)), flush=True)

    def windows(idx):
        X, y = [], []
        for i in idx:
            w, label, _ = train[i]
            if label:                                     # the sign itself: active hand in view
                w = w[w[:, 0] >= .6]
            X.append(w); y.append(np.full(len(w), label))
        return np.concatenate(X), np.concatenate(y)

    models = {'boosted': lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=15,
                                                                class_weight='balanced', random_state=0),
              'logistic': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=.3, class_weight='balanced'))}
    labels = np.array([l for _, l, _ in train]); kinds = np.array([k for _, _, k in train])
    report = {}
    for name, make in models.items():
        cv = np.zeros(len(train))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(train)), kinds):
            m = make().fit(*windows(tr))
            for i in te:
                cv[i] = smooth_max(m.predict_proba(train[i][0])[:, 1])
        sign_neg = np.sort(cv[kinds == 'sign'])
        tau = float(sign_neg[int(np.ceil(.995 * len(sign_neg))) - 1])   # <= 0.5% of training signs above
        model = make().fit(*windows(range(len(train))))
        score = lambda w: smooth_max(model.predict_proba(w)[:, 1])
        held = [(v, round(score(w), 3)) for v, w in held_pos]
        fired = sorted(((g, n, round(score(w), 3)) for g, n, w in val_neg), key=lambda r: -r[2])
        fp = [r for r in fired if r[2] > tau]
        sess, trig, minutes = {}, 0, 0.
        for n, w, times in sessions:
            s = model.predict_proba(w)[:, 1]
            s = np.convolve(s, np.ones(3) / 3, mode='same') if len(s) >= 3 else s
            events, last = [], -1e9
            for i in np.flatnonzero(s > tau):
                t = float(times[min(i * STEP + WINDOW - 1, len(times) - 1)])
                if t - last >= 2.0:
                    events.append(round(t, 1))
                last = t
            m_ = (times[-1] - times[0]) / 60
            sess[n] = dict(minutes=round(m_, 2), triggers=events); trig += len(events); minutes += m_
        report[name] = dict(threshold=tau,
                            cv_recall_train_fs=f"{int((cv[kinds == 'fs'] > tau).sum())}/{int((kinds == 'fs').sum())}",
                            cv_false_train_signs=f"{int((cv[kinds == 'sign'] > tau).sum())}/{int((kinds == 'sign').sum())}",
                            cv_false_train_letters=f"{int((cv[kinds == 'letter'] > tau).sum())}/{int((kinds == 'letter').sum())}",
                            held_out_recall=f"{sum(s > tau for _, s in held)}/{len(held)}", held_out=held,
                            citizen_val_false=f'{len(fp)}/{len(val_neg)}', citizen_val_false_clips=fp,
                            citizen_val_top=fired[:15],
                            letter_val_false=f"{sum(score(w) > tau for w in val_letters)}/{len(val_letters)}",
                            user_sessions=dict(minutes=round(minutes, 1), triggers=trig, sessions=sess))
        print(name, json.dumps({k: v for k, v in report[name].items() if k not in ('held_out', 'user_sessions', 'citizen_val_top', 'citizen_val_false_clips')}),
              '| sessions', trig, 'triggers in', round(minutes, 1), 'min', flush=True)
        print('  held', held, flush=True)
        print('  val false', fp, flush=True)
        with open(HERE / f'model_{name}.pkl', 'wb') as fh:
            pickle.dump(dict(model=model, threshold=tau, window=WINDOW, step=STEP), fh)
    (HERE / 'report.json').write_text(json.dumps(report, indent=1, default=str))


if __name__ == '__main__':
    main()
