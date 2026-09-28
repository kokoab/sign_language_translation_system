"""Fist-letter hand-geometry classifier (A E M N S T) for the letter refinement.

Train split only (label corrections applied); the validation split is the other signers used to
choose how it combines with the letter head (reports/fist_letters_v17_20260929/combine.py).
Output: a plain JSON logistic regression (standardiser + weights) that Python and Swift both read.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.fist_geometry_v17 import FIST, features, load_split

OUT = ROOT / 'artifacts/models/fist_geometry_v17/model.json'


def main():
    Xtr, ytr = load_split('train')
    Xva, yva = load_split('validation')
    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(max_iter=4000, C=1.0).fit(scaler.transform(Xtr), ytr)
    acc = float((model.predict(scaler.transform(Xva)) == yva).mean())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        format='slt_v17_fist_geometry', version=1, classes=list(model.classes_), mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(), coef=model.coef_.tolist(), intercept=model.intercept_.tolist(),
        features=len(scaler.mean_), train_clips=len(ytr), validation_clips=len(yva), validation_accuracy=acc,
        data='letter train split (label corrections applied); validation = other signers', test_accessed=False)))
    print(dict(train=len(ytr), validation=len(yva), validation_accuracy=round(acc, 4), classes=list(model.classes_)))


if __name__ == '__main__':
    main()
