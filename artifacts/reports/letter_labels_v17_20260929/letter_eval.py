"""Letter heads compared on the validation letters (clean) and the corrected train G/Q clips.

Usage: letter_eval.py NAME HEAD [HEAD ...]   (PyTorch recognizer on CPU; geometry rule applied)
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from active.v17.segmental_runtime_v17 import build_runtime, refine_letter_geometry
from scripts.train_letter_head_v17 import corrected_labels
torch.set_num_threads(6)
rt = build_runtime(backend='torch', device='cpu')
rec = rt.recognizer
corr = corrected_labels()
LET = ROOT / 'data/local/fingerspelling_letters_v17/spans'
out = {}
for head in sys.argv[2:]:
    rec.attach_letters(ROOT / head)
    res = {}
    for split, letters in (('validation', None), ('train', 'GQ')):
        rows, batch = [], []
        def run():
            logits = rec.logits([b[0] for b in batch])
            for (x, raw, y), l in zip(batch, logits):
                raw_top = int(l[100:].argmax()); geo_top = int(refine_letter_geometry(l[100:], raw).argmax())
                rows.append((y, raw_top, geo_top))
        paths = sorted(LET.glob(f'{split}/*/*.npz'))
        for p in paths:
            letter = corr.get(str(p.relative_to(LET)), p.parent.name)
            if letters and letter not in letters:
                continue
            with np.load(p) as d:
                if 'landmarks' not in d: continue
                batch.append((tuple(d[k].astype(np.float32) if k != 'hand_valid' else d[k] for k in ('landmarks', 'hand_embeddings', 'hand_valid', 'hand_boxes')),
                              d['raw'].astype(np.float32), ord(letter) - 65))
            if len(batch) == 32: run(); batch = []
        if batch: run()
        y = np.array([r[0] for r in rows]); a = np.array([r[1] for r in rows]); g = np.array([r[2] for r in rows])
        per = {chr(65 + k): [int(((g == k) & (y == k)).sum()), int((y == k).sum())] for k in sorted(set(y.tolist()))}
        res[split] = dict(clips=len(rows), correct_raw=int((a == y).sum()), correct_with_geometry=int((g == y).sum()),
                          per_letter_with_geometry=per)
    out[head] = res
    print(head, {s: (r['clips'], r['correct_raw'], r['correct_with_geometry']) for s, r in res.items()},
          'G', res['validation']['per_letter_with_geometry'].get('G'), 'Q', res['validation']['per_letter_with_geometry'].get('Q'),
          'train G', res['train']['per_letter_with_geometry'].get('G'), 'train Q', res['train']['per_letter_with_geometry'].get('Q'), flush=True)
(Path(__file__).parent / f'letter_eval_{sys.argv[1]}.json').write_text(json.dumps(out, indent=1))
