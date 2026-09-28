"""Out-of-sample check of refine_letter_geometry: the letter TRAIN split (other sessions).

The rule's thresholds were read off validation-split geometry (geometry.json), so the
validation result (1099 -> 1129) is in-sample. Only clips labelled G/Q/R/U can change
(the rule swaps within G<->Q and R<->U), so only those are run. No training, no tuning.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from active.v17.segmental_runtime_v17 import build_runtime, refine_letter_geometry
torch.set_num_threads(4)
rec = build_runtime(device='cpu').recognizer
rows, batch = [], []
def run():
    logits = rec.logits([b[0] for b in batch])
    for (values, raw, meta), l in zip(batch, logits):
        old = int(l[100:].argmax()); new = int(refine_letter_geometry(l[100:], raw).argmax())
        y = ord(meta['letter']) - 65
        rows.append(dict(id=meta['name'], expected=meta['letter'], before=rec.letter_classes[old],
                         after=rec.letter_classes[new], before_correct=old == y, after_correct=new == y))
for letter in 'GQRU':
    for p in sorted((ROOT / 'data/local/fingerspelling_letters_v17/spans/train' / letter).glob('*.npz')):
        with np.load(p) as d:
            if 'landmarks' not in d or 'raw' not in d: continue
            batch.append((tuple(d[k].astype(np.float32) if k != 'hand_valid' else d[k] for k in
                                ('landmarks', 'hand_embeddings', 'hand_valid', 'hand_boxes')),
                          d['raw'].astype(np.float32), json.loads(str(d['meta']))))
        if len(batch) == 16: run(); batch = []
if batch: run()
per = {L: dict(clips=sum(r['expected'] == L for r in rows),
               before=sum(r['expected'] == L and r['before_correct'] for r in rows),
               after=sum(r['expected'] == L and r['after_correct'] for r in rows)) for L in 'GQRU'}
summary = dict(split='train', clips=len(rows), before_correct=sum(r['before_correct'] for r in rows),
               after_correct=sum(r['after_correct'] for r in rows),
               fixes=sum(not r['before_correct'] and r['after_correct'] for r in rows),
               regressions=sum(r['before_correct'] and not r['after_correct'] for r in rows), per_letter=per)
(Path(__file__).parent / 'train_split_geometry_check.json').write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
print(json.dumps(summary), flush=True)
