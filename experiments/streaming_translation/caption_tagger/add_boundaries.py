"""Sentence-boundary supervision for every corpus sentence (live tagger v2).

Reordered sentences previously had all labels masked, so the live tagger never
learned that a clause-final WH sign or NO continues the sentence. Here their
`post` labels are filled: '' inside the sentence, the sentence's end punctuation
on its last sign. Insertion/form labels stay masked. Existing vocabularies are
kept unchanged (the finalizer models depend on their indices). Writes data_b.json.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from align import corpus_rows, row_sentences  # noqa: E402
from build_data import row_labels  # noqa: E402

D = json.loads((HERE / 'data.json').read_text())
qid = {v: i for i, v in enumerate(D['post_vocab'])}
split_rows = {'train': [], 'validation': []}
for r in corpus_rows():
    split_rows[r['split']].append(r)
filled = 0
for split in ('train', 'validation'):
    for enc, r in zip(D[split], split_rows[split]):
        assert [D['glosses'][i - 1] for i in enc['x']] == r['glosses']
        lab, _ = row_labels(r)
        k = 0
        for g, s in row_sentences(r):
            span = range(k, k + len(g)); k += len(g)
            if lab[span[0]] is not None:
                continue
            end = s.strip()[-1] if s.strip()[-1] in '.?!' else '.'
            for i in span:
                enc['post'][i] = qid['' if i != span[-1] else end]
                filled += 1
(HERE / 'data_b.json').write_text(json.dumps(D))
print('filled post labels', filled)
