# Matched contextual transition diagnostic

This replays the four selected checkpoints on the same cached, whole ASLLRP phrase windows used in the paired run. It does not retrain or change live inference.

## Coverage

- Rows: 56 ({'train': 44, 'validation': 12})
- Rows with eligible known cores: {'train': 30, 'validation': 11}
- Eligible known annotated cores: {'validation': 17, 'train': 59}

## Results

### Seed 17421 adapted (epoch 8)
- train: contextual core top-1 50/59 (84.7%); replay CTC exact 39/44; whole CTC known runs 86; gap-located known runs 3.
- validation: contextual core top-1 12/17 (70.6%); replay CTC exact 0/12; whole CTC known runs 13; gap-located known runs 1.
- FRIEND MAYBE: transcript `__OTHER__ MAYBE MAYBE`; contextual cores `FRIEND→YEAR, MAYBE→MAYBE`.

### Seed 17421 frozen (epoch 9)
- train: contextual core top-1 30/59 (50.8%); replay CTC exact 21/44; whole CTC known runs 83; gap-located known runs 2.
- validation: contextual core top-1 10/17 (58.8%); replay CTC exact 0/12; whole CTC known runs 14; gap-located known runs 2.
- FRIEND MAYBE: transcript `SCHOOL`; contextual cores `FRIEND→FRIEND, MAYBE→MAYBE`.

### Seed 17422 adapted (epoch 3)
- train: contextual core top-1 51/59 (86.4%); replay CTC exact 25/44; whole CTC known runs 72; gap-located known runs 2.
- validation: contextual core top-1 9/17 (52.9%); replay CTC exact 0/12; whole CTC known runs 14; gap-located known runs 2.
- FRIEND MAYBE: transcript `SCHOOL MAYBE`; contextual cores `FRIEND→WE, MAYBE→MAYBE`.

### Seed 17422 frozen (epoch 11)
- train: contextual core top-1 30/59 (50.8%); replay CTC exact 24/44; whole CTC known runs 76; gap-located known runs 4.
- validation: contextual core top-1 10/17 (58.8%); replay CTC exact 1/12; whole CTC known runs 11; gap-located known runs 1.
- FRIEND MAYBE: transcript `<blank>`; contextual cores `FRIEND→FRIEND, MAYBE→MAYBE`.

## Interpretation limits

- CTC argmax run clock is an approximate interpolation of each cached normalized window
- A core top-1 is contextual token-classifier evidence, not a separately extracted isolated-sign result or a claim that the encoder independently recognizes the core
- Known runs located in annotation-free core gaps are timing diagnostics, not certified transition false positives: a CTC spike can be delayed from an earlier sign
