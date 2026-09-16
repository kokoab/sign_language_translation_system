"""One fixed existing beam-decoder control on cached emissions; no training/tuning."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from active.v17.continuous_decode_v17 import CTCPrefixDecoder
from active.v17.train_stage_2_other_ctc_v17 import _edit_operations
from scripts.live_stage2_ctc_v17 import collapse_ctc_path, ctc_sequence_log_probability

names = [r['canonical_label'] for r in sorted(json.loads(
    (ROOT/'active/v17/citizen100_manifest.json').read_text())['classes'], key=lambda r:r['class_index'])]
source = json.loads((HERE/'results.json').read_text())['results']
output = []
started = time.monotonic()
for row in source:
    path = HERE/row['item_id']/f'phase_{row["phase"]:.2f}'/'offline_emissions.npz'
    with np.load(path, allow_pickle=False) as archive:
        logits = archive['logits'].astype(np.float64)
    tokens, _ = collapse_ctc_path(logits, len(logits))
    greedy = [names[t-1] for t in tokens if t <= 100]
    assert greedy == row['offline_final'], 'Wrong saved emission stream'
    decoder = CTCPrefixDecoder(beam_width=8, token_topk=12)
    shifted = logits-logits.max(-1, keepdims=True)
    log_probs = shifted-np.log(np.exp(shifted).sum(-1, keepdims=True))
    for step in log_probs:
        decoder.step(step)
    beam = [names[t-1] for t in decoder.alternatives()[0][0] if t <= 100]
    result = dict(item_id=row['item_id'], phase=row['phase'], reference=row['reference'],
                  greedy=greedy, beam=beam, logits_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if row['reference'] is not None:
        result['edits'] = {k:sum(o['operation'] != 'match' for o in _edit_operations(row['reference'], v))
                           for k,v in [('greedy',greedy),('beam',beam)]}
    if row['item_id'] == 'o5s5_lg_when':
        token = names.index('WHEN')+1
        result['exact_log_probability_by_when_count'] = {
            str(count):ctc_sequence_log_probability(logits, (token,)*count) for count in (1,2,3)}
    output.append(result)
assert len(output) == 40
summary = dict(runs=len(output), changed_runs=sum(r['greedy'] != r['beam'] for r in output),
    labelled_phase_runs=sum(r['reference'] is not None for r in output),
    reference_tokens=sum(len(r['reference']) for r in output if r['reference'] is not None),
    greedy_edits=sum(r.get('edits',{}).get('greedy',0) for r in output),
    beam_edits=sum(r.get('edits',{}).get('beam',0) for r in output),
    elapsed_seconds=time.monotonic()-started,
    limitation='Four offsets of four labelled clips are correlated selected diagnostics, not a generalization benchmark.',
    training_performed=False, runtime_modified=False, protected_test_accessed=False)
(HERE/'beam_control.json').write_text(json.dumps(dict(summary=summary, rows=output),indent=2)+'\n')
print(json.dumps(summary))
for row in output:
    if row['item_id'] in {'webcam_eat','webcam_hello','o5s5_lg_when'}:
        print(json.dumps(row))
