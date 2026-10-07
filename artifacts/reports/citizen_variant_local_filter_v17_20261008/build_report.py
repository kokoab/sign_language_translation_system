"""Build REPORT.md for the Citizen-variant local experiment from saved artifacts."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;OUT=HERE/'run'
assert json.loads((OUT/'status.json').read_text())['state']=='complete'
d=json.loads((OUT/'confusions.json').read_text());L=d.pop('labels');ix={l:i for i,l in enumerate(L)}
cm={m:{s:np.array(v) for s,v in d[m].items()} for m in d}
summary=json.loads((HERE/'manifest_summary.json').read_text())
recipe=json.loads((OUT/'recipe.json').read_text())
base_down=Path(recipe['baseline_downstream'])
def tot(m,s):c=cm[m][s];return int(np.trace(c)),int(c.sum())
AFFECTED=['HOME','CHILD','GOODBYE','HEAR','WHAT','BIG','SIGN','ASK','COME','I']
PAIRS=[('BIG','LANGUAGE'),('ASK','NEED'),('CHILD','HEAR'),('CHILD','GOODBYE'),('I','WE'),('WAIT','MAYBE'),('FAMILY','IMPORTANT'),
       ('LESS','SCHOOL'),('GO','ANSWER'),('ANGRY','NOW'),('HOME','HELLO'),('HEAR','LISTEN'),('GOOD','THANKYOU'),('COME','NEED')]
stages=[('Local adaptation','local'),('Multimodal fusion','fusion'),('Phrase adaptation','phrase'),('Interval recognizer','recognizer')]
lines=['# Citizen-variant local data experiment — 2026-10-08','',
 'User decision: for every confused class, the Citizen variant wins. Local clips of a different sign or',
 'variant were removed from local training **and** local validation; Citizen/SemLex data, recipes, seeds',
 'and selection rules are identical to the audited 96.83 chain (Variant C + approved phrase split).','',
 f"Removed local train clips: {summary['train']['removed']} (kept {summary['train']['kept']}); local val removed "
 f"{sum(summary['val']['removed'].values())} (kept {summary['val']['kept']}). I keeps only ME clips (index to chest).",
 'All rows below are scored on the same inputs: Citizen val 378, SemLex val 978, filtered local val '
 f"{summary['val']['kept']}. Validation only; the lab held-out test and protected tests were not used.",'',
 '## Accuracy by stage','','| Stage | Chain | Citizen | SemLex | Local (filtered) | Mean of 3 |','|---|---|---:|---:|---:|---:|']
for title,key in stages:
    for chain,m in (('current',f'base_{key}'),('Citizen-variant',f'new_{key}')):
        vals=[tot(m,s) for s in ('citizen_val','semlex_val','local_val')];pc=[100*a/b for a,b in vals]
        lines.append(f"| {title} | {chain} | {vals[0][0]}/{vals[0][1]} ({pc[0]:.2f}%) | {vals[1][0]}/{vals[1][1]} ({pc[1]:.2f}%) | {vals[2][0]}/{vals[2][1]} ({pc[2]:.2f}%) | {sum(pc)/3:.2f}% |")
vals=[tot('phone_recognizer',s) for s in ('citizen_val','semlex_val','local_val')];pc=[100*a/b for a,b in vals]
lines.append(f"| (phone today) | August local_a | {vals[0][0]}/{vals[0][1]} ({pc[0]:.2f}%) | {vals[1][0]}/{vals[1][1]} ({pc[1]:.2f}%) | {vals[2][0]}/{vals[2][1]} ({pc[2]:.2f}%) | {sum(pc)/3:.2f}% |")
lines+=['','## Affected classes, recognizer stage (Citizen + SemLex validation, correct/total)','','| Class | Phone today | Current chain | Citizen-variant chain |','|---|---:|---:|---:|']
for c in AFFECTED:
    i=ix[c];row=[]
    for m in ('phone_recognizer','base_recognizer','new_recognizer'):
        a=sum(int(cm[m][s][i,i]) for s in ('citizen_val','semlex_val'));b=sum(int(cm[m][s][i].sum()) for s in ('citizen_val','semlex_val'));row.append(f'{a}/{b}')
    lines.append(f'| {c} | '+' | '.join(row)+' |')
lines+=['','## Pair confusions, recognizer stage (A→B + B→A, all three validation sets)','','| Pair | Phone today | Current chain | Citizen-variant chain |','|---|---:|---:|---:|']
for a,b in PAIRS:
    row=[str(sum(int(cm[m][s][ix[a],ix[b]]+cm[m][s][ix[b],ix[a]]) for s in cm[m])) for m in ('phone_recognizer','base_recognizer','new_recognizer')]
    lines.append(f'| {a}/{b} | '+' | '.join(row)+' |')
def rec(path):
    h=json.loads((path/'span_recognizer/history.json').read_text())['history']
    import torch
    sel=torch.load(path/'span_recognizer/best_model.pth',map_location='cpu',weights_only=False)['span_adaptation']['selected']
    return h[0]['tune']['wer'],sel['epoch'],sel['tune']
b0,be,bt=rec(base_down);n0,ne,nt=rec(OUT)
lines+=['','## Continuous signing (tune pool, 89 clips, selection metric — not held-out)','',
 f"Current chain: epoch {be}, WER {100*b0:.2f}% → {100*bt['wer']:.2f}% (P {100*bt['precision']:.1f}, R {100*bt['recall']:.1f}).",
 f"Citizen-variant chain: epoch {ne}, WER {100*n0:.2f}% → {100*nt['wer']:.2f}% (P {100*nt['precision']:.1f}, R {100*nt['recall']:.1f}).",'']
lr=json.loads((OUT/'local_replay/result.json').read_text())
fu=json.loads((OUT/'fusion/result.json').read_text());ph=json.loads((OUT/'phrase_adapt/result.json').read_text())
sem=json.loads((OUT/'semlex_gate/summary.json').read_text())
lines+=['## Selections','',
 f"Local branch: epoch {lr['promotion_gate_candidate']['epoch']} (Citizen {lr['promotion_gate_candidate']['citizen_correct']}/378, floor 361; SemLex gate eval {round(sem['clip_top1']*978)}/978).",
 f"Fusion: seed {fu['selected_seed']} epoch {fu['selected_epoch']}. Phrase adaptation: epoch {ph['selected']['epoch']}, "
 f"phrase segments {ph['selected']['domains']['phrase']['top1']:.2f}%, activity {ph['selected']['domains']['phrase_activity']['top1']:.2f}%.",'',
 '## Limits','','- Local validation is familiar-signer and now omits the removed classes; it measures retention, not the phone.',
 '- Removed classes now rely on Citizen + SemLex only (≈20–35 clips each) — fewer examples in our capture conditions.',
 '- GOOD keeps the current (one-handed) data; the two-handed GOOD option is a separate decision.',
 '- Not exported or installed; the phone still runs the August recognizer with the new finish gesture.']
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
