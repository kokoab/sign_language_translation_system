"""Verify downstream_recipe/ results and write downstream_recipe/REPORT.md with per-phase averages."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path('/Volumes/secret/SLT/SLT');HERE=Path(__file__).resolve().parent;OUT=HERE/'downstream_recipe'
sys.path.insert(0,str(ROOT))
from scripts import segmental_lab_v17 as lab
N=dict(citizen=378,semlex=978,local=2896)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def count(domain,top1):return int(round(top1*N[domain]/100))
assert json.loads((OUT/'status.json').read_text())['state']=='complete'
recipe_info=json.loads((OUT/'recipe.json').read_text())
recipe_path=ROOT/recipe_info['recipe_manifest'];assert sha(recipe_path)==recipe_info['recipe_sha256']
recipe=json.loads(recipe_path.read_text())
train={e['video_sha256'] for e in recipe['entries'] if e['split']=='train'}
held={r['video_sha256'] for r in lab.rows_for('tune')}|{r['video_sha256'] for r in lab.rows_for('test')}
audit={'protected_test_accessed':False,'lab_test_used':False,'chains':{}}
for name in ('chain_9683','august'):
    phrase=json.loads((OUT/name/'phrase_adapt/result.json').read_text())
    assert phrase['base_sha256']==recipe_info['chains'][name]['base_sha256']
    span_history=json.loads((OUT/name/'span_recognizer/history.json').read_text())
    prov=span_history['provenance']
    videos=set(prov['train_video_sha256'])
    assert videos<=train and not videos&held and prov['test_accessed'] is False
    checkpoint=torch.load(OUT/name/'span_recognizer/best_model.pth',map_location='cpu',weights_only=False)
    selected=checkpoint['span_adaptation']['selected']
    assert checkpoint['test_evaluated'] is False
    epochs=[r for r in span_history['history']]
    best_trained=min((r for r in epochs if r['epoch']>0 and r['eligible']),key=lambda r:r['tune']['wer'],default=None)
    audit['chains'][name]=dict(
        phrase=dict(base=phrase['base'],selected_epoch=phrase['selected']['epoch'],
                    baseline={k:v['top1_correct'] for k,v in phrase['baseline'].items()},
                    selected={k:v['top1_correct'] for k,v in phrase['selected']['domains'].items()},
                    samples={k:v['samples'] for k,v in phrase['selected']['domains'].items()}),
        recognizer=dict(selected_epoch=selected['epoch'],domains={d:count(d,v) for d,v in selected['domains'].items()},
                        tune=selected['tune'],floors=prov['floors'],train_videos=len(videos),span_examples=prov['span_examples'],
                        epoch0_tune_wer=epochs[0]['tune']['wer'],
                        best_trained_epoch=None if best_trained is None else dict(epoch=best_trained['epoch'],wer=best_trained['tune']['wer'])))
letters=torch.load(OUT/'chain_9683/letter_head_a/model.pth',map_location='cpu',weights_only=False)
old_letters=torch.load(ROOT/'artifacts/models/letter_head_v17_a/model.pth',map_location='cpu',weights_only=False)
def letter_summary(p):
    row=p['selected'];th=[t for t in row['thresholds'] if abs(t['tau']-p['threshold'])<1e-9]
    return dict(epoch=p['epoch'],threshold=p['threshold'],letter_top1_26way=row['letter_top1_26way'],**(th[0] if th else {}))
audit['letter_head']=dict(chain_9683=letter_summary(letters),august_head_a=letter_summary(old_letters))
export=json.loads((OUT/'export_check.json').read_text()) if (OUT/'export_check.json').is_file() else None
audit['exports']=export
phone=OUT.parent/'phone_recognizer'
audit['phone']=json.loads((phone/'device_samples.json').read_text()) if (phone/'device_samples.json').is_file() else None

# Per-phase table: earlier phases from audited chain reports.
c9683=json.loads((HERE/'chain_9683_floor361_mildroll/audit.json').read_text())
sel=c9683['local_replay']['selection']['candidates'][0]
fusion=c9683['fusion']['New (96.83 chain)'];aug_fusion=c9683['fusion']['August record']
rows=[('96.83 chain','Isolated landmark (96.83%)',366,853,1857),
      ('96.83 chain','Local adaptation',sel['citizen_correct'],sel['semlex_correct'],count('local',sel['local_top1'])),
      ('96.83 chain','Multimodal fusion',fusion['citizen'],fusion['semlex'],fusion['local'])]
for name,label in (('chain_9683','96.83 chain'),):
    a=audit['chains'][name]
    rows.append((label,'Phrase-segment adaptation',a['phrase']['selected']['citizen'],a['phrase']['selected']['semlex'],a['phrase']['selected']['local']))
    r=a['recognizer']['domains'];rows.append((label,'Interval recognizer',r['citizen'],r['semlex'],r['local']))
rows+=[('August chain','Isolated landmark (a7490409)',362,839,1765),('August chain','Local adaptation',361,860,2790),
       ('August chain','Multimodal fusion',aug_fusion['citizen'],aug_fusion['semlex'],aug_fusion['local'])]
a=audit['chains']['august']
rows.append(('August chain, approved split','Phrase-segment adaptation',a['phrase']['selected']['citizen'],a['phrase']['selected']['semlex'],a['phrase']['selected']['local']))
r=a['recognizer']['domains'];rows.append(('August chain, approved split','Interval recognizer',r['citizen'],r['semlex'],r['local']))
rows+=[('August chain, original leaking split','Phrase-activity adapted (reel_v2)',363,872,2810),
       ('August chain, original leaking split','Interval recognizer (local_a)',360,877,2789)]
audit['phase_table']=[dict(chain=c,phase=p,citizen=x,semlex=y,local=z) for c,p,x,y,z in rows]
(OUT/'audit.json').write_text(json.dumps(audit,indent=2,default=str)+'\n')

pct=lambda c,n:100*c/n
L=['# Downstream stages on the approved phrase split — verified results','',
   'Recipe: `active/v17/phrase_segment_recipe_manifest_20261007.json` (train = approved train ∩ lab train, 179 local phrases;',
   'validation = approved validation − lab test, 139; lab tune pool for recognizer selection; the 72-clip lab held-out test',
   'was never trained on, selected on or evaluated). Commands reproduce the August stages exactly (see reproductions/);',
   'only the base checkpoint, isolated caches and phrase split differ. Validation development evidence; no protected test access.','',
   '## Accuracy by phase across all validation sets','',
   '| Chain | Phase | Citizen (378) | SemLex (978) | Local (2,896) | Mean of 3 | Pooled (4,252) |','|---|---|---:|---:|---:|---:|---:|']
for c,p,x,y,z in rows:
    v=[pct(x,378),pct(y,978),pct(z,2896)]
    L.append(f'| {c} | {p} | {v[0]:.2f}% | {v[1]:.2f}% | {v[2]:.2f}% | {sum(v)/3:.2f}% | {pct(x+y+z,4252):.2f}% |')
L+=['','## Phrase-segment adaptation and interval recognizer','',
    '| Chain | Phrase epoch | Phrase segments | Activity crops | Recognizer epoch | Tune WER (epoch 0 → selected) | Floors (C/S/L) |','|---|---:|---:|---:|---:|---|---|']
for name,label in (('chain_9683','96.83 chain'),('august','August chain, approved split')):
    a=audit['chains'][name];ph=a['phrase'];rc=a['recognizer'];f=rc['floors']
    L.append(f"| {label} | {ph['selected_epoch']} | {ph['selected']['phrase']}/{ph['samples']['phrase']} ({pct(ph['selected']['phrase'],ph['samples']['phrase']):.2f}%) | "
             f"{ph['selected']['phrase_activity']}/{ph['samples']['phrase_activity']} ({pct(ph['selected']['phrase_activity'],ph['samples']['phrase_activity']):.2f}%) | "
             f"{rc['selected_epoch']} | {100*rc['epoch0_tune_wer']:.2f}% → {100*rc['tune']['wer']:.2f}% | {f['citizen']:.2f}/{f['semlex']:.2f}/{f['local']:.2f} |")
L+=['',f"Recognizer training: {audit['chains']['chain_9683']['recognizer']['train_videos']} approved-train videos, "
    f"{audit['chains']['chain_9683']['recognizer']['span_examples']} spans (August original: 282 videos incl. 103 approved-validation, 6,254 spans).",
    'Tune WER is the selection metric on the 89-clip tune pool (development evidence, not held-out WER).','']
lc,la=audit['letter_head']['chain_9683'],audit['letter_head']['august_head_a']
L+=['## Letter head (head-A recipe retrained on the new recognizer)','',
    f"New: 26-way top-1 {100*lc['letter_top1_26way']:.2f}%, recall {100*lc.get('letter_recall',float('nan')):.2f}% at τ={lc['threshold']}; "
    f"head A: {100*la['letter_top1_26way']:.2f}%, recall {100*la.get('letter_recall',float('nan')):.2f}% at τ={la['threshold']}.",'']
if export:
    L+=['## Core ML conversion (Mac, 378 Citizen validation clips, identical inputs)','','| Package | Size (MB) | PyTorch correct | Core ML correct | Top-5 | Agreement |','|---|---:|---:|---:|---:|---:|']
    for tag,e in export['exports'].items():
        L.append(f"| 96.83 chain {tag} | {e['package_mb']:.2f} | {e['pytorch_top1_correct']} | {e['coreml_top1_correct']} ({pct(e['coreml_top1_correct'],378):.2f}%) | {e['coreml_top5_correct']} | {e['top1_agreement']}/378 |")
    L.append(f"| August FP32 / FP16 (reference) | {export['august_package_mb']['FP32']:.2f} / {export['august_package_mb']['FP16']:.2f} | 360 | 360 / 359 | 374 / 374 | — |")
    L.append('')
if audit['phone']:
    import statistics
    L+=['## iPhone 13 timing (preparation + recognition per frame, ms)','','| Configuration | Run medians | Median of run medians | Thermal before/after |','|---|---|---:|---|']
    for conf in dict.fromkeys(r['configuration'] for r in audit['phone']):
        rr=[r for r in audit['phone'] if r['configuration']==conf]
        medians=' / '.join('%.2f' % r['median_ms'] for r in rr)
        L.append(f"| {conf} | {medians} | {statistics.median(r['median_ms'] for r in rr):.2f} | "
                 f"{','.join(str(r['thermal_before']) for r in rr)} / {','.join(str(r['thermal_after']) for r in rr)} |")
    L.append('')
L+=['Not claims: tune WER is development selection; the lab held-out test was not run for either rebuilt chain.',
    'Isolated SemLex validation reuses SemLex train signers; local validation is familiar-signer.']
(OUT/'REPORT.md').write_text('\n'.join(L)+'\n')
print('\n'.join(L))
