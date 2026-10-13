"""Reproducible summary; semantic validity is reviewed separately, not guessed by BLEU."""
import csv
import json
import statistics
from pathlib import Path
from replay import norm

HERE=Path(__file__).resolve().parent

def main():
    summary=[];comparisons=[]
    for model in ['tiny','flan','flan_fewshot','qwen','qwen_direct']:
        path=HERE/f'{model}_replay.json'
        if not path.exists():continue
        replay=json.loads(path.read_text())
        outputs=json.loads((HERE/f'{model}_outputs.json').read_text())
        for row in outputs: comparisons.append(dict(model=model,**row))
        for family in ['phone','diagnostic','stress']:
            for policy in ['finish','pause','revisable','adaptive']:
                rows=[r for r in replay if r['family']==family and r['policy']==policy]
                summary.append(dict(model=model,family=family,policy=policy,n=len(rows),
                    same_as_full=sum(r['full_equal'] for r in rows),
                    tail_finalized=sum(r['tail_finalized'] for r in rows),
                    conflicting_cases=sum(r['conflicts']>0 for r in rows),
                    median_first_caption_s=statistics.median(r['first_caption_seconds'] for r in rows),
                    median_last_caption_lag_s=statistics.median(r['last_caption_lag'] for r in rows),
                    max_queue_s=max(r['max_queue_seconds'] for r in rows),
                    mean_erasure=statistics.mean(r['normalized_erasure'] for r in rows)))
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (HERE/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    with (HERE/'outputs.csv').open('w') as f:
        columns=['model','id','family','glosses','text','mode','ms','reference_match']
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader()
        for row in comparisons:w.writerow(dict(row,glosses=' '.join(row['glosses'])))
    native=HERE.parents[1]/'artifacts/reports/stage3_shared_meanings_v17_20261011/phone_results.jsonl'
    old={r['id']:r['full'] for r in map(json.loads,native.open())}
    tiny=[r for r in comparisons if r['model']=='tiny' and r['family']=='phone']
    parity=dict(n=len(tiny),exact=sum(r['text']==old[r['id']] for r in tiny))
    assert parity==dict(n=26,exact=26),parity
    # Make an evidence index of every draft and commitment for review.
    with (HERE/'traces.csv').open('w') as f:
        fields=['model','id','family','policy','kind','n','at','ready','text']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for model in sorted({r['model'] for r in summary}):
            for row in json.loads((HERE/f'{model}_replay.json').read_text()):
                for event in row['events']:
                    w.writerow(dict(model=model,id=row['id'],family=row['family'],policy=row['policy'],**event))
    (HERE/'verification.json').write_text(json.dumps(dict(native_full_output_parity=parity,
        summary_rows=len(summary),cases_sha256=__import__('prepare').sha(HERE/'cases.json')),indent=2)+'\n')
    print('Native full-output parity:',parity)
    for model in sorted({r['model'] for r in summary}):
        rows=[r for r in comparisons if r['model']==model and r['reference_match'] is not None and r['family']=='diagnostic']
        print(model,'restricted diagnostic reference agreement',sum(r['reference_match'] for r in rows),'/',len(rows))

if __name__=='__main__':main()
