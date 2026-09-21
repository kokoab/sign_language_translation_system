"""Compare saved clean-baseline train/validation fits; no optimization or test access."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from active.v17.approved_phrase_data_v17 import verify_manifest,digest
from active.v17.model_unified_streaming_ctc_v17 import load_unified_streaming_head
from scripts.train_clean_phrase_baseline_v17 import REPORT,RUN_MANIFEST,evaluate
from scripts.train_youtube_motion_pilot_v17 import align_tokens


def breakdown(rows,names):
    groups=defaultdict(list);refs=Counter();preds=Counter();matched=Counter()
    for r in rows:
        groups[(r['source'],tuple(r['expected']))].append(r)
        expected=[t for t in r['expected'] if t!=101];predicted=[t for t in r['predicted'] if t!=101]
        _,matches=align_tokens(expected,predicted)
        refs.update(expected);preds.update(predicted);matched.update(expected[i] for i,j in matches)
    phrases=[]
    for (source,target),items in sorted(groups.items()):
        edits=Counter()
        for r in items:
            c,_=align_tokens([t for t in r['expected'] if t!=101],[t for t in r['predicted'] if t!=101]);edits.update(c)
        tokens=sum(t!=101 for t in target)*len(items)
        phrases.append(dict(source=source,phrase=' '.join(names[t] for t in target),samples=len(items),
                            exact=sum(r['expected']==r['predicted'] for r in items),
                            blank_only=sum(not r['predicted'] for r in items),known_tokens=tokens,
                            known_wer=sum(edits.values())/max(tokens,1),**edits))
    signs={names[t]:dict(references=refs[t],predictions=preds[t],matched=matched[t],missed=refs[t]-matched[t],
                        unmatched_predictions=preds[t]-matched[t],recall=matched[t]/refs[t] if refs[t] else None)
           for t in sorted(set(refs)|set(preds))}
    return phrases,signs


@torch.inference_mode()
def run():
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required')
    torch.mps.set_per_process_memory_fraction(.35)
    verified=verify_manifest(RUN_MANIFEST);manifest=json.loads(RUN_MANIFEST.read_text())
    previous=json.loads((REPORT/'results.json').read_text())
    assert previous['manifest']['sha256']==verified['sha256']
    cache=torch.load(ROOT/manifest['evidence_cache'],map_location='cpu',weights_only=False)
    names={v+1:k for k,v in cache['labels'].items()};names[101]='OTHER'
    coverage={role:Counter(t for s in samples for t in s.targets if t!=101) for role,samples in cache['samples'].items()}
    output=dict(manifest=verified,training_started=False,test_accessed=False,
                known_class_coverage={role:len(c) for role,c in coverage.items()},
                validation_labels_missing_in_training={names[t]:n for t,n in coverage['validation'].items() if not coverage['train'][t]},
                results={})
    for seed,saved in previous['results'].items():
        path=ROOT/saved['checkpoint'];assert digest(path)==saved['checkpoint_sha256']
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['dataset_provenance']['sha256']==verified['sha256']
        model=load_unified_streaming_head(checkpoint,device='mps')
        result=dict(checkpoint=saved['checkpoint'],checkpoint_sha256=saved['checkpoint_sha256'],selected_epoch=saved['selected_epoch'])
        for role in ('train','validation'):
            metrics,rows=evaluate(model,cache['samples'][role])
            if role=='validation':
                assert rows==saved['predictions'] and metrics==saved['validation'], 'validation reproduction mismatch'
            phrases,signs=breakdown(rows,names)
            result[role]=dict(metrics=metrics,phrases=phrases,signs=signs,predictions=rows)
        result['wer_gap_by_source']={s:result['validation']['metrics']['by_source'][s]['known_wer']-result['train']['metrics']['by_source'][s]['known_wer']
                                     for s in result['validation']['metrics']['by_source']}
        output['results'][seed]=result
        print(json.dumps(dict(seed=seed,train=result['train']['metrics'],validation=result['validation']['metrics'],gap=result['wer_gap_by_source'])),flush=True)
        del model;torch.mps.empty_cache()
    (REPORT/'fit_diagnostic.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='results'}))


if __name__=='__main__':run()
