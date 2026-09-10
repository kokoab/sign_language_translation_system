#!/usr/bin/env python3
"""Evaluate provisional sequence and immutable prefix behavior on matched dev inputs."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from active.v17.model_stage2_v17 import load_stage2_other_preserving
from active.v17.model_stage2_live_adapt_v17 import load_stage2_live_adapted
from active.v17.train_stage_2_v17 import RealPhraseDataset,edit_distance
from active.v17.streaming_ctc_prefix_v17 import StreamingCTCPrefix
from scripts.live_stage2_ctc_v17 import collapse_ctc_path,supported_ctc_path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/models/stage2_v17_transition_repair_v3/seed_1702.pth');p.add_argument('--adapted',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'artifacts/reports/stage2_v17_live_matched_v1/prefix_baseline.json');args=p.parse_args()
 torch.set_num_threads(2)
 model,payload=(load_stage2_live_adapted if args.adapted else load_stage2_other_preserving)(args.checkpoint);model.eval()
 cache=json.loads((ROOT/'artifacts/reports/stage2_v17_live_matched_v1/cache.json').read_text());details={r['item_id']:r for r in cache['rows']}
 if args.adapted:assert payload['input_contract']==cache['contract']
 labels=[r['canonical_label'] for r in sorted(json.loads((ROOT/'active/v17/citizen100_manifest.json').read_text())['classes'],key=lambda r:r['class_index'])]
 dataset=RealPhraseDataset(ROOT/'data/local/stage2_v17_live_matched_v1/frozen_features','validation');results=[]
 with torch.inference_mode():
  for row in dataset:
   state=StreamingCTCPrefix();updates=[];times=[]
   ends=[w['end'] for w in details[row.item_id]['windows'] if w['accepted']]
   for windows in range(1,len(row.features)+1):
    features=torch.from_numpy(row.features[:windows].astype(np.float32))[None];mask=torch.ones((1,windows),dtype=torch.bool)
    t=time.perf_counter();logits,lengths=model(features,mask);duration=time.perf_counter()-t
    tokens,positions=collapse_ctc_path(logits.numpy(),int(lengths[0]));tokens,positions=supported_ctc_path(tokens,positions)
    hyp=[labels[i-1] for i in tokens];status=state.update(hyp,list(positions),windows*8)
    updates.append(dict(windows=windows,source_end_seconds=ends[windows-1],hypothesis=hyp,positions=list(positions),**status));times.append(duration*1000)
   ref=[labels[i-1] for i in row.targets if i<=100];final=updates[-1];committed=final['committed']
   results.append(dict(item_id=row.item_id,source=row.source,reference=ref,provisional_final=final['hypothesis'],committed=committed,committed_is_reference_prefix=committed==ref[:len(committed)],conflicts=sum(u['conflict'] for u in updates),updates=updates,head_ms=times))
 metrics={}
 for source in sorted({r['source'] for r in results}):
  rs=[r for r in results if r['source']==source];tokens=sum(len(r['reference']) for r in rs)
  metrics[source]=dict(samples=len(rs),tokens=tokens,final_edits=sum(edit_distance(r['reference'],r['provisional_final']) for r in rs),final_exact=sum(r['reference']==r['provisional_final'] for r in rs),committed_edits=sum(edit_distance(r['reference'],r['committed']) for r in rs),committed_exact=sum(r['reference']==r['committed'] for r in rs),nonempty_committed=sum(bool(r['committed']) for r in rs),incorrect_committed_prefixes=sum(not r['committed_is_reference_prefix'] for r in rs),prefix_conflict_samples=sum(r['conflicts']>0 for r in rs),committed_tokens=sum(len(r['committed']) for r in rs))
 args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(dict(checkpoint=str(args.checkpoint),metrics=metrics,rows=results,disclosure='Matched development features; prefix timing in source-window units excludes extraction/scheduling. Empty prefixes are not successful recognition. Prefix conflicts and wrong commits are reported; no accuracy guarantee from agreement.'),indent=2)+'\n')
 print(json.dumps(metrics),flush=True)

if __name__=='__main__':main()
