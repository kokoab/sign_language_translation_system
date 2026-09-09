import sys,json,time
from pathlib import Path
from collections import Counter,defaultdict
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from active.v17.model_stage2_v17 import preserve_known_ctc_logits
from active.v17.train_stage_2_other_ctc_v17 import _metric_summary,collapse_ctc
torch.set_num_threads(2)
root=ROOT/'artifacts/models/stage2_v17_other_preservation_v1'
train=torch.load(root/'train_cache.pt',weights_only=False)
validation=torch.load(root/'validation_cache.pt',weights_only=False)
masses={'asllrp_other_ctc':.4,'local_phrases':.2,'asllrp_contiguous':.1,'isolated_citizen_train':.1,'asl_stem_wiki_verified_interval':.1,'asllrp_segmented_train':.1}
counts=Counter(r['source'] for r in train['rows'])
weights=torch.tensor([masses[r['source']]/counts[r['source']] for r in train['rows']])

def batch(cache,seed,indices,base_head):
 rows=[cache['rows'][i] for i in indices]
 x=pad_sequence([cache['models'][str(seed)]['values'][i] for i in indices],batch_first=True)
 known=pad_sequence([cache['models']['accepted']['values'][i] for i in indices],batch_first=True)
 lengths=torch.tensor([len(cache['models']['accepted']['values'][i]) for i in indices])
 targets=torch.tensor([t for r in rows for t in r['reference']]);sizes=torch.tensor([len(r['reference']) for r in rows])
 return rows,x,known,lengths,targets,sizes,base_head(x).logsumexp(-1,keepdim=True)

def evaluate(seed,head,base_head):
 domains=defaultdict(list)
 with torch.inference_mode():
  for start in range(0,len(validation['rows']),64):
   rows,x,known,lengths,targets,sizes,norm=batch(validation,seed,list(range(start,min(start+64,len(validation['rows'])))),base_head)
   logits=preserve_known_ctc_logits(known,head(x)-norm)
   for i,r in enumerate(rows):
    ref=[v-1 for v in r['reference']];pred=collapse_ctc(logits[i,:lengths[i]].argmax(-1).numpy())
    if r['source']=='asllrp_other_ctc':
     domains['target_full'].append(dict(r,reference=ref,prediction=pred))
     ref=[v for v in ref if v!=100];pred=[v for v in pred if v!=100]
    domains[r['source']].append(dict(r,reference=ref,prediction=pred))
 metrics={k:_metric_summary(v) for k,v in domains.items()}
 return metrics,domains

for seed in (1701,1702):
 torch.manual_seed(seed)
 payload=torch.load(ROOT/train['models'][str(seed)]['path'],weights_only=False,map_location='cpu')
 state=payload['model_state_dict'];head=nn.Linear(256,1);base_head=nn.Linear(256,101)
 with torch.no_grad():
  head.weight.copy_(state['ctc_head.weight'][101:]);head.bias.copy_(state['ctc_head.bias'][101:]);base_head.weight.copy_(state['ctc_head.weight'][:101]);base_head.bias.copy_(state['ctc_head.bias'][:101])
 base_head.requires_grad_(False)
 opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.02)
 history=[];best=None
 for epoch in range(21):
  if epoch:
   indices=torch.multinomial(weights,1800,replacement=True)
   total=0
   for start in range(0,len(indices),64):
    ids=indices[start:start+64].tolist();rows,x,known,lengths,targets,sizes,norm=batch(train,seed,ids,base_head)
    logits=preserve_known_ctc_logits(known,head(x)-norm)
    loss=nn.functional.ctc_loss(logits.log_softmax(-1).transpose(0,1),targets,lengths,sizes,zero_infinity=False)
    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step();total+=float(loss)
  metrics,domains=evaluate(seed,head,base_head)
  a=metrics
  passing=a['local_phrases']['edits']<=6 and a['asllrp_contiguous']['edits']<=9 and a['asllrp_segmented_validation']['edits']<=43 and a['isolated_citizen_validation']['exact']>=328 and a['asllrp_other_ctc']['edits']<=542 and a['asl_stem_wiki_verified_interval']['exact']>=15
  key=(passing,-a['asllrp_other_ctc']['edits'],-sum(a[s]['edits'] for s in ['local_phrases','asllrp_contiguous','asllrp_segmented_validation']))
  history.append({'epoch':epoch,'metrics':metrics,'passing':passing})
  print(seed,epoch,passing,{k:(v['edits'],v['exact']) for k,v in metrics.items()},flush=True)
  if best is None or key>best:
   best=key
   torch.save({'head_state_dict':head.state_dict(),'seed':seed,'epoch':epoch,'metrics':metrics},root/f'head_{seed}.pth')
   (root/f'predictions_{seed}.json').write_text(json.dumps(domains))
 (root/f'history_{seed}.json').write_text(json.dumps(history,indent=2))
