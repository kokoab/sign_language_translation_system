#!/usr/bin/env python3
"""Bounded temporal adaptation on matched live inputs with sequence retention."""
import argparse,copy,json,random,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from active.v17.model_stage2_v17 import load_stage2_other_preserving
from active.v17.model_stage2_live_adapt_v17 import Stage2LiveAdaptedCTCV17,make_stage2_live_adapted_checkpoint
from active.v17.train_stage_2_v17 import RealPhraseDataset,collate,Sample
from active.v17.train_stage_2_accuracy_repair_v17 import IsolatedPoolDataset
from active.v17.train_stage_2_other_ctc_v17 import CombinedDataset,evaluate_transition,evaluate,eligibility,collapse_ctc,sha256
from scripts.evaluate_stage2_other_preservation_v17 import validation_dataset

def transition_blank_loss(logits, batch, supervision):
    """Supervise annotated gap interiors only, never unknown signs or replay inputs."""
    terms = []
    for index, (item, source) in enumerate(zip(batch['item_ids'], batch['sources'])):
        if source.startswith('replay:') or item not in supervision:
            continue
        row = supervision[item]
        if row['role'] != 'train':
            raise ValueError('development annotation entered training: ' + item)
        positions = row['blank_positions']
        if positions:
            if min(positions) < 0 or max(positions) >= logits.shape[1]:
                raise ValueError('gap position exceeds input: ' + item)
            terms.append(-logits[index, positions].log_softmax(-1)[:, 0].mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def known_core_loss(logits, batch, supervision):
    """Counterbalance no-emission supervision with annotation-backed known sign cores."""
    terms = []
    for index, (item, source) in enumerate(zip(batch['item_ids'], batch['sources'])):
        if source.startswith('replay:') or item not in supervision:
            continue
        row = supervision[item]
        if row['role'] != 'train':
            raise ValueError('development core entered training: ' + item)
        cores = row.get('known_core_positions', {})
        if cores:
            positions = [int(p) for p in cores]
            labels = [int(v) for v in cores.values()]
            if min(positions) < 0 or max(positions) >= logits.shape[1] or not all(1 <= v <= 100 for v in labels):
                raise ValueError('invalid known core target: ' + item)
            terms.append(torch.nn.functional.cross_entropy(logits[index, positions], torch.tensor(labels)))
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def prefix_ctc_loss(model, batch, supervision, criterion, step):
    features, targets, masks = [], [], []
    for index, (item, source) in enumerate(zip(batch['item_ids'], batch['sources'])):
        if source.startswith('replay:') or item not in supervision:
            continue
        row = supervision[item]
        if row['role'] != 'train':
            raise ValueError('development prefix entered training: ' + item)
        prefixes = row.get('prefix_targets', {})
        ends = sorted(int(k) for k in prefixes if int(k) < int(batch['window_mask'][index].sum()))
        if not ends:
            continue
        windows = ends[step % len(ends)]
        target = prefixes[str(windows)]
        value = batch['features'][index].clone()
        mask = batch['window_mask'][index].clone()
        value[windows:] = 0
        mask[windows:] = False
        features.append(value); masks.append(mask); targets.append(target)
        if len(features) == 4:
            break
    if not features:
        return batch['features'].new_zeros(())
    logits, lengths = model(torch.stack(features), torch.stack(masks))
    return criterion(logits.log_softmax(-1).transpose(0, 1),
        torch.tensor([v for seq in targets for v in seq], dtype=torch.long), lengths,
        torch.tensor([len(seq) for seq in targets], dtype=torch.long))

def matched_evaluate(model, loader):
    metrics = evaluate(model, loader, torch.device('cpu'))
    summary = {}
    for name, source in [('local','local_phrases'),('exact','asllrp_contiguous'),('target','asllrp_other_ctc')]:
        domain = metrics['target_only'][source]
        summary[name+'_edits'] = domain['edits']
        summary[name+'_tokens'] = domain['tokens']
    return dict(summary=summary, domains=metrics)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--epochs',type=int,default=12);p.add_argument('--seed',type=int,default=17101);p.add_argument('--samples-per-epoch',type=int,default=2048)
 p.add_argument('--output-dir',type=Path);p.add_argument('--transition-supervision',type=Path)
 p.add_argument('--boundary-weight',type=float,default=.25);p.add_argument('--prefix-weight',type=float,default=.25)
 p.add_argument('--core-weight',type=float,default=0.)
 args=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
 report=ROOT/'artifacts/reports/stage2_v17_live_matched_v1';out=args.output_dir or ROOT/f'artifacts/models/stage2_v17_live_adapt_v1/seed_{args.seed}';out.mkdir(parents=True,exist_ok=True)
 if args.transition_supervision and args.output_dir is None:raise ValueError('supervised experiment requires separate --output-dir')
 supervision = {} if args.transition_supervision is None else json.loads(args.transition_supervision.read_text())['items']
 cache=json.loads((report/'cache.json').read_text());frozen=json.loads((report/'frozen_inputs.json').read_text());assert len(cache['rows'])==len(frozen['rows'])==1647
 if args.transition_supervision:
  supervision_payload=json.loads(args.transition_supervision.read_text())
  assert supervision_payload['inputs']['cache_sha256']==sha256(report/'cache.json')
  assert supervision_payload['inputs']['frozen_inputs_sha256']==sha256(report/'frozen_inputs.json')
  assert all(supervision[r['item_id']]['role']==r['role'] for r in cache['rows'])
 assert all(r['feature_windows']>0 for r in cache['rows'])
 matchedroot=ROOT/'data/local/stage2_v17_live_matched_v1/frozen_features'
 train=RealPhraseDataset(matchedroot,'train');valid=RealPhraseDataset(matchedroot,'validation')
 assert len(train)==1313 and len(valid)==334
 assert not {s.item_id for s in train.samples}&{s.item_id for s in valid.samples}
 replaysets=[RealPhraseDataset(ROOT/'data/local/stage2_v17_frozen_features','train'),RealPhraseDataset(ROOT/'data/local/stage2_v17_asllrp_segmented_train_frozen_features','train'),RealPhraseDataset(ROOT/'data/local/stage2_v17_transition_adapt_v2/frozen_features','train'),IsolatedPoolDataset(ROOT/'data/local/stage2_v17_synthetic/citizen_train_isolated_pool.npz','isolated_citizen_train',augment_boundaries=False)]
 teacher,payload=load_stage2_other_preserving(ROOT/'artifacts/models/stage2_v17_transition_repair_v3/seed_1702.pth')
 replay=CombinedDataset(replaysets);samples=list(train.samples);teacher_targets={}
 print('Caching sequence teacher',len(replay),flush=True)
 with torch.inference_mode():
  for batch in DataLoader(replay,batch_size=16,collate_fn=collate):
   logits,lengths=teacher(batch['features'],batch['window_mask'])
   for item,logit,length in zip(batch['item_ids'],logits,lengths.tolist()):teacher_targets[item]=[i+1 for i in collapse_ctc(logit[:length].argmax(-1).numpy()) if i<100]
 for i in range(len(replay)):
  s=replay[i];samples.append(Sample(s.features,s.targets,'replay:'+s.source,s.item_id,s.target_sequence))
 counts=Counter(s.source for s in samples)
 masses={'local_phrases':.20,'asllrp_contiguous':.10,'asllrp_other_ctc':.30}
 replaysources=[s for s in counts if s.startswith('replay:')]
 masses.update({s:.40/len(replaysources) for s in replaysources})
 weights=[masses[s.source]/counts[s.source] for s in samples]
 sampler=WeightedRandomSampler(weights,args.samples_per_epoch,replacement=True)
 loader=DataLoader(samples,batch_size=16,sampler=sampler,collate_fn=collate)
 oldvalid=validation_dataset();oldloader=DataLoader(oldvalid,batch_size=16,collate_fn=collate)
 vloader=DataLoader(valid,batch_size=16,collate_fn=collate)
 baseline=matched_evaluate(teacher,vloader);oldbase=evaluate_transition(teacher,oldloader,torch.device('cpu'))
 model=Stage2LiveAdaptedCTCV17(copy.deepcopy(teacher))
 optimizer=torch.optim.AdamW([{'params':model.accepted.parameters(),'lr':1e-5},{'params':list(model.other_head.parameters())+[model.other_shift],'lr':1e-3}],weight_decay=1e-4)
 criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
 history=[];bestkey=None;started=time.monotonic()
 design=dict(seed=args.seed,epochs=args.epochs,samples_per_epoch=args.samples_per_epoch,input_contract=cache['contract'],source_counts=dict(counts),source_masses=masses,teacher_sha256=sha256(ROOT/'artifacts/models/stage2_v17_transition_repair_v3/seed_1702.pth'),loss='gold CTC +0.5 decoded-teacher-sequence CTC on original replay; no framewise distillation',stage1_frozen=True,additional_endpoint_augmentation=False,protected_test_accessed=False)
 if args.transition_supervision:
  design.update(transition_supervision=str(args.transition_supervision),supervision_sha256=sha256(args.transition_supervision),boundary_weight=args.boundary_weight,prefix_weight=args.prefix_weight,core_weight=args.core_weight,auxiliary_loss='blank cross entropy on annotated gap interiors + CTC on eligible observed prefixes + optional known-core cross entropy; no synthetic holds')
 (out/'design.json').write_text(json.dumps(design,indent=2)+'\n')
 for epoch in range(args.epochs+1):
  losses=[]
  if epoch:
   model.train()
   for batch_index,b in enumerate(loader):
    optimizer.zero_grad(set_to_none=True)
    logits,lengths=model(b['features'],b['window_mask']);lp=logits.log_softmax(-1).transpose(0,1)
    loss=criterion(lp,b['targets'],lengths,b['target_lengths'])
    if supervision:
     loss=loss+args.boundary_weight*transition_blank_loss(logits,b,supervision)
     if args.core_weight:loss=loss+args.core_weight*known_core_loss(logits,b,supervision)
     if args.prefix_weight:loss=loss+args.prefix_weight*prefix_ctc_loss(model,b,supervision,criterion,epoch+batch_index)
    indices=[i for i,s in enumerate(b['sources']) if s.startswith('replay:')]
    if indices:
     seq=[teacher_targets[b['item_ids'][i]] for i in indices]
     target=torch.tensor([v for s in seq for v in s],dtype=torch.long)
     loss=loss+.5*criterion(lp[:,indices],target,lengths[indices],torch.tensor([len(s) for s in seq],dtype=torch.long))
    assert torch.isfinite(loss),('nonfinite loss',epoch)
    loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
  model.eval();metrics=matched_evaluate(model,vloader);old=evaluate_transition(model,oldloader,torch.device('cpu'))
  a=metrics['summary'];b=old['summary'];passes=eligibility(b,oldbase['summary']) and b['stem_correct']>=15
  matchedpass=a['local_edits']<=baseline['summary']['local_edits'] and a['exact_edits']<=baseline['summary']['exact_edits'] and a['target_edits']<baseline['summary']['target_edits']
  key=(int(passes and matchedpass),-sum(a[k]/max(1,a[k.replace('edits','tokens')]) for k in ['local_edits','exact_edits','target_edits']))
  record=dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,matched=metrics,original=old,original_gates_pass=passes,matched_gates_pass=matchedpass,elapsed_seconds=time.monotonic()-started)
  history.append(record)
  checkpoint=make_stage2_live_adapted_checkpoint(model,input_contract=cache['contract']);checkpoint.update(epoch=epoch,seed=args.seed,design=design)
  torch.save(checkpoint,out/'last.pth')
  if bestkey is None or key>bestkey:
   bestkey=key;torch.save(checkpoint,out/'best.pth')
  (out/'history.json').write_text(json.dumps(dict(design=design,baseline=baseline,original_baseline=oldbase,epochs=history,best_key=bestkey),indent=2)+'\n')
  print('epoch',epoch,'loss',record['loss'],'matched',a,'old',b,'passes',passes,matchedpass,'secs',round(time.monotonic()-started),flush=True)
 print('completed',out,flush=True)

if __name__=='__main__':main()
