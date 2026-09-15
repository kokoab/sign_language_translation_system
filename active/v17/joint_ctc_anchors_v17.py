"""Sparse positive alignment supervision; unverified frames have no CE target."""
import numpy as np
import torch
import torch.nn.functional as F


def event_anchors(times, events, targets):
    times=np.asarray(times,dtype=float)
    if (len(events)!=len(targets) or times.ndim!=1 or not len(times)
            or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
            or any(not 1<=t<=101 for t in targets)):
        raise ValueError('invalid anchor timestamps or CTC targets')
    intervals=np.array([(e['start_seconds'],e['end_seconds']) for e in events],dtype=float).reshape(-1,2)
    if not np.isfinite(intervals).all() or np.any(intervals[:,1]<=intervals[:,0]):
        raise ValueError('invalid event boundaries')
    masks=(times[None,:]>=intervals[:,0,None])&(times[None,:]<=intervals[:,1,None])
    unambiguous=masks.sum(0)==1
    result=[]
    for mask,(start,end),target in zip(masks,intervals,targets):
        indices=np.flatnonzero(mask&unambiguous)
        if len(indices):
            index=indices[np.argmin(np.abs(times[indices]-(start+end)/2))]
            result.append((int(index),int(target)))
    return result


def anchor_loss(logits, anchors, population):
    if logits.ndim!=3 or logits.shape[-1]!=102 or len(anchors)!=len(logits):
        raise ValueError('anchor batch must match [B,T,102] logits')
    selected=[];targets=[]
    for row,events in enumerate(anchors):
        for index,target in events:
            if not 0<=index<logits.shape[1] or not 1<=target<=101:
                raise ValueError('invalid anchor index or nonblank target')
            selected.append(logits[row,index]);targets.append(target)
    if not selected:
        return logits.sum()*0
    targets=torch.tensor(targets,device=logits.device)
    losses=F.cross_entropy(torch.stack(selected),targets,reduction='none')
    if len(population)!=2 or any(n<0 for n in population) or not sum(population):
        raise ValueError('expected known/OTHER training anchor populations')
    groups=[]
    for mask,count in zip((targets<101,targets==101),population):
        if mask.any() and not count:raise ValueError('anchor absent from declared population')
        if count:groups.append(losses[mask].sum()/count)
    return torch.stack(groups).mean()
