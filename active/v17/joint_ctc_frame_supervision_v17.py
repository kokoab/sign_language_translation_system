"""Supervise the actual sequence path where complete annotations verify a label."""
import numpy as np
import torch
import torch.nn.functional as F
from active.v17.joint_ctc_v17 import sequence_targets
from active.v17.live_transition_supervision_v17 import interior_gaps


def sequence_frame_targets(times, row, labels):
    targets=sequence_targets(row,labels)
    times=np.asarray(times,dtype=float)
    if times.ndim!=1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('invalid sequence frame timestamps')
    events=row['intervals']
    intervals=np.array([(e['start_seconds'],e['end_seconds']) for e in events],dtype=float).reshape(-1,2)
    if not np.isfinite(intervals).all() or np.any(intervals[:,1]<=intervals[:,0]):
        raise ValueError('invalid annotation intervals')
    masks=(times[None,:]>=intervals[:,0,None])&(times[None,:]<=intervals[:,1,None])
    unambiguous=masks.sum(0)==1
    output=np.full(len(times),-100,dtype=np.int64)
    for mask,target in zip(masks,targets):output[mask&unambiguous]=target
    # Reuse the existing 0.10-second guards; never label clip edges or O5S5 context.
    for start,end in interior_gaps(events,.10):output[(times>=start)&(times<=end)]=0
    return output


def sequence_frame_loss(logits, targets, population):
    if logits.ndim!=3 or logits.shape[-1]!=102 or len(logits)!=len(targets) or population<=0:
        raise ValueError('invalid sequence supervision batch or training frame population')
    padded=torch.full(logits.shape[:2],-100,dtype=torch.long,device=logits.device)
    for i,target in enumerate(targets):
        target=torch.as_tensor(target,device=logits.device)
        if target.ndim!=1 or len(target)>logits.shape[1] or not ((target==-100)|((target>=0)&(target<=101))).all():
            raise ValueError('invalid supervised frame targets')
        padded[i,:len(target)]=target
    return F.cross_entropy(logits.flatten(0,1),padded.flatten(),ignore_index=-100,reduction='sum')/population
