"""Put alignment likelihood on a per-output-frame scale beside positive frame CE."""
import torch
from active.v17.joint_ctc_v17 import ctc_loss


def frame_normalized_ctc_loss(logits, targets, lengths):
    if len(logits)!=len(targets) or len(targets)!=len(lengths) or not len(targets):
        raise ValueError('CTC requires matching nonempty batch, targets and lengths')
    # Reuse the shared feasibility/finite checks and differentiable CPU CTC path.
    return torch.stack([ctc_loss(logits[i:i+1],[target],[length])*max(1,len(target))/length
                        for i,(target,length) in enumerate(zip(targets,lengths))]).mean()
