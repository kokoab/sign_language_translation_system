"""Direct CTC supervision for verified isolated signs and positive-only cores."""
from active.v17.joint_ctc_v17 import ctc_loss


def positive_ctc_loss(logits, targets):
    targets=targets.detach().cpu().tolist()
    if any(not 0 <= target < 100 for target in targets):
        raise ValueError('positive targets must use the frozen 100-class isolated indices')
    return ctc_loss(logits,[(target+1,) for target in targets],[logits.shape[1]]*len(targets))
