"""Research-only direct text decoding from unpooled v17 Stage-1 features."""
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def epoch_batches(utterances, isolated, seed):
    if utterances < 1 or isolated < 1:
        raise ValueError('both training sources must be nonempty')
    rng = np.random.default_rng(seed)
    order = rng.permutation(utterances).tolist()
    translation = [order[i:i+2] for i in range(0, len(order), 2)]
    replay = np.array_split(rng.permutation(isolated), len(translation))
    return [(a, b.tolist()) for a, b in zip(translation, replay)]


class DirectTranslation(nn.Module):
    def __init__(self, base, text, prefix_ids):
        super().__init__()
        self.base = base
        self.text = text
        self.projection = nn.Linear(base.config.dim, text.config.d_model)
        self.register_buffer('prefix_ids', prefix_ids.long(), persistent=False)

    def visual_inputs(self, windows, valid, zero_visual=False):
        if not windows or len(windows) != len(valid):
            raise ValueError('one window-valid mask is required per utterance')
        for x, mask in zip(windows, valid):
            if x.ndim != 4 or tuple(x.shape[1:]) != (32,61,5):
                raise ValueError('expected [windows,32,61,5] Apple v17 features')
            if mask.shape != (len(x),) or mask.dtype != torch.bool or not mask.any():
                raise ValueError('each utterance needs at least one valid window')
        joined, usable = torch.cat(windows), torch.cat(valid)
        encoded, _ = self.base.encode(joined[usable])
        # Invalid windows retain their temporal positions without polluting encoder BN.
        all_encoded = encoded.new_zeros(len(joined),32,encoded.shape[-1])
        all_encoded = all_encoded.index_copy(0, usable.nonzero().flatten(), encoded)
        sequences = [part.flatten(0,1) for part in all_encoded.split([len(x) for x in windows])]
        embeddings = self.projection(pad_sequence(sequences, batch_first=True))
        if embeddings.shape[1]>256:
            raise ValueError('this experiment admits at most 256 visual tokens')
        if zero_visual:
            embeddings = torch.zeros_like(embeddings)
        mask = pad_sequence([v.repeat_interleave(32) for v in valid], batch_first=True).long()
        # Fixed masked padding bounds MPS graph variants; it discards no observations.
        embeddings=nn.functional.pad(embeddings,(0,0,0,256-embeddings.shape[1]))
        mask=nn.functional.pad(mask,(0,256-mask.shape[1]))
        prefix = self.text.get_input_embeddings()(self.prefix_ids).expand(len(windows),-1,-1)
        return dict(inputs_embeds=torch.cat((prefix,embeddings),dim=1),
                    attention_mask=torch.cat((mask.new_ones(len(windows),prefix.shape[1]),mask),dim=1))

    def translation_loss(self, windows, valid, labels):
        return self.text(**self.visual_inputs(windows,valid), labels=labels).loss

    def translate(self, windows, valid, *, max_new_tokens=100, num_beams=4, zero_visual=False):
        # Targets are intentionally absent from the generation API.
        return self.text.generate(**self.visual_inputs(windows,valid,zero_visual),
                                  max_new_tokens=max_new_tokens, num_beams=num_beams)
