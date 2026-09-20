"""Small utterance-complete CTC head for the locked v17 vocabulary."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FinishCTCHead(nn.Module):
    """Use both sides of a finished utterance; expose blank + 100 + UNKNOWN."""

    def __init__(self, input_dim: int = 356, stage1_dim: int = 256,
                 hidden_dim: int = 128):
        super().__init__()
        if input_dim - stage1_dim != 100:
            raise ValueError("evidence must end with exactly 100 Stage-1 logits")
        self.stage1_dim = stage1_dim
        self.norm = nn.LayerNorm(input_dim)
        self.temporal = nn.GRU(input_dim, hidden_dim, batch_first=True,
                               bidirectional=True)
        self.blank = nn.Linear(hidden_dim * 2, 1)
        self.gloss_delta = nn.Linear(hidden_dim * 2, 100)
        self.unknown = nn.Linear(hidden_dim * 2, 1)
        nn.init.zeros_(self.gloss_delta.weight)
        nn.init.zeros_(self.gloss_delta.bias)

    def forward(self, evidence: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        if evidence.ndim != 3 or evidence.shape[-1] != self.norm.normalized_shape[0]:
            raise ValueError("invalid Finish CTC evidence shape")
        if len(lengths) != len(evidence) or any(n < 1 or n > evidence.shape[1] for n in lengths):
            raise ValueError("invalid Finish CTC lengths")
        packed = nn.utils.rnn.pack_padded_sequence(
            self.norm(evidence), lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.temporal(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            encoded, batch_first=True, total_length=evidence.shape[1])
        return torch.cat((self.blank(encoded),
                          evidence[..., self.stage1_dim:] + self.gloss_delta(encoded),
                          self.unknown(encoded)), dim=-1)


def ctc_loss(logits: torch.Tensor, targets: list[tuple[int, ...]],
             lengths: list[int]) -> torch.Tensor:
    """One CTC objective for phrases, single signs, and verified empty clips."""
    if logits.ndim != 3 or logits.shape[-1] != 102 or len(targets) != len(logits):
        raise ValueError("CTC requires [B,T,102] and one target per sequence")
    for target, length in zip(targets, lengths):
        required = len(target) + sum(a == b for a, b in zip(target, target[1:]))
        if required > length or length < 1 or length > logits.shape[1]:
            raise ValueError("infeasible CTC alignment")
        if any(value < 1 or value > 101 for value in target):
            raise ValueError("blank or invalid target in transcript")
    flat = torch.tensor([value for target in targets for value in target], dtype=torch.long)
    losses = F.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1).cpu(), flat,
                        torch.tensor(lengths), torch.tensor([len(t) for t in targets]),
                        blank=0, reduction="none", zero_infinity=False)
    scale = torch.tensor([max(1, len(t)) for t in targets])
    result = (losses / scale).mean().to(logits.device)
    if not torch.isfinite(result):
        raise ValueError("nonfinite CTC loss")
    return result


def decode(path) -> list[int]:
    """Collapse runs first, then hide blank and UNKNOWN from the public gloss tape."""
    output, previous = [], None
    for raw in path:
        value = int(raw)
        if value != previous and 1 <= value <= 100:
            output.append(value)
        previous = value
    return output


def error_rate(operations, reference_tokens: int) -> float:
    """Word error rate; alignment matches are evidence, not errors."""
    edits = sum(int(operations.get(name, 0))
                for name in ("substitution", "deletion", "insertion"))
    return 100.0 * edits / max(1, reference_tokens)
