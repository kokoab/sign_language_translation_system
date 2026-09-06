"""Motion-conditioned contextual scoring of recognition alternatives.

This model receives landmark-encoder embeddings and scores or proposes gloss
sequences. Proposed sequences must also compete under the recognizer's CTC scores.
English surface rendering remains a separate consumer.
"""

from __future__ import annotations

import math
import random

import torch
from torch import nn
import torch.nn.functional as F


@torch.inference_mode()
def ctc_candidate_scores(log_probabilities, candidates):
    """Exact all-alignment CTC mass for comparable beam and motion proposals."""
    if log_probabilities.ndim != 2 or not len(log_probabilities) or not candidates:
        raise ValueError("nonempty time/class probabilities and candidate list required")
    if any(any(t <= 0 or t >= log_probabilities.shape[1] for t in c) for c in candidates):
        raise ValueError("candidate token outside nonblank vocabulary")
    scores = log_probabilities.float().cpu()[:, None].expand(-1, len(candidates), -1).contiguous()
    targets = torch.tensor([token for candidate in candidates for token in candidate], dtype=torch.long)
    lengths = torch.tensor([len(c) for c in candidates], dtype=torch.long)
    return -F.ctc_loss(scores, targets, torch.full_like(lengths, len(scores)), lengths,
                       blank=0, reduction="none", zero_infinity=False)


def mismatched_motion_indices(rows, seed=17102):
    """Same-source, different-label counterfactuals with nearest duration support.

    Background-only groups have no different label and retain their own evidence;
    those rows cannot establish semantic dependence and must be reported as such.
    """
    rng = random.Random(seed)
    indices = []
    for index, row in enumerate(rows):
        choices = [i for i, other in enumerate(rows) if other["source"] == row["source"]
                   and tuple(other["targets"]) != tuple(row["targets"])]
        if choices:
            choices.sort(key=lambda i: abs(math.log(len(rows[i]["evidence"]) / len(row["evidence"]))))
            indices.append(rng.choice(choices[:min(20, len(choices))]))
        else:
            indices.append(index)
    return indices


@torch.inference_mode()
def rerank_motion_candidates(model, evidence, alternatives, weight, *, log_probabilities=None):
    """Experimental final-utterance correction; never reinterpret an unknown span.

Blank or OTHER in the leading hypothesis disables correction. The scorer was
not reliable on the open-vocabulary development corpus, so a language prior must
not turn those observations into confident known words.
"""
    if not alternatives or weight <= 0 or not alternatives[0][0] or model.num_glosses + 1 in alternatives[0][0]:
        return alternatives
    if log_probabilities is not None:
        candidates = list(dict.fromkeys([prefix for prefix, _ in alternatives] + model.propose_candidates(evidence)))
        ctc_scores = ctc_candidate_scores(log_probabilities, candidates).tolist()
        alternatives = list(zip(candidates, ctc_scores))
    scores = model.score_candidates(evidence, [prefix for prefix, _ in alternatives]).cpu().tolist()
    return sorted([(prefix, score + weight * context) for (prefix, score), context in zip(alternatives, scores)],
                  key=lambda item: item[1], reverse=True)


class MotionContextScorer(nn.Module):
    def __init__(self, evidence_dim=1068, hidden_dim=128, num_glosses=100, dropout=.15):
        super().__init__()
        self.evidence_dim, self.hidden_dim, self.num_glosses = evidence_dim, hidden_dim, num_glosses
        self.eos_index = num_glosses + 2
        self.motion = nn.Sequential(nn.LayerNorm(evidence_dim), nn.Linear(evidence_dim, hidden_dim), nn.GELU())
        self.tokens = nn.Embedding(num_glosses + 3, hidden_dim, padding_idx=0)
        self.history = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, num_glosses + 3))
        self.dropout = nn.Dropout(dropout)

    def forward(self, evidence, lengths, previous_tokens):
        if evidence.ndim != 3 or evidence.shape[-1] != self.evidence_dim:
            raise ValueError("motion evidence contract mismatch")
        if lengths.shape != (len(evidence),) or (lengths < 1).any() or (lengths > evidence.shape[1]).any():
            raise ValueError("motion lengths outside sequence bounds")
        memory = self.motion(evidence)
        positions = torch.arange(memory.shape[1], device=memory.device, dtype=memory.dtype)
        frequencies = torch.exp(torch.arange(0, self.hidden_dim, 2, device=memory.device, dtype=memory.dtype)
                                * (-math.log(10000.) / self.hidden_dim))
        phase = positions[:, None] * frequencies[None]
        position = torch.stack((phase.sin(), phase.cos()), -1).flatten(1)
        memory = memory + .1 * position[None]
        query, _ = self.history(self.dropout(self.tokens(previous_tokens)))
        padding = torch.arange(memory.shape[1], device=memory.device)[None] >= lengths[:, None]
        observed, _ = self.attention(query, memory, memory, key_padding_mask=padding, need_weights=False)
        return self.output(torch.cat((query, observed), -1))

    @torch.inference_mode()
    def propose_candidates(self, evidence, beam_width=4, max_tokens=12):
        """Propose gloss sequences from motion, for scored evaluation only.

        These are unverified alternatives, not permission to speak a corrected
        sentence. Each must compete with the raw recognizer and unknown outcome.
        """
        if evidence.ndim != 2 or len(evidence) < 1 or beam_width < 1 or max_tokens < 1:
            raise ValueError("nonempty motion and positive decoding limits required")
        beams = [((), 0., False)]
        for _ in range(max_tokens):
            expanded = []
            for prefix, score, ended in beams:
                if ended:
                    expanded.append((prefix, score, True))
                    continue
                tokens = torch.tensor([[0, *prefix]], device=evidence.device)
                logits = self(evidence[None], torch.tensor([len(evidence)], device=evidence.device), tokens)[0, -1]
                probabilities = logits.log_softmax(-1)
                probabilities[0] = -torch.inf
                values, indices = probabilities.topk(min(beam_width, self.num_glosses + 2))
                for value, index in zip(values.tolist(), indices.tolist()):
                    expanded.append((prefix if index == self.eos_index else (*prefix, index),
                                     score + value, index == self.eos_index))
            beams = sorted(expanded, key=lambda item: item[1], reverse=True)[:beam_width]
            if all(ended for _, _, ended in beams):
                break
        return list(dict.fromkeys(prefix for prefix, _, _ in beams))

    @torch.inference_mode()
    def score_candidates(self, evidence: torch.Tensor, candidates: list[tuple[int, ...]]):
        if evidence.ndim != 2 or not len(evidence) or not candidates:
            raise ValueError("nonempty evidence and recognition candidates required")
        width = max(len(c) for c in candidates) + 1
        inputs = torch.zeros((len(candidates), width), dtype=torch.long, device=evidence.device)
        targets = torch.full_like(inputs, -100)
        for i, candidate in enumerate(candidates):
            if any(t < 1 or t > self.num_glosses + 1 for t in candidate):
                raise ValueError("candidate contains nonlexical/unknown token index")
            inputs[i, 1:len(candidate) + 1] = torch.tensor(candidate, device=evidence.device)
            targets[i, :len(candidate) + 1] = torch.tensor((*candidate, self.eos_index), device=evidence.device)
        logits = self(evidence[None].expand(len(candidates), -1, -1),
                      torch.full((len(candidates),), len(evidence), device=evidence.device), inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100, reduction="none")
        return -loss.reshape(len(candidates), width).sum(1)
