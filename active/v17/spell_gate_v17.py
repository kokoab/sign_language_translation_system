"""Spelling-vs-signing gate: per-frame probability that the signer is fingerspelling right now.

Same camera-invariant landmark features and symmetric depthwise-convolution blocks as the letter reader
(active/v17/letter_ctc_v17.py), so it streams with the same fixed look-ahead. Letters from the reader are
kept only while the gate is on; no trigger sign or button is involved.
"""
from __future__ import annotations

import torch
from torch import nn

from active.v17.letter_ctc_v17 import LM_DIM, Block

FORMAT = 'spell_gate_v17_a'


class SpellGate(nn.Module):
    def __init__(self, d=128, layers=6, kernel=7, dropout=.15):
        super().__init__()
        self.config = dict(d=d, layers=layers, kernel=kernel, dropout=dropout)
        self.inp = nn.Sequential(nn.LayerNorm(LM_DIM), nn.Linear(LM_DIM, d))
        self.blocks = nn.ModuleList(Block(d, kernel, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)

    @property
    def lookahead_frames(self):
        return self.config['layers'] * (self.config['kernel'] // 2)

    def forward(self, lm, mask):
        """lm [B,T,LM_DIM], mask [B,T] bool -> logits [B,T]."""
        x = self.inp(lm)
        for b in self.blocks:
            x = b(x, mask)
        return self.out(self.norm(x)).squeeze(-1)


def load_gate(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != FORMAT:
        raise ValueError('not a spell_gate_v17 checkpoint')
    model = SpellGate(**payload['config'])
    model.load_state_dict(payload['state_dict'])
    return model.to(device).eval(), payload
