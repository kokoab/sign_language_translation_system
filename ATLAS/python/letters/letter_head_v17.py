"""Supplemental static fingerspelling head (FS_A..FS_Z + NONE) on the frozen span recognizer.

The 100-sign recognizer is untouched: this head reads its landmark and hand features (and its
z-scored word logits) and says whether a span is a held letter. Letters are separate classes and
never alias lexical signs (letter I is not the sign I/ME).
"""
from __future__ import annotations

import torch
from torch import nn

LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
CLASSES = ['FS_' + c for c in LETTERS] + ['NONE']
NONE = len(CLASSES) - 1
FORMAT = 'slt_v17_letter_head'


def unified_features(model, landmarks, hand_embeddings, hand_valid, hand_boxes):
    """(landmark features, hand features, fused word logits) of a UnifiedMultimodalStage1V17."""
    landmark_logits, landmark_features = model.landmark_model(landmarks, return_embeddings=True)
    hand_features = model.hand_model.forward_features(hand_embeddings, hand_valid, hand_boxes)
    hand_logits = model.hand_model.classifier(hand_features)
    fused = model.fusion_head(landmark_features, hand_features, landmark_logits, hand_logits)
    return landmark_features, hand_features, fused


class LetterHead(nn.Module):
    def __init__(self, feature_dim=256, words=100, hidden=256, dropout=.2):
        super().__init__()
        self.config = dict(feature_dim=feature_dim, words=words, hidden=hidden, dropout=dropout)
        self.norm_l, self.norm_h = nn.LayerNorm(feature_dim), nn.LayerNorm(feature_dim)
        self.net = nn.Sequential(nn.Linear(2 * feature_dim + words, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, len(CLASSES)))

    def forward(self, landmark_features, hand_features, word_logits):
        z = (word_logits - word_logits.mean(-1, keepdim=True)) / word_logits.std(-1, keepdim=True).clamp_min(1e-6)
        return self.net(torch.cat([self.norm_l(landmark_features), self.norm_h(hand_features), z], -1))


class SpanWithLetters(nn.Module):
    """One graph for export: word logits (100) and letter logits (27) from the same inputs."""

    def __init__(self, recognizer, head):
        super().__init__()
        self.recognizer, self.head = recognizer, head

    def forward(self, landmarks, hand_embeddings, hand_valid, hand_boxes):
        lf, hf, words = unified_features(self.recognizer, landmarks, hand_embeddings, hand_valid, hand_boxes)
        return words, self.head(lf, hf, words)
