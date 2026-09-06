"""Continuous landmark runtime with scored alternatives and revisable prefixes."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from active.v17.continuous_decode_v17 import CTCPrefixDecoder, RevisableTranscript
from active.v17.continuous_evidence_v17 import (
    ContinuousConfig, ContinuousEvidenceModel, observation_windows,
)
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config


class ContinuousRecognizer:
    def __init__(self, checkpoint: Path, *, device="cpu", beam_width=8, context_checkpoint: Path | None = None,
                 context_proposals=False):
        if context_proposals and context_checkpoint is None:
            raise ValueError("motion proposals require a matching context checkpoint")
        self.context_proposals = bool(context_proposals)
        stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if stored.get("format") != "slt_continuous_evidence_v17":
            raise ValueError("expected a continuous-evidence checkpoint")
        base = Path(stored["base"])
        if hashlib.sha256(base.read_bytes()).hexdigest() != stored["base_sha256"]:
            raise ValueError("continuous model's Stage-1 encoder changed")
        encoder = torch.load(base, map_location="cpu", weights_only=False)
        if encoder["label_to_index"] != stored["label_to_index"]:
            raise ValueError("encoder and decoder vocabularies differ")
        self.config = ContinuousConfig(**stored["config"])
        self.device = torch.device(device)
        self.encoder = SLTStage1V17(Stage1V17Config(**encoder["model_config"]))
        self.encoder.load_state_dict(encoder["model_state_dict"], strict=True)
        self.encoder.to(self.device).eval()
        self.model = ContinuousEvidenceModel(self.config)
        self.model.load_state_dict(stored["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.labels = {int(index) + 1: label for label, index in stored["label_to_index"].items()}
        self.labels[self.config.other_index] = "OTHER"
        self.decoder = CTCPrefixDecoder(beam_width)
        self.transcript = RevisableTranscript()
        self.context = None
        if context_checkpoint is not None:
            from active.v17.stage3_motion_context_v17 import MotionContextScorer
            context = torch.load(context_checkpoint, map_location="cpu", weights_only=False)
            if context.get("format") != "slt_stage3_motion_context_v17" or context["recognizer_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
                raise ValueError("motion context must match this exact recognizer")
            if context["label_to_index"] != stored["label_to_index"]:
                raise ValueError("motion context vocabulary differs")
            self.context = MotionContextScorer(context["evidence_dim"], num_glosses=context["num_glosses"])
            self.context.load_state_dict(context["model_state_dict"], strict=True)
            self.context.eval()
            self.context_weight = context["selected_weight"]
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=max(self.config.windows))
        self.evidence = deque(maxlen=1 + 2 * sum(2**i for i in range(self.config.blocks)))
        self.frame_count = self.processed_count = 0
        self.last = None
        self.context_evidence = []
        self.context_log_probs = []
        self.decoder.reset(); self.transcript.reset()

    @torch.inference_mode()
    def _observe(self, final=False):
        started = time.perf_counter()
        if self.processed_count != self.frame_count:
            frames = np.stack(self.frames)
            windows = observation_windows(frames, len(frames), tuple(self.config.windows))
            logits, embedding = self.encoder(torch.from_numpy(windows).to(self.device), return_embeddings=True)
            evidence = torch.cat((embedding, logits), -1).flatten().detach()
            self.evidence.append(evidence)
            if self.context is not None and len(self.context_evidence) <= 4500:
                self.context_evidence.append(evidence.cpu())
            output, motion = self.model(torch.stack(list(self.evidence))[None], return_embeddings=True)
            log_probs = output[0, -1].log_softmax(-1).cpu().numpy()
            if self.context_proposals and len(self.context_log_probs) <= 4500:
                self.context_log_probs.append(torch.from_numpy(log_probs.copy()))
            alternatives = self.decoder.step(log_probs)
            self.processed_count = self.frame_count
            self.last_motion = motion[0, -1].cpu().numpy()
        else:
            alternatives = self.decoder.alternatives()
        raw_alternatives = alternatives
        context_applied = final and self.context is not None and 0 < len(self.context_evidence) <= 4500
        if context_applied:
            from active.v17.stage3_motion_context_v17 import rerank_motion_candidates
            alternatives = rerank_motion_candidates(self.context, torch.stack(self.context_evidence),
                alternatives, self.context_weight,
                log_probabilities=torch.stack(self.context_log_probs) if self.context_proposals else None)
        hypothesis = self.transcript.update(alternatives, self.frame_count / self.config.fps, final=final)
        result = asdict(hypothesis)
        result.update({
            "seconds": self.frame_count / self.config.fps,
            "glosses": [self.labels[t] for t in hypothesis.tokens],
            "stable_glosses": [self.labels[t] for t in hypothesis.stable_tokens],
            "nbest": [{"glosses": [self.labels[t] for t in prefix], "log_score": score}
                      for prefix, score in alternatives],
            "latency_ms": 1000 * (time.perf_counter() - started),
            "final": final,
            "raw_glosses": [self.labels[t] for t in raw_alternatives[0][0]],
            "context_requested": self.context is not None,
            "context_proposals_requested": self.context_proposals,
            "context_changed": alternatives[0][0] != raw_alternatives[0][0],
        })
        self.last = result
        return result

    def add(self, frame: np.ndarray):
        value = np.asarray(frame, dtype=np.float32)
        if value.shape != (61, 5) or not np.isfinite(value).all():
            raise ValueError("expected a finite v17 landmark frame [61,5]")
        self.frames.append(value.copy())
        self.frame_count += 1
        # Timing is the sole trigger. Recognition confidence never pauses input.
        if self.frame_count % self.config.stride:
            return None
        return self._observe()

    def finish(self):
        if not self.frames:
            return None
        return self._observe(final=True)
