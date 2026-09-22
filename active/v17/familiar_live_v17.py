"""Bounded-history greedy live adapter for the reviewed familiar CTC candidate."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
from scripts.train_local_familiar_ctc_v17 import ROOT, digest, model
from active.v17.train_unified_streaming_ctc_v17 import rolling_windows

DEFAULT = ROOT / 'artifacts/models/local_familiar_ctc_v17_20260922/seed_17521_familiar.pth'

class FamiliarRecognizer:
    def __init__(self, checkpoint=DEFAULT, *, device='cpu'):
        checkpoint = Path(checkpoint).resolve()
        results = json.loads((ROOT/'artifacts/reports/local_familiar_signer_v17_20260922/results.json').read_text())['results']
        allowed = [v for k,v in results.items() if k.endswith(':familiar') and (ROOT/v['checkpoint']).resolve() == checkpoint]
        if len(allowed) != 1 or digest(checkpoint) != allowed[0]['checkpoint_sha256']:
            raise ValueError('expected a reviewed familiar candidate with matching hash')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if payload.get('format') != 'local_familiar_ctc_v17':
            raise ValueError('wrong candidate format')
        self.device = torch.device(device)
        self.encoder, self.head = model(payload['recipe'], self.device)
        self.head.load_state_dict(payload['head_state_dict'], strict=True)
        self.head.eval()
        source = torch.load(ROOT/payload['recipe']['base_checkpoint'], map_location='cpu', weights_only=False)
        self.labels = {int(i)+1: label for label,i in source['label_to_index'].items()}
        self.labels[101] = 'OTHER'
        self.config = SimpleNamespace(fps=30)
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=8)
        self.evidence = deque(maxlen=self.head.config.receptive_field_steps)
        self.frame_count = self.processed_count = 0
        self.previous_token = 0
        self.tokens = []
        self.last_logits = None

    def add(self, frame):
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (61,5) or not np.isfinite(frame).all():
            raise ValueError('expected finite 61x5 feature frame')
        self.frames.append(frame.copy()); self.frame_count += 1
        if self.frame_count >= 8 and (self.frame_count-8) % 4 == 0:
            return self._step()
        return None

    def _step(self):
        window = rolling_windows(np.stack(self.frames), stride=4, window_frames=8)[-1]
        with torch.inference_mode():
            logits, pooled = self.encoder(torch.from_numpy(window).unsqueeze(0).to(self.device), return_embeddings=True)
            self.evidence.append(torch.cat((pooled,logits),-1)[0])
            value = self.head(torch.stack(list(self.evidence)).unsqueeze(0))[0,-1]
            if not torch.isfinite(value).all():
                raise ValueError('nonfinite live logits')
            self.last_logits = value.cpu().numpy()
        token = int(self.last_logits.argmax())
        if token and token != self.previous_token:
            self.tokens.append(token)
        self.previous_token = token
        self.processed_count = self.frame_count
        return {'glosses': [self.labels[t] for t in self.tokens], 'stable_glosses': [], 'token':token}

    def finish(self):
        if not self.frame_count:
            return None
        if self.processed_count != self.frame_count:
            return self._step()
        return {'glosses': [self.labels[t] for t in self.tokens], 'stable_glosses': []}
