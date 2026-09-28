"""Stage 3 renderer on Core ML (the same graphs and token tables the iPhone app uses).

Drop-in for scripts/live_isolated_v17.TinyStage3Naturalizer when the live pipeline runs on Core ML:
no PyTorch model and no transformers import. The composition model uses greedy T5 without
phrase overrides; older token manifests may retain their reviewed-template behavior.
(Encoder + one argmax decoder step per token.) Literal rendering when generation is empty,
over 300 characters, or a word has no token entry. Parity with HF generate is recorded in
artifacts/coreml/stage3_composition_v17_20260929_v2/export.json (200/200 numerical checks).
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ROOT / 'artifacts/coreml/stage3_composition_v17_20260929_v2'
MANIFEST = ROOT / 'active/v17/stage3_mobile_naturalizer_manifest_v17.json'


class CoreMLStage3Naturalizer:
    def __init__(self, packages: Path = PACKAGES, compute_units: str = 'CPU_AND_GPU'):
        from active.v17.stage3_mobile_naturalizer_v17 import literal_render
        from active.v17 import coreml_runtime_v17 as cml
        self.encoder = cml.load(packages / 'Stage3T5EncoderV17.mlpackage', compute_units)
        self.decoder = cml.load(packages / 'Stage3T5DecoderStepV17.mlpackage', compute_units)
        tokens = json.loads((packages / 'stage3_tokens.json').read_text())
        self.word_ids = tokens['word_ids']
        self.pieces = tokens['pieces']
        self.special = set(tokens['special_ids'])
        self.eos, self.start, self.length = tokens['eos_id'], tokens['decoder_start_id'], tokens['max_length']
        self.full_utterance = tokens.get('utterance_segmentation') == 'model'
        self.manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        self.templates = {tuple(r['glosses']): str(r['english']) for r in self.manifest['reviewed_templates']}
        if not tokens.get('reviewed_templates_enabled', True):
            self.templates = {}
        self.literal = lambda glosses: literal_render(glosses, self.manifest)
        self.checkpoint = str(packages)

    def _generate(self, glosses, confidences=None):
        from active.v17.stage3_asl_encoding_v17 import encode
        ids = []
        for word in encode(glosses, confidences).split():
            if word not in self.word_ids:
                raise KeyError(f'no Stage-3 tokens for {word!r}')
            ids += self.word_ids[word]
        if self.full_utterance and len(ids) >= self.length:
            raise ValueError('Finished utterance exceeds model context; refuse to drop words')
        ids = ids[:self.length - 1] + [self.eos]
        input_ids = np.zeros((1, self.length), np.int32); input_ids[0, :len(ids)] = ids
        mask = np.zeros((1, self.length), np.int32); mask[0, :len(ids)] = 1
        hidden = self.encoder.predict({'input_ids': input_ids, 'attention_mask': mask})['hidden']
        out = [self.start]
        decoder_ids = np.zeros((1, self.length), np.int32); decoder_ids[0, 0] = self.start
        while len(out) < self.length:
            nxt = int(np.asarray(self.decoder.predict({
                'decoder_input_ids': decoder_ids, 'encoder_hidden': hidden, 'encoder_mask': mask,
                'position': np.array([len(out) - 1], np.int32)})['next_token']).reshape(-1)[0])
            if nxt == self.eos:
                break
            decoder_ids[0, len(out)] = nxt
            out.append(nxt)
        text = ''.join(self.pieces[i] for i in out[1:] if i not in self.special).replace('▁', ' ')
        return ' '.join(text.split())

    def warm(self):
        started = time.perf_counter()
        try:
            self._generate(['HELLO'])
            return dict(ok=True, naturalizer='t5_efficient_tiny_coreml', checkpoint=self.checkpoint,
                        latency_ms=1000 * (time.perf_counter() - started))
        except Exception as exc:
            return dict(ok=False, naturalizer='t5_efficient_tiny_coreml', error=f'{type(exc).__name__}: {exc}',
                        latency_ms=1000 * (time.perf_counter() - started))

    def rephrase(self, glosses, confidences=None):
        """Same contract as TinyStage3Naturalizer.rephrase."""
        started = time.perf_counter()
        glosses = list(glosses)
        if confidences is not None and len(confidences) != len(glosses):
            confidences = None
        template = self.templates.get(tuple(glosses))
        fallback, fallback_mode = (template, 'reviewed_template') if template else (self.literal(glosses), 'literal_fallback')
        result = dict(model=self.checkpoint, input_glosses=glosses,
                      input_confidences=None if confidences is None else list(confidences),
                      stage3_encoding='evidence', literal_sentence=self.literal(glosses),
                      fallback_sentence=fallback, fallback_mode=fallback_mode)
        if template:
            result.update(sentence=template, rendering_mode='reviewed_template', safe_fallback_used=False)
        else:
            try:
                sentence = self._generate(glosses, confidences)
                if not sentence or len(sentence) > 300:
                    raise ValueError('tiny Stage-3 model returned an empty or overlong sentence')
                result.update(sentence=sentence, rendering_mode='t5_efficient_tiny', safe_fallback_used=False)
            except Exception as exc:
                result.update(sentence=fallback, rendering_mode=fallback_mode, safe_fallback_used=True,
                              error=f'{type(exc).__name__}: {exc}')
        result['latency_ms'] = 1000 * (time.perf_counter() - started)
        return result
