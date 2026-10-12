#!/usr/bin/env python3
"""Export the Stage 3 T5 renderer (slot model) to Core ML for on-device greedy decoding.

Two graphs: an encoder (input ids -> hidden states) and a decoder step that returns the argmax
next token at a given position, so the phone never copies full vocabulary logits. Also writes the
token tables the phone needs: an exact word -> ids map for every word the input encoding can
contain (confidence buckets, lowercased glosses, spelled-word slots) and the id -> piece table for
decoding. Parity: tokenization and Core ML greedy output vs HF generate on held-out corpus rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import coremltools as ct
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from active.v17.stage3_asl_encoding_v17 import encode

LENGTH = 64


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.get_encoder()

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long()).last_hidden_state


class DecoderStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, decoder_input_ids, encoder_hidden, encoder_mask, position):
        logits = self.model(encoder_outputs=(encoder_hidden,), attention_mask=encoder_mask.long(),
                            decoder_input_ids=decoder_input_ids.long()).logits
        row = torch.index_select(logits[0], 0, position.long())
        return torch.argmax(row, dim=-1).to(torch.int32)


def decode_pieces(pieces):
    """What the phone does: join SentencePiece pieces, '▁' -> space (HF T5 decode equivalent)."""
    text = ''.join(pieces).replace('▁', ' ').strip()
    return ' '.join(text.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, default=ROOT / 'artifacts/models/stage3_v17_asl_order_fs_v1')
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/coreml/stage3_t5_fs_v1')
    ap.add_argument('--rows', type=int, default=200)
    ap.add_argument('--corpus', type=Path, default=ROOT / 'data/local/stage3_asl_corpus_v17/corpus_with_fs_slots.jsonl')
    ap.add_argument('--parity-split', choices=['train', 'validation', 'test'], default='test',
                    help='Numerical export parity only; not translation quality or checkpoint selection.')
    ap.add_argument('--tokens-only', action='store_true', help='rewrite stage3_tokens.json only')
    args = ap.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    args.corpus = args.corpus.resolve()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint, local_files_only=True).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- token tables
    torch.set_num_threads(4)
    corpus = [json.loads(l) for l in args.corpus.open()]
    contract_path = args.checkpoint / 'stage3_input_contract.json'
    contract = json.loads(contract_path.read_text()) if contract_path.exists() else {}
    glosses = sorted({g for r in corpus for g in r['glosses']} | {f'FS{i}' for i in range(4)})
    # Every live label must be encodable, including ones the corpus never used (e.g. DIFFERENT).
    payload = torch.load(ROOT / json.loads((ROOT / 'artifacts/reports/segmental_decoder_v17_20260927/'
                                             'stream_config_v3_final.json').read_text())['recognizer'],
                         map_location='cpu', weights_only=False)
    words = sorted({'hi', 'mid', 'lo'} | {g.lower() for g in glosses} | {l.lower() for l in payload['label_to_index']})
    word_ids = {w: tok(w, add_special_tokens=False).input_ids for w in words}
    eos = tok.eos_token_id
    mismatched = 0
    held = [r for r in corpus if r['split'] == args.parity_split]
    if not held:
        raise ValueError('No rows in requested numerical-parity split')
    for r in held:
        text = encode(r['glosses'], r['confidences'])
        full = tok(text, truncation=True, max_length=LENGTH).input_ids
        joined = [i for w in text.split() for i in word_ids[w]][:LENGTH - 1] + [eos]
        mismatched += int(full != joined)
    pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
    special = sorted(set(tok.all_special_ids))
    (args.output_dir / 'stage3_tokens.json').write_text(json.dumps(dict(
        word_ids=word_ids, pieces=pieces, special_ids=special, eos_id=eos, pad_id=tok.pad_token_id,
        decoder_start_id=model.config.decoder_start_token_id, max_length=LENGTH,
        buckets=dict(high_edge=.60, mid_edge=.40), checkpoint=str(args.checkpoint.relative_to(ROOT)),
        reviewed_templates_enabled=contract.get('reviewed_templates_enabled', True),
        utterance_segmentation=contract.get('utterance_segmentation', 'pause'),
        output=contract.get('output', 'sentence')), ensure_ascii=False))
    if args.tokens_only:
        print(dict(words=len(words), tokenization_mismatches=mismatched))
        return

    # ---- Core ML graphs
    ids = torch.zeros(1, LENGTH, dtype=torch.int32)
    mask = torch.ones(1, LENGTH, dtype=torch.int32)
    with torch.inference_mode():
        enc_t = torch.jit.trace(Encoder(model).eval(), (ids, mask), strict=False)
        hidden = Encoder(model)(ids, mask)
        pos = torch.tensor([0], dtype=torch.int32)
        dec_t = torch.jit.trace(DecoderStep(model).eval(), (ids, hidden, mask, pos), strict=False)
    enc = ct.convert(enc_t, inputs=[ct.TensorType(name='input_ids', shape=(1, LENGTH), dtype=np.int32),
                                    ct.TensorType(name='attention_mask', shape=(1, LENGTH), dtype=np.int32)],
                     outputs=[ct.TensorType(name='hidden')], convert_to='mlprogram',
                     minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT32)
    dec = ct.convert(dec_t, inputs=[ct.TensorType(name='decoder_input_ids', shape=(1, LENGTH), dtype=np.int32),
                                    ct.TensorType(name='encoder_hidden', shape=tuple(hidden.shape), dtype=np.float32),
                                    ct.TensorType(name='encoder_mask', shape=(1, LENGTH), dtype=np.int32),
                                    ct.TensorType(name='position', shape=(1,), dtype=np.int32)],
                     outputs=[ct.TensorType(name='next_token')], convert_to='mlprogram',
                     minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT32)
    enc.save(str(args.output_dir / 'Stage3T5EncoderV17.mlpackage'))
    dec.save(str(args.output_dir / 'Stage3T5DecoderStepV17.mlpackage'))
    # Compile into the internal-disk cache; Core ML can fail to plan large weights
    # directly from the project's external exFAT drive.
    from active.v17.coreml_runtime_v17 import load
    enc_rt = load(args.output_dir / 'Stage3T5EncoderV17.mlpackage', 'CPU_AND_GPU')
    dec_rt = load(args.output_dir / 'Stage3T5DecoderStepV17.mlpackage', 'CPU_AND_GPU')

    def coreml_generate(text):
        input_ids = [i for w in text.split() for i in word_ids[w]][:LENGTH - 1] + [eos]
        n = len(input_ids)
        ids_arr = np.zeros((1, LENGTH), np.int32); ids_arr[0, :n] = input_ids
        mask_arr = np.zeros((1, LENGTH), np.int32); mask_arr[0, :n] = 1
        h = enc_rt.predict({'input_ids': ids_arr, 'attention_mask': mask_arr})['hidden']
        out = [model.config.decoder_start_token_id]
        while len(out) < LENGTH:
            d = np.zeros((1, LENGTH), np.int32); d[0, :len(out)] = out
            nxt = int(np.asarray(dec_rt.predict({'decoder_input_ids': d, 'encoder_hidden': h, 'encoder_mask': mask_arr,
                                                 'position': np.array([len(out) - 1], np.int32)})['next_token']).reshape(-1)[0])
            if nxt == eos:
                break
            out.append(nxt)
        return decode_pieces([pieces[i] for i in out[1:] if i not in special])

    same = decode_same = 0
    rows = held[:args.rows]
    for r in rows:
        text = encode(r['glosses'], r['confidences'])
        enc_in = tok(text, return_tensors='pt', truncation=True, max_length=LENGTH)
        with torch.inference_mode():
            gen = model.generate(**enc_in, max_new_tokens=LENGTH, num_beams=1, do_sample=False)[0].tolist()
        reference = tok.decode(gen, skip_special_tokens=True).strip()
        mine = decode_pieces([pieces[i] for i in gen if i not in special])
        decode_same += int(' '.join(reference.split()) == mine)
        same += int(coreml_generate(text) == ' '.join(reference.split()))
    result = dict(tokenization_rows=len(held), tokenization_mismatches=mismatched, words=len(words),
                  parity_split=args.parity_split, parity_corpus=str(args.corpus),
                  generation_rows=len(rows), coreml_equals_hf=same, piece_decode_equals_hf=decode_same,
                  vocabulary=len(pieces), hidden_shape=list(hidden.shape))
    (args.output_dir / 'export.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    if mismatched or same != len(rows) or decode_same != len(rows):
        raise RuntimeError('Core ML export did not preserve tokenization and greedy outputs')


if __name__ == '__main__':
    main()
