"""T5 export dry-run — tests whether the existing Flan-T5-Base checkpoint can be
pushed through torch.onnx.export for the encoder and a single decoder step.

This does NOT attempt full autoregressive decoding on mobile (that needs a native
generation loop using KV-cache, which is a Phase-5 problem).
"""
from __future__ import annotations
import os, sys, json, time, warnings, traceback
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import T5_DIR, ARTIFACTS

warnings.filterwarnings('ignore')


def main():
    torch.set_grad_enabled(False)
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    tok = T5Tokenizer.from_pretrained(str(T5_DIR))
    model = T5ForConditionalGeneration.from_pretrained(str(T5_DIR)).eval()
    print(f'▶ Loaded T5: d_model={model.config.d_model}, vocab={model.config.vocab_size}')

    # Quick smoke test — run generate to confirm weights decode meaningfully.
    prompt = 'Translate this ASL gloss to natural conversational English: HELLO HOW YOU'
    ids = tok(prompt, return_tensors='pt').input_ids
    t0 = time.perf_counter()
    gen = model.generate(ids, max_length=32, num_beams=1)
    print(f'  generate: {tok.decode(gen[0], skip_special_tokens=True)!r}  ({(time.perf_counter()-t0)*1000:.0f}ms)')

    out_dir = ARTIFACTS / 't5'
    out_dir.mkdir(exist_ok=True)
    report = {'t5_dir': str(T5_DIR), 'd_model': model.config.d_model,
              'vocab_size': model.config.vocab_size, 'attempts': {}}

    # ── Encoder ONNX export ──
    print('\n▶ Exporting T5 encoder to ONNX (dynamo, opset 18)...')
    encoder = model.encoder
    class EncWrap(torch.nn.Module):
        def __init__(self, enc): super().__init__(); self.enc = enc
        def forward(self, input_ids, attention_mask):
            return self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    ew = EncWrap(encoder).eval()
    dummy_ids = torch.ones((1, 16), dtype=torch.long)
    dummy_am  = torch.ones((1, 16), dtype=torch.long)
    try:
        enc_path = out_dir / 'T5_encoder.onnx'
        torch.onnx.export(
            ew, (dummy_ids, dummy_am), str(enc_path),
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'seq'},
                'attention_mask': {0: 'batch', 1: 'seq'},
                'last_hidden_state': {0: 'batch', 1: 'seq'},
            },
            opset_version=18, do_constant_folding=True, dynamo=True,
        )
        size = enc_path.stat().st_size / 1e6
        data_path = enc_path.with_suffix('.onnx.data')
        if data_path.exists():
            size += data_path.stat().st_size / 1e6
        print(f'✓ encoder exported: {size:.1f} MB')
        report['attempts']['encoder_onnx'] = {'ok': True, 'size_mb': size, 'path': str(enc_path)}
    except Exception as e:
        err = f'{type(e).__name__}: {str(e)[:300]}'
        print(f'✗ encoder ONNX failed: {err}')
        report['attempts']['encoder_onnx'] = {'ok': False, 'error': err,
                                              'traceback': traceback.format_exc()[-1200:]}

    # ── Decoder (1-step) ONNX export ──
    # We don't attempt cached autoregressive decoding — that requires past_key_values
    # input wiring which is non-trivial and usually handled by optimum.onnxruntime.
    print('\n▶ Exporting T5 decoder single-step (no KV-cache) to ONNX...')
    class DecStep(torch.nn.Module):
        def __init__(self, model): super().__init__(); self.m = model
        def forward(self, decoder_input_ids, encoder_hidden_states, attention_mask):
            out = self.m(
                encoder_outputs=(encoder_hidden_states,),
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            return out.logits
    ds = DecStep(model).eval()
    dummy_dec_ids = torch.zeros((1, 1), dtype=torch.long)
    dummy_hidden = torch.randn(1, 16, model.config.d_model)
    try:
        dec_path = out_dir / 'T5_decoder_step.onnx'
        torch.onnx.export(
            ds, (dummy_dec_ids, dummy_hidden, dummy_am), str(dec_path),
            input_names=['decoder_input_ids', 'encoder_hidden_states', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'decoder_input_ids': {0: 'batch', 1: 'dec_seq'},
                'encoder_hidden_states': {0: 'batch', 1: 'enc_seq'},
                'attention_mask': {0: 'batch', 1: 'enc_seq'},
                'logits': {0: 'batch', 1: 'dec_seq'},
            },
            opset_version=18, do_constant_folding=True, dynamo=True,
        )
        size = dec_path.stat().st_size / 1e6
        data_path = dec_path.with_suffix('.onnx.data')
        if data_path.exists():
            size += data_path.stat().st_size / 1e6
        print(f'✓ decoder (1-step) exported: {size:.1f} MB')
        report['attempts']['decoder_step_onnx'] = {'ok': True, 'size_mb': size, 'path': str(dec_path)}
    except Exception as e:
        err = f'{type(e).__name__}: {str(e)[:300]}'
        print(f'✗ decoder ONNX failed: {err}')
        report['attempts']['decoder_step_onnx'] = {'ok': False, 'error': err,
                                                   'traceback': traceback.format_exc()[-1200:]}

    (ARTIFACTS / 't5_dryrun_report.json').write_text(json.dumps(report, indent=2))
    print(f'\n▶ Report: {ARTIFACTS/"t5_dryrun_report.json"}')


if __name__ == '__main__':
    main()
