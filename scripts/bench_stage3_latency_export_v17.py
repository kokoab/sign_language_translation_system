#!/usr/bin/env python3
"""Export Stage 3 candidate architectures to Core ML for on-device latency timing only.

Weights are random: latency depends on architecture, precision and decoding scheme, not
on trained values, so no checkpoint is downloaded. These packages must never be used as
renderers. Each candidate is two graphs, matching how the phone generates:

  t5 kv       encoder (ids -> per-layer cross-attention K/V) + stateful decoder step
              (self-attention K/V kept in Core ML state; one token per call)
  t5 nocache  encoder + decoder step that recomputes all 64 positions every token
              (the deployed exporter's scheme, active/v17/export_stage3_t5_coreml_v17.py)
  lm kv       decoder-only prefill (64-token prompt -> K/V) + stateful one-token step

Output: artifacts/coreml/stage3_latency_bench_v17/<name>/{First,Step}.mlpackage and
manifest.json, which the RunnerTests benchmark reads from the app's Documents folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import coremltools as ct
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/coreml/stage3_latency_bench_v17"
LIN = 64      # encoder / prompt length (the deployed contract)
LOUT = 64     # decoder positions
LCTX = 128    # decoder-only cache: prompt + output

T5 = {
    # name: d_model, heads, d_kv, d_ff, enc layers, dec layers, gated FFN
    "t5_tiny_v2": (256, 4, 64, 1024, 4, 4, False),        # deployed stage3_composition_v2 config
    "t5_small": (512, 8, 64, 2048, 6, 6, False),          # google-t5/t5-small (60M)
    "flan_t5_small": (512, 6, 64, 1024, 8, 8, True),      # google/flan-t5-small (77M)
    "flan_t5_base": (768, 12, 64, 2048, 12, 12, True),    # google/flan-t5-base (248M)
}
LM_CONFIGS = {
    # name: hidden, layers, heads, kv heads, head_dim, intermediate, vocab, qkv bias
    "smollm2_135m": (576, 30, 9, 3, 64, 1536, 49152, False),
    "smollm2_360m": (960, 32, 15, 5, 64, 2560, 49152, False),
    "gemma3_270m": (640, 18, 4, 1, 256, 2048, 262144, False),
    "qwen25_05b": (896, 24, 14, 2, 64, 4864, 151936, True),
}
T5_VOCAB = 32128


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class FFN(nn.Module):
    def __init__(self, d, ff, gated):
        super().__init__()
        self.gated = gated
        self.wi = nn.Linear(d, ff, bias=False)
        self.wg = nn.Linear(d, ff, bias=False) if gated else None
        self.wo = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        h = self.wi(x)
        h = torch.nn.functional.gelu(h, approximate="tanh") * self.wg(x) if self.gated else torch.relu(h)
        return self.wo(h)


def attend(q, k, v, bias):
    """q (H,Q,dk), k/v (H,K,dk), bias broadcast to (H,Q,K); T5 has no 1/sqrt(d) scale."""
    return torch.softmax(q @ k.transpose(-1, -2) + bias, dim=-1) @ v


class T5Layer(nn.Module):
    def __init__(self, d, h, dk, ff, gated, cross):
        super().__init__()
        self.h, self.dk = h, dk
        self.n1, self.q, self.k, self.v, self.o = RMSNorm(d), *(nn.Linear(d, h * dk, bias=False) for _ in range(3)), nn.Linear(h * dk, d, bias=False)
        if cross:
            self.n2, self.cq, self.ck, self.cv, self.co = RMSNorm(d), *(nn.Linear(d, h * dk, bias=False) for _ in range(3)), nn.Linear(h * dk, d, bias=False)
        self.n3, self.ffn = RMSNorm(d), FFN(d, ff, gated)

    def heads(self, x):
        return x.view(x.shape[0], self.h, self.dk).transpose(0, 1)

    def merge(self, x):
        return x.transpose(0, 1).reshape(x.shape[1], self.h * self.dk)


class T5Encoder(nn.Module):
    """ids -> cross-attention K/V for every decoder layer, (Ldec, H, LIN, dk) each."""

    def __init__(self, cfg, decoder):
        super().__init__()
        d, h, dk, ff, le, _, gated = cfg
        self.embed = nn.Embedding(T5_VOCAB, d)
        self.layers = nn.ModuleList(T5Layer(d, h, dk, ff, gated, False) for _ in range(le))
        self.norm = RMSNorm(d)
        self.bias = nn.Parameter(torch.randn(h, LIN, LIN) * .1)
        self.decoder_layers = decoder.layers

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids[0].long())
        mask = (1.0 - attention_mask[0].float())[None, None, :] * -1e4
        for layer in self.layers:
            y = layer.n1(x)
            a = attend(layer.heads(layer.q(y)), layer.heads(layer.k(y)), layer.heads(layer.v(y)), self.bias + mask)
            x = x + layer.o(layer.merge(a))
            x = x + layer.ffn(layer.n3(x))
        x = self.norm(x)
        ks = torch.stack([l.heads(l.ck(x)) for l in self.decoder_layers])
        vs = torch.stack([l.heads(l.cv(x)) for l in self.decoder_layers])
        return ks, vs


class T5Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h, dk, ff, _, ld, gated = cfg
        self.h, self.dk, self.ld = h, dk, ld
        self.embed = nn.Embedding(T5_VOCAB, d)
        self.layers = nn.ModuleList(T5Layer(d, h, dk, ff, gated, True) for _ in range(ld))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, T5_VOCAB, bias=False)
        self.bias = nn.Parameter(torch.randn(h, LOUT, LOUT) * .1)


class T5StepKV(nn.Module):
    """One token per call; self-attention K/V live in Core ML state."""

    def __init__(self, dec):
        super().__init__()
        self.dec = dec
        shape = (dec.ld, dec.h, LOUT, dec.dk)
        self.register_buffer("self_k", torch.zeros(shape, dtype=torch.float16))
        self.register_buffer("self_v", torch.zeros(shape, dtype=torch.float16))

    def forward(self, token, position, cross_k, cross_v, encoder_mask):
        dec = self.dec
        x = dec.embed(token[0].long())                                    # (1, d)
        slot = (torch.arange(LOUT) == position.long()).float()             # one-hot write position
        causal = (torch.arange(LOUT) > position.long()).float() * -1e4
        bias = torch.index_select(dec.bias, 1, position.long()) + causal   # (H,1,LOUT)
        cmask = (1.0 - encoder_mask[0].float())[None, None, :] * -1e4
        for i, layer in enumerate(dec.layers):
            y = layer.n1(x)
            k_new, v_new = layer.heads(layer.k(y)), layer.heads(layer.v(y))  # (H,1,dk)
            keep = (1.0 - slot)[None, :, None]
            k_all = self.self_k[i].float() * keep + k_new * slot[None, :, None]
            v_all = self.self_v[i].float() * keep + v_new * slot[None, :, None]
            self.self_k[i] = k_all.half()
            self.self_v[i] = v_all.half()
            x = x + layer.o(layer.merge(attend(layer.heads(layer.q(y)), k_all, v_all, bias)))
            y = layer.n2(x)
            x = x + layer.co(layer.merge(attend(layer.heads(layer.cq(y)), cross_k[i], cross_v[i], cmask)))
            x = x + layer.ffn(layer.n3(x))
        logits = dec.head(dec.norm(x))
        return torch.argmax(logits, dim=-1).to(torch.int32)


class T5StepNoCache(nn.Module):
    """The deployed scheme: rerun the decoder over all LOUT positions for every token."""

    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, decoder_input_ids, position, cross_k, cross_v, encoder_mask):
        dec = self.dec
        x = dec.embed(decoder_input_ids[0].long())                        # (LOUT, d)
        causal = torch.triu(torch.full((LOUT, LOUT), -1e4), 1)
        cmask = (1.0 - encoder_mask[0].float())[None, None, :] * -1e4
        for i, layer in enumerate(dec.layers):
            y = layer.n1(x)
            x = x + layer.o(layer.merge(attend(layer.heads(layer.q(y)), layer.heads(layer.k(y)), layer.heads(layer.v(y)), dec.bias + causal)))
            y = layer.n2(x)
            x = x + layer.co(layer.merge(attend(layer.heads(layer.cq(y)), cross_k[i], cross_v[i], cmask)))
            x = x + layer.ffn(layer.n3(x))
        row = torch.index_select(x, 0, position.long())
        return torch.argmax(dec.head(dec.norm(row)), dim=-1).to(torch.int32)


class LMLayer(nn.Module):
    def __init__(self, d, h, kvh, hd, ff, qkv_bias):
        super().__init__()
        self.h, self.kvh, self.hd = h, kvh, hd
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.q = nn.Linear(d, h * hd, bias=qkv_bias)
        self.k = nn.Linear(d, kvh * hd, bias=qkv_bias)
        self.v = nn.Linear(d, kvh * hd, bias=qkv_bias)
        self.o = nn.Linear(h * hd, d, bias=False)
        self.gate, self.up = nn.Linear(d, ff, bias=False), nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def mlp(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


def rope(x, cos, sin):
    """x (H,T,hd); cos/sin (T,hd)."""
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


class LM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, layers, h, kvh, hd, ff, vocab, qkv_bias = cfg
        self.h, self.kvh, self.hd, self.nl = h, kvh, hd, layers
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(LMLayer(d, h, kvh, hd, ff, qkv_bias) for _ in range(layers))
        self.norm = RMSNorm(d)
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        angles = torch.arange(LCTX).float()[:, None] * inv[None, :]
        angles = torch.cat([angles, angles], dim=-1)
        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())

    def logits(self, x):
        return self.norm(x) @ self.embed.weight.t()   # tied embeddings


class LMPrefill(nn.Module):
    """Prompt (LIN tokens) -> stacked K/V (L, KVH, LIN, hd) and the next token."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, input_ids, attention_mask, last):
        lm = self.lm
        x = lm.embed(input_ids[0].long())
        cos, sin = lm.cos[:LIN], lm.sin[:LIN]
        mask = torch.triu(torch.full((LIN, LIN), -1e4), 1) + (1.0 - attention_mask[0].float())[None, :] * -1e4
        ks, vs = [], []
        rep = lm.h // lm.kvh
        for layer in lm.layers:
            y = layer.n1(x)
            q = rope(layer.q(y).view(LIN, lm.h, lm.hd).transpose(0, 1), cos, sin)
            k = rope(layer.k(y).view(LIN, lm.kvh, lm.hd).transpose(0, 1), cos, sin)
            v = layer.v(y).view(LIN, lm.kvh, lm.hd).transpose(0, 1)
            ks.append(k)
            vs.append(v)
            a = torch.softmax(q @ k.repeat_interleave(rep, 0).transpose(-1, -2) / lm.hd ** .5 + mask, -1) @ v.repeat_interleave(rep, 0)
            x = x + layer.o(a.transpose(0, 1).reshape(LIN, lm.h * lm.hd))
            x = x + layer.mlp(layer.n2(x))
        row = torch.index_select(x, 0, last.long())
        return torch.stack(ks), torch.stack(vs), torch.argmax(lm.logits(row), -1).to(torch.int32)


class LMStepKV(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        shape = (lm.nl, lm.kvh, LCTX, lm.hd)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=torch.float16))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=torch.float16))

    def forward(self, token, position):
        lm = self.lm
        x = lm.embed(token[0].long())
        cos, sin = torch.index_select(lm.cos, 0, position.long()), torch.index_select(lm.sin, 0, position.long())
        slot = (torch.arange(LCTX) == position.long()).float()
        mask = (torch.arange(LCTX) > position.long()).float() * -1e4
        rep = lm.h // lm.kvh
        for i, layer in enumerate(lm.layers):
            y = layer.n1(x)
            q = rope(layer.q(y).view(1, lm.h, lm.hd).transpose(0, 1), cos, sin)
            k_new = rope(layer.k(y).view(1, lm.kvh, lm.hd).transpose(0, 1), cos, sin)
            v_new = layer.v(y).view(1, lm.kvh, lm.hd).transpose(0, 1)
            keep = (1.0 - slot)[None, :, None]
            k_all = self.k_cache[i].float() * keep + k_new * slot[None, :, None]
            v_all = self.v_cache[i].float() * keep + v_new * slot[None, :, None]
            self.k_cache[i] = k_all.half()
            self.v_cache[i] = v_all.half()
            a = torch.softmax(q @ k_all.repeat_interleave(rep, 0).transpose(-1, -2) / lm.hd ** .5 + mask, -1) @ v_all.repeat_interleave(rep, 0)
            x = x + layer.o(a.transpose(0, 1).reshape(1, lm.h * lm.hd))
            x = x + layer.mlp(layer.n2(x))
        return torch.argmax(lm.logits(x), -1).to(torch.int32)


def i32(name, shape):
    return ct.TensorType(name=name, shape=shape, dtype=np.int32)


def f32(name, shape):
    return ct.TensorType(name=name, shape=shape, dtype=np.float32)


def convert(module, examples, inputs, outputs, precision, states=None):
    with torch.inference_mode():
        traced = torch.jit.trace(module.eval(), examples, strict=False)
    return ct.convert(traced, inputs=inputs, outputs=[ct.TensorType(name=n) for n in outputs],
                      states=states, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                      compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32)


def quantize(model, bits):
    from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights
    config = OptimizationConfig(global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype=f"int{bits}", granularity="per_block", block_size=32))
    return linear_quantize_weights(model, config=config)


def params(module):
    return sum(p.numel() for p in module.parameters())


def export_t5(name, precision, scheme, bits=None):
    cfg = T5[name]
    torch.manual_seed(0)
    dec = T5Decoder(cfg)
    enc = T5Encoder(cfg, dec)
    for p in list(enc.parameters()) + list(dec.parameters()):
        nn.init.normal_(p, std=.02)
    d, h, dk = cfg[0], cfg[1], cfg[2]
    ids, mask = torch.zeros(1, LIN, dtype=torch.int32), torch.ones(1, LIN, dtype=torch.int32)
    ck, cv = enc(ids, mask)
    first = convert(enc, (ids, mask), [i32("input_ids", (1, LIN)), i32("attention_mask", (1, LIN))],
                    ["cross_k", "cross_v"], precision)
    pos = torch.zeros(1, dtype=torch.int32)
    cross = [f32("cross_k", tuple(ck.shape)), f32("cross_v", tuple(cv.shape)), i32("encoder_mask", (1, LIN))]
    if scheme == "kv":
        step_module = T5StepKV(dec)
        states = [ct.StateType(wrapped_type=ct.TensorType(shape=tuple(step_module.self_k.shape), dtype=np.float16), name=n)
                  for n in ("self_k", "self_v")]
        step = convert(step_module, (torch.zeros(1, 1, dtype=torch.int32), pos, ck, cv, mask),
                       [i32("token", (1, 1)), i32("position", (1,)), *cross], ["next_token"], precision, states)
        token_input = "token"
    else:
        step = convert(T5StepNoCache(dec), (torch.zeros(1, LOUT, dtype=torch.int32), pos, ck, cv, mask),
                       [i32("decoder_input_ids", (1, LOUT)), i32("position", (1,)), *cross], ["next_token"], precision)
        token_input = "decoder_input_ids"
    if bits:
        first, step = quantize(first, bits), quantize(step, bits)
    tag = f"{name}_{scheme}_{precision}" + (f"_int{bits}" if bits else "")
    return tag, first, step, dict(family="t5", scheme=scheme, precision=precision, weight_bits=bits or (16 if precision == "fp16" else 32),
                                  params_m=round((params(enc) + params(dec) - params(dec.layers)) / 1e6 + params(dec.layers) / 1e6, 1),
                                  token_input=token_input, pass_through={"cross_k": "cross_k", "cross_v": "cross_v"})


def export_lm(name, precision, bits=None):
    cfg = LM_CONFIGS[name]
    torch.manual_seed(0)
    lm = LM(cfg)
    for n, p in lm.named_parameters():
        nn.init.normal_(p, std=.02)
    ids, mask, last = torch.zeros(1, LIN, dtype=torch.int32), torch.ones(1, LIN, dtype=torch.int32), torch.tensor([LIN - 1], dtype=torch.int32)
    first = convert(LMPrefill(lm), (ids, mask, last), [i32("input_ids", (1, LIN)), i32("attention_mask", (1, LIN)), i32("last", (1,))],
                    ["prompt_k", "prompt_v", "next_token"], precision)
    step_module = LMStepKV(lm)
    states = [ct.StateType(wrapped_type=ct.TensorType(shape=tuple(step_module.k_cache.shape), dtype=np.float16), name=n)
              for n in ("k_cache", "v_cache")]
    step = convert(step_module, (torch.zeros(1, 1, dtype=torch.int32), torch.tensor([LIN], dtype=torch.int32)),
                   [i32("token", (1, 1)), i32("position", (1,))], ["next_token"], precision, states)
    if bits:
        first, step = quantize(first, bits), quantize(step, bits)
    tag = f"{name}_kv_{precision}" + (f"_int{bits}" if bits else "")
    return tag, first, step, dict(family="lm", scheme="kv", precision=precision, weight_bits=bits or 16,
                                  params_m=round(params(lm) / 1e6, 1), token_input="token", pass_through={},
                                  position_offset=LIN, prefill_copies_state={"prompt_k": "k_cache", "prompt_v": "v_cache"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="export only these tags")
    args = parser.parse_args()
    jobs = [
        ("t5", "t5_tiny_v2", "fp32", "nocache", None),
        ("t5", "t5_tiny_v2", "fp32", "kv", None),
        ("t5", "t5_tiny_v2", "fp16", "kv", None),
        ("t5", "flan_t5_small", "fp32", "nocache", None),
        ("t5", "flan_t5_small", "fp32", "kv", None),
        ("t5", "flan_t5_small", "fp16", "kv", None),
        ("t5", "t5_small", "fp16", "kv", None),
        ("t5", "flan_t5_base", "fp16", "kv", None),
        ("t5", "flan_t5_base", "fp16", "kv", 8),
        ("lm", "smollm2_135m", "fp16", None, None),
        ("lm", "smollm2_360m", "fp16", None, None),
        ("lm", "gemma3_270m", "fp16", None, None),
        ("lm", "gemma3_270m", "fp16", None, 4),
        ("lm", "qwen25_05b", "fp16", None, None),
        ("lm", "qwen25_05b", "fp16", None, 4),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for family, name, precision, scheme, bits in jobs:
        tag = (f"{name}_{scheme}_{precision}" if family == "t5" else f"{name}_kv_{precision}") + (f"_int{bits}" if bits else "")
        if args.only and tag not in args.only:
            continue
        print(f"exporting {tag}", flush=True)
        tag, first, step, meta = export_t5(name, precision, scheme, bits) if family == "t5" else export_lm(name, precision, bits)
        folder = OUT / tag
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        first.save(str(folder / "First.mlpackage"))
        step.save(str(folder / "Step.mlpackage"))
        meta["size_mb"] = round(sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 2**20, 1)
        manifest[tag] = meta
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"  {tag}: {meta}", flush=True)


if __name__ == "__main__":
    main()
