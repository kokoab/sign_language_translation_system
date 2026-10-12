#!/usr/bin/env python3
"""Export the Android (MediaPipe) live chain to LiteRT/TFLite and measure parity and speed.

Run with the conversion env: artifacts/generated/litert_env/bin/python (litert-torch, torch 2.13).
Models: MediaPipe boundary student, MediaPipe span recognizer (fixed batch 8, as the iPhone package),
MobileCLIP2-S0 image tower (normalised embedding), Stage 3 T5 encoder + decoder step (64 tokens, no
cache, argmax at a position, as the Core ML export). Each is written FP32 and, where it converts,
dynamic-range int8 (`dynamic_wi8_afp32`). Parity is measured against PyTorch on MediaPipe validation
data only (tuning-pool boundary windows, isolated validation spans, validation crops, Stage 3 corpus
validation rows). Timing is the LiteRT CPU interpreter (XNNPACK) on this Mac, not a phone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MP_ROOT = ROOT / "data/local/mediapipe_full_v17_20261003"
CONFIG = ROOT / "artifacts/reports/mediapipe_rebuild_v17_20261004/stream_config_mediapipe_v2.json"
RAW_CACHE = ROOT / "artifacts/generated/mp_unfrozen_phrase_adapt_v17_20261004"
STAGE3 = ROOT / "artifacts/models/stage3_multisentence_tiny_v17_20260929"
STAGE3_CORPUS = ROOT / "data/local/stage3_multisentence_corpus_v17/corpus.jsonl"
LENGTH = 64
BATCH = 8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- runtime helpers

class Lite:
    def __init__(self, path: Path, threads: int = 4):
        from ai_edge_litert.interpreter import Interpreter
        self.it = Interpreter(model_path=str(path), num_threads=threads)
        self.it.allocate_tensors()
        self.inputs = self.it.get_input_details()
        self.outputs = self.it.get_output_details()

    def __call__(self, *arrays):
        for detail, value in zip(self.inputs, arrays):
            self.it.set_tensor(detail["index"], np.ascontiguousarray(value, dtype=detail["dtype"]))
        self.it.invoke()
        return [self.it.get_tensor(o["index"]) for o in self.outputs]


def timing(path: Path, sample, runs: int = 40) -> dict[str, float]:
    result = {}
    for threads in (1, 4):
        lite = Lite(path, threads)
        for _ in range(5):
            lite(*sample)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            lite(*sample)
            times.append(1000 * (time.perf_counter() - start))
        result[f"ms_median_{threads}t"] = round(float(np.median(times)), 3)
        result[f"ms_p90_{threads}t"] = round(float(np.percentile(times, 90)), 3)
    return result


def convert(module: nn.Module, sample, path: Path) -> Path:
    import litert_torch
    if path.exists():
        raise FileExistsError(path)
    litert_torch.convert(module.eval(), tuple(torch.from_numpy(np.asarray(s)) for s in sample)).export(str(path))
    return path


def quantize(float_path: Path, path: Path) -> Path | None:
    from ai_edge_quantizer import quantizer, recipe
    if path.exists():
        raise FileExistsError(path)
    try:
        q = quantizer.Quantizer(str(float_path))
        q.load_quantization_recipe(recipe.dynamic_wi8_afp32())
        q.quantize().export_model(str(path))
        return path
    except Exception as error:  # report, keep FP32
        print(f"quantization failed for {float_path.name}: {error!r}"[:300], flush=True)
        return None


# ----------------------------------------------------------------------------- models

class BoundaryWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, valid):
        return self.model(features, valid > 0.5)


class SpanWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, landmarks, hand_embeddings, hand_valid, hand_boxes):
        return self.model(landmarks, hand_embeddings, hand_valid > 0.5, hand_boxes)


class ImageTower(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, image):
        value = self.visual(image)
        return value / torch.sqrt(torch.sum(value * value, dim=-1, keepdim=True)).clamp_min(1e-8)


class T5Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.get_encoder()

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long()).last_hidden_state


class T5DecoderStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, decoder_input_ids, encoder_hidden, encoder_mask, position):
        logits = self.model(encoder_outputs=(encoder_hidden,), attention_mask=encoder_mask.long(),
                            decoder_input_ids=decoder_input_ids.long()).logits
        row = torch.index_select(logits[0], 0, position.long())
        return torch.argmax(row, dim=-1).to(torch.int32)


def load_unified(path: Path):
    from active.v17.model_hand_mobileclip2_v17 import HandMobileCLIP2Stage1Config, HandMobileCLIP2Stage1V17
    from active.v17.model_unified_multimodal_v17 import (
        UnifiedFusionHeadV17, UnifiedMultimodalStage1V17, UnifiedMultimodalV17Config)
    from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    landmark = SLTStage1V17(Stage1V17Config(**checkpoint["landmark_model_config"]))
    landmark.load_state_dict(checkpoint["landmark_model_state_dict"], strict=True)
    hand = HandMobileCLIP2Stage1V17(HandMobileCLIP2Stage1Config(**checkpoint["hand_model_config"]))
    hand.load_state_dict(checkpoint["hand_model_state_dict"], strict=True)
    head = UnifiedFusionHeadV17(UnifiedMultimodalV17Config(**checkpoint["head_config"]))
    head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    return UnifiedMultimodalStage1V17(landmark, hand, head).eval(), checkpoint


# ----------------------------------------------------------------------------- per model

def export_boundary(out: Path, config: dict) -> dict:
    from active.v17.av_boundary_v17 import AVBoundary, FRAMES, windows
    from active.v17.temporal_boundary_v17 import boundary_features
    payload = torch.load(ROOT / config["boundary"], map_location="cpu", weights_only=False)
    model = AVBoundary(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    wrapper = BoundaryWrapper(model.eval()).eval()
    jobs = json.loads(sorted((MP_ROOT / "continuous").glob("_jobs_*.json"))[-1].read_text())
    xs, vs = [], []
    for row in [r for r in jobs if r["set"] == "tune"]:  # tuning pool: never boundary training
        data = np.load(row["av_raw"])
        x, v = windows(boundary_features(data["raw"].astype(np.float32), data["times"].astype(np.float64), True),
                       model.lookahead)
        xs.append(x)
        vs.append(v.astype(np.float32))
    xs, vs = np.concatenate(xs), np.concatenate(vs)
    sample = (xs[:1], vs[:1])
    report = {"inputs": {"features": [1, FRAMES, 450], "valid": [1, FRAMES]}, "output": "O/B/I logits [1,3]",
              "parity_frames": int(len(xs)), "variants": {}}
    with torch.inference_mode():
        ref = np.concatenate([wrapper(torch.from_numpy(xs[i:i + 512]), torch.from_numpy(vs[i:i + 512])).numpy()
                              for i in range(0, len(xs), 512)])
    fp32 = convert(wrapper, sample, out / "boundary_mediapipe_fp32.tflite")
    for name, path in (("fp32", fp32), ("dynamic_int8", quantize(fp32, out / "boundary_mediapipe_int8.tflite"))):
        if path is None:
            continue
        lite = Lite(path)
        got = np.concatenate([lite(xs[i:i + 1], vs[i:i + 1])[0] for i in range(len(xs))])
        report["variants"][name] = dict(
            file=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha256(path),
            argmax_mismatch_frames=int((got.argmax(1) != ref.argmax(1)).sum()),
            max_abs_logit_diff=float(np.abs(got - ref).max()), **timing(path, sample))
    return report


def export_span(out: Path, config: dict) -> dict:
    model, checkpoint = load_unified(ROOT / config["recognizer"])
    wrapper = SpanWrapper(model).eval()
    data = {}
    for source in ("citizen", "semlex", "local"):
        with np.load(RAW_CACHE / f"{source}_val.npz", allow_pickle=False) as d:
            data[source] = {k: d[k] for k in d.files}
    keys = ("landmarks", "hand_embeddings", "hand_valid", "hand_boxes")

    def batches(values):
        n = len(values["targets"])
        for i in range(0, n, BATCH):
            chunk = [values[k][i:i + BATCH].astype(np.float32) for k in keys]
            real = len(chunk[0])
            if real < BATCH:  # fixed batch: pad with the last row, ignore padded outputs
                chunk = [np.concatenate([c, np.repeat(c[-1:], BATCH - real, 0)]) for c in chunk]
            yield chunk, real

    sample = next(batches(data["citizen"]))[0]
    report = {"inputs": {"landmarks": [BATCH, 32, 61, 5], "hand_embeddings": [BATCH, 16, 3, 512],
                         "hand_valid": [BATCH, 16, 3], "hand_boxes": [BATCH, 16, 3, 4]},
              "output": f"word logits [{BATCH},100]", "variants": {}}
    with torch.inference_mode():
        ref = {s: np.concatenate([wrapper(*(torch.from_numpy(c) for c in chunk)).numpy()[:real]
                                  for chunk, real in batches(v)]) for s, v in data.items()}
    fp32 = convert(wrapper, sample, out / "span_recognizer_mediapipe_b8_fp32.tflite")
    for name, path in (("fp32", fp32), ("dynamic_int8", quantize(fp32, out / "span_recognizer_mediapipe_b8_int8.tflite"))):
        if path is None:
            continue
        lite = Lite(path)
        row = dict(file=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha256(path), domains={})
        for source, values in data.items():
            got = np.concatenate([lite(*chunk)[0][:real] for chunk, real in batches(values)])
            targets = values["targets"]
            row["domains"][source] = dict(
                samples=int(len(targets)), top1_changes=int((got.argmax(1) != ref[source].argmax(1)).sum()),
                torch_top1=round(100 * float((ref[source].argmax(1) == targets).mean()), 3),
                lite_top1=round(100 * float((got.argmax(1) == targets).mean()), 3),
                max_abs_logit_diff=float(np.abs(got - ref[source]).max()))
        row.update(timing(path, sample))
        report["variants"][name] = row
    return report


def export_span_landmark(out: Path) -> dict:
    """Android landmark-only recognizer (no hand images): one input, landmarks [8,32,61,5]."""
    from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
    path = ROOT / "artifacts/models/mp_span_recognizer_v17_local_a_landmark_only_20261004/best_model.pth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    net = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    net.load_state_dict(checkpoint["model_state_dict"], strict=True)
    net.eval()
    data = {}
    for source in ("citizen", "semlex", "local"):
        with np.load(RAW_CACHE / f"{source}_val.npz", allow_pickle=False) as d:
            data[source] = (d["landmarks"].astype(np.float32), d["targets"])

    def batches(landmarks):
        for i in range(0, len(landmarks), BATCH):
            chunk = landmarks[i:i + BATCH]
            real = len(chunk)
            if real < BATCH:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], BATCH - real, 0)])
            yield chunk, real

    sample = (data["citizen"][0][:BATCH],)
    report = {"checkpoint": str(path.relative_to(ROOT)), "inputs": {"landmarks": [BATCH, 32, 61, 5]},
              "output": f"word logits [{BATCH},100]", "variants": {}}
    with torch.inference_mode():
        ref = {s: np.concatenate([net(torch.from_numpy(c)).numpy()[:r] for c, r in batches(v[0])]) for s, v in data.items()}
    fp32 = convert(net, sample, out / "span_recognizer_landmark_only_b8_fp32.tflite")
    for name, file in (("fp32", fp32), ("dynamic_int8", quantize(fp32, out / "span_recognizer_landmark_only_b8_int8.tflite"))):
        if file is None:
            continue
        lite = Lite(file)
        row = dict(file=str(file.relative_to(ROOT)), bytes=file.stat().st_size, sha256=sha256(file), domains={})
        for source, (landmarks, targets) in data.items():
            got = np.concatenate([lite(c)[0][:r] for c, r in batches(landmarks)])
            row["domains"][source] = dict(
                samples=int(len(targets)), top1_changes=int((got.argmax(1) != ref[source].argmax(1)).sum()),
                torch_top1=round(100 * float((ref[source].argmax(1) == targets).mean()), 3),
                lite_top1=round(100 * float((got.argmax(1) == targets).mean()), 3),
                max_abs_logit_diff=float(np.abs(got - ref[source]).max()))
        row.update(timing(file, sample))
        report["variants"][name] = row
    return report


def export_image(out: Path, count: int = 600) -> dict:
    import cv2
    from active.v17.extract_hand_rgb_v17 import decode_packed_crops
    from active.v17.extract_mobileclip2_v17 import build_encoder
    from active.v17.schema_hand_rgb_v17 import CROP_SIZE
    model, preprocess = build_encoder(torch.device("cpu"), "fp32")
    for child in model.modules():
        if hasattr(child, "fused_attn"):
            child.fused_attn = False
    tower = ImageTower(model.visual).eval()
    crops = []
    for path in sorted((MP_ROOT / "hand_rgb/citizen100_v17/landmarks/val").glob("*/*.npz")):
        with np.load(path, allow_pickle=False) as d:
            rgb = decode_packed_crops(d["jpeg_blob"], d["jpeg_offsets"], CROP_SIZE)
            for f, v in np.argwhere(d["valid"]):
                crops.append(rgb[f, v])
        if len(crops) >= count:
            break
    crops = crops[:count]
    # Same preprocessing as the Apple Stage-2 encoder: 256px RGB crops / 255, NCHW, mean 0 std 1.
    images = np.stack([c.transpose(2, 0, 1).astype(np.float32) / 255.0 for c in crops])
    sample = (images[:1],)
    with torch.inference_mode():
        ref = np.concatenate([tower(torch.from_numpy(images[i:i + 32])).numpy() for i in range(0, len(images), 32)])
        via_preprocess = model.encode_image(torch.stack([preprocess(__import__("PIL.Image").Image.fromarray(c))
                                                        for c in crops[:16]]), normalize=True).numpy()
    report = {"inputs": {"image": [1, 3, 256, 256]}, "input_scale": "RGB / 255, NCHW, no further normalization",
              "output": "L2-normalised embedding [1,512]", "crops": len(crops),
              "torch_vs_openclip_preprocess_min_cosine": float((via_preprocess * ref[:16]).sum(1).min()),
              "variants": {}}
    fp32 = convert(tower, sample, out / "mobileclip2_s0_image_fp32.tflite")
    for name, path in (("fp32", fp32), ("dynamic_int8", quantize(fp32, out / "mobileclip2_s0_image_int8.tflite"))):
        if path is None:
            continue
        lite = Lite(path)
        got = np.concatenate([lite(images[i:i + 1])[0] for i in range(len(images))])
        cosine = (got * ref).sum(1)
        report["variants"][name] = dict(file=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                                        sha256=sha256(path), cosine_min=float(cosine.min()),
                                        cosine_mean=float(cosine.mean()), **timing(path, sample, runs=20))
    return report


def export_t5(out: Path, rows: int = 200) -> dict:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from active.v17.stage3_asl_encoding_v17 import encode
    tok = AutoTokenizer.from_pretrained(STAGE3, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(STAGE3, local_files_only=True).eval()
    enc, dec = T5Encoder(model).eval(), T5DecoderStep(model).eval()
    ids = np.zeros((1, LENGTH), np.int32)
    mask = np.zeros((1, LENGTH), np.int32)
    with torch.inference_mode():
        hidden = enc(torch.from_numpy(ids), torch.from_numpy(mask)).numpy()
    start, eos = model.config.decoder_start_token_id, tok.eos_token_id
    report = {"checkpoint": str(STAGE3.relative_to(ROOT)), "length": LENGTH,
              "encoder_inputs": {"input_ids": [1, LENGTH], "attention_mask": [1, LENGTH]},
              "decoder_inputs": {"decoder_input_ids": [1, LENGTH], "encoder_hidden": list(hidden.shape),
                                 "encoder_mask": [1, LENGTH], "position": [1]},
              "decoder_output": "next token id [1] int32 (argmax)", "variants": {}}
    enc_fp32 = convert(enc, (ids, mask), out / "stage3_t5_encoder_fp32.tflite")
    dec_fp32 = convert(dec, (ids, hidden, mask, np.zeros(1, np.int32)), out / "stage3_t5_decoder_step_fp32.tflite")
    corpus = [json.loads(line) for line in STAGE3_CORPUS.open()]
    held = [r for r in corpus if r["split"] == "validation"][:rows]

    def torch_generate(text):
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=LENGTH)
        with torch.inference_mode():
            return model.generate(**inputs, max_new_tokens=LENGTH, num_beams=1, do_sample=False)[0].tolist()

    def lite_generate(encoder, decoder, text):
        piece = tok(text, truncation=True, max_length=LENGTH).input_ids
        x = np.zeros((1, LENGTH), np.int32)
        m = np.zeros((1, LENGTH), np.int32)
        x[0, :len(piece)] = piece
        m[0, :len(piece)] = 1
        h = encoder(x, m)[0]
        tokens = [start]
        while len(tokens) < LENGTH:
            d = np.zeros((1, LENGTH), np.int32)
            d[0, :len(tokens)] = tokens
            nxt = int(decoder(d, h, m, np.asarray([len(tokens) - 1], np.int32))[0].reshape(-1)[0])
            tokens.append(nxt)
            if nxt == eos:
                break
        return tokens

    texts = [encode(r["glosses"], r["confidences"]) for r in held]
    reference = [torch_generate(t) for t in texts]
    variants = [("fp32", enc_fp32, dec_fp32)]
    enc_q = quantize(enc_fp32, out / "stage3_t5_encoder_int8.tflite")
    dec_q = quantize(dec_fp32, out / "stage3_t5_decoder_step_int8.tflite")
    if enc_q and dec_q:
        variants.append(("dynamic_int8", enc_q, dec_q))
    for name, enc_path, dec_path in variants:
        encoder, decoder = Lite(enc_path), Lite(dec_path)
        started = time.perf_counter()
        outputs = [lite_generate(encoder, decoder, t) for t in texts]
        seconds = time.perf_counter() - started
        same = sum(a == b for a, b in zip(outputs, reference))
        steps = sum(len(o) - 1 for o in outputs)
        report["variants"][name] = dict(
            encoder=dict(file=str(enc_path.relative_to(ROOT)), bytes=enc_path.stat().st_size, sha256=sha256(enc_path)),
            decoder=dict(file=str(dec_path.relative_to(ROOT)), bytes=dec_path.stat().st_size, sha256=sha256(dec_path)),
            identical_outputs=f"{same}/{len(texts)}", ms_per_sentence_median_4t=round(1000 * seconds / len(texts), 1),
            ms_per_decoder_step_4t=round(1000 * seconds / max(steps, 1), 2),
            encoder_timing=timing(enc_path, (x := np.zeros((1, LENGTH), np.int32), x)),
            decoder_timing=timing(dec_path, (np.zeros((1, LENGTH), np.int32), hidden, np.zeros((1, LENGTH), np.int32),
                                             np.zeros(1, np.int32))))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "artifacts/tflite/mediapipe_v17_20261004")
    ap.add_argument("--only", nargs="+", default=["boundary", "span", "image", "t5"],
                    choices=["boundary", "span", "span_landmark", "image", "t5"])
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    config = json.loads(CONFIG.read_text())
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "format": "slt_android_tflite_manifest_v1", "config": str(CONFIG.relative_to(ROOT)),
        "timing_note": "LiteRT CPU interpreter (XNNPACK) on an Apple M4 Mac; not phone timing",
        "parity_scope": "MediaPipe validation / tuning data only; no sealed test split", "models": {}}
    steps = {"boundary": lambda: export_boundary(args.output, config), "span": lambda: export_span(args.output, config),
             "span_landmark": lambda: export_span_landmark(args.output),
             "image": lambda: export_image(args.output), "t5": lambda: export_t5(args.output)}
    for name in args.only:
        started = time.perf_counter()
        manifest["models"][name] = steps[name]()
        manifest["models"][name]["export_seconds"] = round(time.perf_counter() - started, 1)
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
        print(json.dumps({name: manifest["models"][name]}, indent=1)[:3000], flush=True)


if __name__ == "__main__":
    main()
