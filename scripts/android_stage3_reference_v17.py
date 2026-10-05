"""Python Stage-3 reference for the Android parity fixtures.

For each fixture's Python words (glosses + scores) produce the T5 greedy output the app's
render_sentence should give (no spelled words, reviewed templates disabled for this checkpoint), and
compare with the device's LiveParityActivity results (``--device-results``) when given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STAGE3 = ROOT / "artifacts/models/stage3_multisentence_tiny_v17_20260929"
LENGTH = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--device-results", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from active.v17.stage3_asl_encoding_v17 import encode
    tok = AutoTokenizer.from_pretrained(STAGE3, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(STAGE3, local_files_only=True).eval()
    tokens = json.loads((ROOT / "artifacts/coreml/stage3_multisentence_tiny_v17_20260929/stage3_tokens.json").read_text())
    assert not tokens.get("reviewed_templates_enabled", True), "templates would bypass T5"
    device = {}
    if a.device_results:
        for r in json.loads(a.device_results.read_text())["results"]:
            device[r["id"]] = r
    rows, same, compared = [], 0, 0
    for entry in json.loads((a.fixtures / "index.json").read_text()):
        fixture = json.loads((a.fixtures / entry["file"]).read_text())
        words = fixture["words"]
        if not words:
            continue
        glosses = [w["gloss"] for w in words]
        if any(g.startswith("fs-") for g in glosses):
            continue
        text = encode(glosses, [w["score"] for w in words])
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=LENGTH)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=LENGTH, num_beams=1, do_sample=False)[0]
        sentence = " ".join(tok.decode(out, skip_special_tokens=True).split())
        row = {"id": fixture["id"], "glosses": glosses, "python": sentence}
        if fixture["id"] in device and "stage3" in device[fixture["id"]]:
            row["device"] = device[fixture["id"]]["stage3"]
            row["match"] = row["device"] == sentence
            compared += 1
            same += row["match"]
        rows.append(row)
    summary = {"sentences": len(rows), "compared": compared, "identical": same}
    a.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(summary)
    for r in rows:
        if r.get("match") is False:
            print("DIFF", r["glosses"], "| py:", r["python"], "| device:", r["device"])


if __name__ == "__main__":
    main()
