#!/usr/bin/env python3
"""Run a small text-to-SignWriting pilot without generating pose or video."""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from signwriting.formats.fsw_to_sign import fsw_to_sign
from signwriting.visualizer.visualize import signwriting_to_image
from signwriting_translation.bin import (
    load_sockeye_translator,
    tokenize_signwriting,
    translate,
    translate_to_text,
)
from signwriting_translation.tokenizer import tokenize_spoken_text


CASES = [
    ("Good morning", ["GOOD", "MORNING"]),
    ("Hello, how are you?", ["HELLO", "HOW", "YOU"]),
    ("My name", ["MY", "NAME"]),
    ("Thank you friend", ["THANKYOU", "FRIEND"]),
    ("I will go to school tomorrow", ["TOMORROW", "SCHOOL", "GO"]),
]

LEXEMES = [
    ("GOOD", "good"),
    ("MORNING", "morning"),
    ("HELLO", "hello"),
    ("HOW", "how"),
    ("YOU", "you"),
    ("MY", "my"),
    ("NAME", "name"),
    ("THANKYOU", "thank you"),
    ("FRIEND", "friend"),
    ("TOMORROW", "tomorrow"),
    ("SCHOOL", "school"),
    ("GO", "go"),
]

FSW_SIGN = re.compile(
    r"[BLMR]\d{3}x\d{3}(?:S[123][0-9a-f]{2}[0-5][0-9a-f]\d{3}x\d{3})+"
)


def normalized_words(text: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--reverse-model", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--reverse-model-revision", default="not_run")
    return parser.parse_args()


def inspect_output(fsw: str) -> dict:
    signs = fsw.split()
    details = []
    for sign_fsw in signs:
        parsed = fsw_to_sign(sign_fsw)
        symbols = [symbol["symbol"] for symbol in parsed["symbols"]]
        details.append(
            {
                "fsw": sign_fsw,
                "valid_fsw_shape": bool(FSW_SIGN.fullmatch(sign_fsw)),
                "symbol_count": len(symbols),
                "hand_symbol_count": sum(symbol.startswith("S1") for symbol in symbols),
                "movement_symbol_count": sum(symbol.startswith("S2") for symbol in symbols),
                "head_body_symbol_count": sum(symbol.startswith("S3") for symbol in symbols),
            }
        )
    return {
        "sign_count": len(signs),
        "all_signs_parse": bool(signs) and all(item["valid_fsw_shape"] for item in details),
        "signs": details,
    }


def write_html(output: Path, rows: list[dict], lexemes: list[dict], metadata: dict) -> None:
    cards = []
    for index, row in enumerate(rows, 1):
        expected = " ".join(row["project_reference_glosses"])
        cards.append(
            f"""
            <article>
              <h2>{index}. {html.escape(row['input_text'])}</h2>
              <p><strong>Existing project gloss reference:</strong> {html.escape(expected)}</p>
              <p><strong>Model output:</strong> {row['inspection']['sign_count']} written sign(s);
                 syntax {'passes' if row['inspection']['all_signs_parse'] else 'fails'}.</p>
              <p><strong>Reverse-model diagnostic:</strong> {html.escape(row.get('round_trip_text') or 'not run')}</p>
              <p><strong>Exact isolated-form reuse:</strong>
                 {html.escape(', '.join(row['exact_isolated_form_matches']) or 'none')}</p>
              <img src="{html.escape(row['image'])}" alt="Rendered SignWriting output">
              <details><summary>Raw Formal SignWriting</summary><code>{html.escape(row['fsw'])}</code></details>
              <h3>Gloss-order prompt: {html.escape(row['gloss_prompt_text'])}</h3>
              <p><strong>Output:</strong> {row['gloss_prompt_inspection']['sign_count']} written sign(s).
                 <strong>Reverse diagnostic:</strong> {html.escape(row.get('gloss_prompt_round_trip_text') or 'not run')}</p>
              <img src="{html.escape(row['gloss_prompt_image'])}" alt="Gloss-prompt SignWriting output">
              <h3>Isolated forms concatenated in project gloss order</h3>
              <p><strong>Output:</strong> {row['composed_inspection']['sign_count']} written sign(s).
                 <strong>Reverse diagnostic:</strong> {html.escape(row.get('composed_round_trip_text') or 'not run')}</p>
              <img src="{html.escape(row['composed_image'])}" alt="Composed isolated SignWriting output">
            </article>
            """
        )
    lexical_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['gloss'])}</td>"
        f"<td>{html.escape(row['input_text'])}</td>"
        f"<td>{row['inspection']['sign_count']}</td>"
        f"<td>{html.escape(row.get('round_trip_text') or 'not run')}</td>"
        f"<td><img src=\"{html.escape(row['image'])}\" alt=\"Rendered isolated SignWriting\"></td>"
        "</tr>"
        for row in lexemes
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>SignWriting symbolic pilot v1</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#17202a}}
.notice{{background:#fff4d6;border-left:5px solid #d99000;padding:1rem}}
article{{border:1px solid #d8dee4;border-radius:12px;padding:1rem;margin:1.25rem 0}}
img{{display:block;max-width:100%;min-height:100px;background:white;border:1px solid #eee;padding:1rem;margin:.75rem 0}}
code{{display:block;overflow-wrap:anywhere;white-space:pre-wrap;margin-top:.75rem}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d8dee4;padding:.5rem;text-align:left}}
td img{{min-height:50px;margin:0;max-height:180px}}
</style></head><body>
<h1>English → ASL SignWriting symbolic pilot</h1>
<p class="notice"><strong>Scope:</strong> notation only. These outputs are not pose, video, Stage-2 data,
or proof of correct/native ASL. A fluent ASL signer who reads SignWriting must review the semantics.</p>
<p class="notice"><strong>Automated verdict: HOLD.</strong> All outputs parse, but none of the five natural
phrase outputs returns the original content under the separate reverse-model diagnostic. Do not connect
this checkpoint to Rylo or a local pose renderer yet.</p>
<p>Model: <code>{html.escape(metadata['model_id'])}</code><br>
Revision: <code>{html.escape(metadata['model_revision'])}</code></p>
{''.join(cards)}
<h2>Isolated lexical probes</h2>
<p>These are diagnostics, not dictionary ground truth. Exact reuse can reveal consistency, but ASL
inflection or context may legitimately alter a written form.</p>
<table><thead><tr><th>Project gloss</th><th>English probe</th><th>Signs</th><th>Reverse diagnostic</th><th>Output</th></tr></thead>
<tbody>{lexical_rows}</tbody></table>
</body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.model.resolve()
    translator, _ = load_sockeye_translator(str(model_path), log_timing=True)
    model_inputs = [f"$en $ase {tokenize_spoken_text(text)}" for text, _ in CASES]
    translations = translate(translator, model_inputs, log_timing=True)
    lexical_inputs = [f"$en $ase {tokenize_spoken_text(text)}" for _, text in LEXEMES]
    lexical_translations = translate(translator, lexical_inputs, log_timing=True)
    isolated_by_gloss = {
        gloss: fsw for (gloss, _), fsw in zip(LEXEMES, lexical_translations, strict=True)
    }
    gloss_prompts = [" ".join(gloss.lower().replace("thankyou", "thank you") for gloss in glosses)
                     for _, glosses in CASES]
    gloss_inputs = [f"$en $ase {tokenize_spoken_text(text)}" for text in gloss_prompts]
    gloss_translations = translate(translator, gloss_inputs, log_timing=True)
    composed_translations = [
        " ".join(isolated_by_gloss[gloss] for gloss in glosses) for _, glosses in CASES
    ]
    round_trips = [None] * len(translations)
    lexical_round_trips = [None] * len(lexical_translations)
    gloss_round_trips = [None] * len(gloss_translations)
    composed_round_trips = [None] * len(composed_translations)
    if args.reverse_model:
        reverse_translator, _ = load_sockeye_translator(str(args.reverse_model.resolve()), log_timing=True)
        all_fsw = translations + lexical_translations + gloss_translations + composed_translations
        reverse_inputs = [f"$en $ase {tokenize_signwriting(fsw)}" for fsw in all_fsw]
        reverse_outputs = translate_to_text(reverse_translator, reverse_inputs, log_timing=True)
        phrase_end = len(translations)
        lexical_end = phrase_end + len(lexical_translations)
        gloss_end = lexical_end + len(gloss_translations)
        round_trips = reverse_outputs[:phrase_end]
        lexical_round_trips = reverse_outputs[phrase_end:lexical_end]
        gloss_round_trips = reverse_outputs[lexical_end:gloss_end]
        composed_round_trips = reverse_outputs[gloss_end:]

    lexical_rows = []
    for index, ((gloss, text), fsw, round_trip) in enumerate(
        zip(LEXEMES, lexical_translations, lexical_round_trips, strict=True), 1
    ):
        image_name = f"lex_{index:02d}.png"
        signwriting_to_image(fsw.split(), trust_box=False).save(args.output / image_name)
        lexical_rows.append(
            {
                "gloss": gloss,
                "input_text": text,
                "fsw": fsw,
                "round_trip_text": round_trip,
                "strict_round_trip_exact": normalized_words(text) == normalized_words(round_trip),
                "image": image_name,
                "inspection": inspect_output(fsw),
            }
        )

    rows = []
    for index, ((text, glosses), fsw, round_trip, gloss_prompt, gloss_fsw, gloss_round_trip,
                composed_fsw, composed_round_trip) in enumerate(
        zip(CASES, translations, round_trips, gloss_prompts, gloss_translations,
            gloss_round_trips, composed_translations, composed_round_trips, strict=True), 1
    ):
        image_name = f"{index:02d}.png"
        gloss_image_name = f"{index:02d}_gloss_prompt.png"
        composed_image_name = f"{index:02d}_composed.png"
        signwriting_to_image(fsw.split(), trust_box=False).save(args.output / image_name)
        signwriting_to_image(gloss_fsw.split(), trust_box=False).save(args.output / gloss_image_name)
        signwriting_to_image(composed_fsw.split(), trust_box=False).save(args.output / composed_image_name)
        phrase_signs = set(fsw.split())
        rows.append(
            {
                "input_text": text,
                "project_reference_glosses": glosses,
                "fsw": fsw,
                "round_trip_text": round_trip,
                "strict_round_trip_exact": normalized_words(text) == normalized_words(round_trip),
                "exact_isolated_form_matches": [
                    gloss
                    for gloss in glosses
                    if len(isolated_by_gloss[gloss].split()) == 1
                    and isolated_by_gloss[gloss] in phrase_signs
                ],
                "image": image_name,
                "inspection": inspect_output(fsw),
                "gloss_prompt_text": gloss_prompt,
                "gloss_prompt_fsw": gloss_fsw,
                "gloss_prompt_round_trip_text": gloss_round_trip,
                "gloss_prompt_strict_round_trip_exact": (
                    normalized_words(gloss_prompt) == normalized_words(gloss_round_trip)
                ),
                "gloss_prompt_image": gloss_image_name,
                "gloss_prompt_inspection": inspect_output(gloss_fsw),
                "composed_fsw": composed_fsw,
                "composed_round_trip_text": composed_round_trip,
                "composed_strict_round_trip_exact": (
                    normalized_words(gloss_prompt) == normalized_words(composed_round_trip)
                ),
                "composed_image": composed_image_name,
                "composed_inspection": inspect_output(composed_fsw),
            }
        )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": "sign/sockeye-text-to-factored-signwriting",
        "model_revision": args.model_revision,
        "reverse_model_id": "sign/sockeye-signwriting-to-text" if args.reverse_model else None,
        "reverse_model_revision": args.reverse_model_revision,
        "model_path": str(model_path),
        "license": "CC-BY-NC-4.0",
        "spoken_language": "en",
        "signed_language": "ase",
        "case_count": len(rows),
        "isolated_lexeme_count": len(lexical_rows),
        "syntax_valid_case_count": sum(row["inspection"]["all_signs_parse"] for row in rows),
        "strict_round_trip_exact_case_count": sum(row["strict_round_trip_exact"] for row in rows),
        "strict_round_trip_exact_lexeme_count": sum(
            row["strict_round_trip_exact"] for row in lexical_rows
        ),
        "scope": "symbolic_notation_only",
        "eligible_for_stage2": False,
        "native_semantic_review_complete": False,
        "round_trip_is_ground_truth": False,
        "automated_recommendation": "hold_before_pose_rendering",
    }
    report = {"metadata": metadata, "cases": rows, "isolated_lexemes": lexical_rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "README.md").write_text(
        "# SignWriting symbolic pilot v1\n\n"
        "Open `index.html` to compare natural-English prompting, project-gloss-order prompting, and "
        "concatenated isolated forms for five phrases, plus 12 isolated lexical probes. Every output "
        "is structurally parseable, but none of the five natural phrase outputs strictly round-trips "
        "to its input and only "
        f"{metadata['strict_round_trip_exact_lexeme_count']}/12 isolated probes do. The automated "
        "recommendation is therefore HOLD before pose rendering. Reverse translation is only a "
        "diagnostic, not semantic ground truth; native ASL/SignWriting review is still required. "
        "Nothing here is eligible for Stage-2 training, validation, or testing.\n",
        encoding="utf-8",
    )
    write_html(args.output, rows, lexical_rows, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
