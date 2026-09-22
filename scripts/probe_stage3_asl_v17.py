#!/usr/bin/env python3
"""Side-by-side probe of the deployed and retrained Stage-3 renderers.

Held-out BLEU measures agreement with generated English. It cannot say whether the
reported live failures are fixed, so this probe runs both checkpoints over two sets the
generated corpus never contained:

  * every gloss buffer the user actually produced, read from artifacts/app_sessions/
  * a fixed ASL-structure set covering the orders the deployed model mishandles

There is no automatic score here. The output is for reading.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from active.v17.stage3_asl_encoding_v17 import encode  # noqa: E402

DEPLOYED = ROOT / "artifacts/models/stage3_v17_t5_efficient_tiny_locked100_v1"
RETRAINED = ROOT / "artifacts/models/stage3_v17_asl_order_v1"
SESSIONS = ROOT / "artifacts/app_sessions"

# Structures the deployed checkpoint gets wrong, with the reading each one should get.
STRUCTURE_PROBES: tuple[tuple[str, str], ...] = (
    ("I TIRED", "I am tired"),
    ("WATER I WANT", "I want water"),
    ("HOME I GO NOW", "I am going home now"),
    ("DOCTOR I GO TOMORROW", "I am going to the doctor tomorrow"),
    ("SCHOOL I GO", "I go to school"),
    ("YOUR NAME WHAT", "What is your name"),
    ("YOU GO WHERE", "Where are you going"),
    ("I NO LIKE WATER", "I do not like water"),
    ("MY MOTHER SICK", "My mother is sick"),
    ("TOMORROW I GO WORK", "I am going to work tomorrow"),
    ("YOU HUNGRY", "Are you hungry"),
    ("I WANT EAT", "I want to eat"),
    ("FRIEND I HELP", "I help my friend"),
    ("YESTERDAY MY FATHER SICK", "My father was sick yesterday"),
    ("WATER COLD", "The water is cold"),
    # Verbless wh + noun-phrase questions, which dominate the real sessions. The
    # locked vocabulary has no copula, so the renderer must supply one rather than
    # invent an action.
    ("WHO YOU", "Who are you"),
    ("WHERE YOUR FAMILY", "Where is your family"),
    ("WHAT TIME TOMORROW", "What is the time tomorrow"),
    ("HOW YOUR DAY", "How is your day"),
    ("WHAT WORK TIME", "What is the work time"),
)


# Buffers the user actually produced where a gloss looks like a recognizer error, with
# the suspect gloss marked at the low confidence such errors carried in the sessions.
# The deployed renderer has no way to receive this and renders every one of them.
NOISE_PROBES: tuple[tuple[str, int, str], ...] = (
    ("I GO LESS SCHOOL TOMORROW MORNING", 2, "I am going to school tomorrow morning"),
    ("I UNDERSTAND SIGN TIME LANGUAGE WORK TIME", 3, "I understand sign language work time"),
    ("WHO YOU HAPPY I UNDERSTAND YEAR SIGN LANGUAGE", 5, "who are you? I understand sign language"),
    ("I FEEL SICK TIRED HUNGRY", -1, "I feel sick, tired and hungry (nothing to drop)"),
    ("I TIRED TIME", 2, "I am tired"),
    ("HELLO HOW YOU MAYBE", 3, "hello, how are you?"),
    ("I WANT EAT STOP", 3, "I want to eat"),
    ("MY MOTHER SAME SICK", 2, "my mother is sick"),
    # Observed live 2026-09-22, session 20260922_122517_239185: MY was recognized at
    # 0.91 with no noun after it, and the renderer answered "my family is hungry",
    # inventing a noun the signer never produced. A stranded determiner has to be
    # dropped despite high confidence.
    ("I SICK MY HUNGRY", 2, "I am sick and hungry"),
    ("I COLD YOUR TIRED", 2, "I am cold and tired"),
    ("HE HAPPY OUR", 2, "he is happy"),
)


def load_session_buffers() -> list[tuple[str, list[str], list[float] | None]]:
    """Real gloss buffers with per-gloss commit scores where the session recorded them."""
    rows: list[tuple[str, list[str], list[float] | None]] = []
    for path in sorted(glob.glob(str(SESSIONS / "*/history.json"))):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        scores: dict[str, list[float]] = {}
        for prediction in payload.get("predictions", []):
            gloss = prediction.get("committed_gloss")
            score = prediction.get("model_score")
            if gloss and score is not None:
                scores.setdefault(str(gloss), []).append(float(score))
        for utterance in payload.get("utterances", []):
            glosses = utterance.get("input_glosses") or utterance.get("glosses") or []
            if not glosses:
                continue
            # Sessions store scores per prediction, not per finished gloss; reuse the
            # recorded score for each gloss where one exists, else leave it unknown.
            confidences: list[float] | None = []
            for gloss in glosses:
                pool = scores.get(str(gloss))
                if not pool:
                    confidences = None
                    break
                confidences.append(pool[0])
            rows.append((Path(path).parent.name, [str(g) for g in glosses], confidences))
    return rows


def load(checkpoint: Path, device: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint, local_files_only=True
    ).to(torch.device(device)).eval()
    return tokenizer, model


def render(tokenizer, model, text: str, device: str) -> str:
    import torch

    encoded = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=64
    ).to(torch.device(device))
    with torch.inference_mode():
        out = model.generate(**encoded, max_new_tokens=64, num_beams=1, do_sample=False)
    return " ".join(tokenizer.decode(out[0].cpu(), skip_special_tokens=True).split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployed", type=Path, default=DEPLOYED)
    parser.add_argument("--retrained", type=Path, default=RETRAINED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    old_tokenizer, old_model = load(args.deployed, args.device)
    new_available = args.retrained.exists()
    if new_available:
        new_tokenizer, new_model = load(args.retrained, args.device)
    else:
        print(f"note: {args.retrained} does not exist yet; showing deployed only\n")

    records: list[dict] = []

    print("=" * 100)
    print("ASL STRUCTURE PROBES")
    print("=" * 100)
    for gloss, intended in STRUCTURE_PROBES:
        glosses = gloss.split()
        old = render(old_tokenizer, old_model, gloss.lower(), args.device)
        new = (
            render(new_tokenizer, new_model, encode(glosses), args.device)
            if new_available else None
        )
        print(f"\n{gloss}")
        print(f"   should mean : {intended}")
        print(f"   deployed    : {old}")
        if new is not None:
            print(f"   retrained   : {new}")
        records.append({
            "set": "structure", "glosses": glosses, "intended": intended,
            "deployed": old, "retrained": new,
        })

    print("\n" + "=" * 100)
    print("NOISE SUPPRESSION PROBES  (low-confidence gloss marked with *)")
    print("=" * 100)
    for gloss, noise_index, intended in NOISE_PROBES:
        glosses = gloss.split()
        # A spurious gloss gets a score from the band where real false positives
        # landed. A stranded determiner instead keeps the high score it actually had
        # live, because the reason to drop it is grammatical, not evidential.
        stranded = (
            noise_index >= 0 and glosses[noise_index] in {"MY", "YOUR", "OUR"}
        )
        confidences = [
            (0.91 if stranded else 0.29) if i == noise_index else 0.78
            for i in range(len(glosses))
        ]
        shown = " ".join(
            f"*{g}" if i == noise_index else g for i, g in enumerate(glosses)
        )
        old = render(old_tokenizer, old_model, gloss.lower(), args.device)
        new = (
            render(new_tokenizer, new_model, encode(glosses, confidences), args.device)
            if new_available else None
        )
        print(f"\n{shown}")
        print(f"   should mean : {intended}")
        print(f"   deployed    : {old}")
        if new is not None:
            print(f"   retrained   : {new}")
        records.append({
            "set": "noise", "glosses": glosses, "confidences": confidences,
            "noise_index": noise_index, "intended": intended,
            "deployed": old, "retrained": new,
        })

    print("\n" + "=" * 100)
    print("REAL SESSION BUFFERS")
    print("=" * 100)
    for session, glosses, confidences in load_session_buffers():
        old = render(old_tokenizer, old_model, " ".join(glosses).lower(), args.device)
        new = (
            render(new_tokenizer, new_model, encode(glosses, confidences), args.device)
            if new_available else None
        )
        marker = "" if confidences else "   (no per-gloss scores recorded)"
        print(f"\n{' '.join(glosses)}{marker}")
        print(f"   deployed    : {old}")
        if new is not None:
            print(f"   retrained   : {new}")
        records.append({
            "set": "session", "session": session, "glosses": glosses,
            "confidences": confidences, "deployed": old, "retrained": new,
        })

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8"
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
