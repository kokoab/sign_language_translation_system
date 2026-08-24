"""Fail-closed, mobile-equivalent Stage-3 rendering for the locked 100 glosses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from active.v17.stage2_stage3_contract_v17 import (
    CONTRACT_PATH,
    load_contract,
    validate_stage2_output,
)


MANIFEST_PATH = Path("active/v17/stage3_mobile_naturalizer_manifest_v17.json")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_naturalizer_manifest(
    path: Path = MANIFEST_PATH,
    *,
    contract_path: Path = CONTRACT_PATH,
) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "slt_stage3_mobile_naturalizer_manifest_v17":
        raise ValueError("invalid Stage-3 mobile naturalizer manifest format")
    if manifest.get("version") != 1:
        raise ValueError("unsupported Stage-3 mobile naturalizer manifest version")
    contract = load_contract(contract_path)
    if sha256_file(contract_path) != manifest.get("stage2_contract_sha256"):
        raise ValueError("Stage-3 manifest pins a different Stage-2 contract")
    if manifest.get("vocabulary_manifest_sha256") != contract["vocabulary"]["manifest_sha256"]:
        raise ValueError("Stage-3 vocabulary hash differs from the Stage-2 contract")
    if manifest.get("recognizer_checkpoint_sha256") != contract["recognizer"]["checkpoint_sha256"]:
        raise ValueError("Stage-3 recognizer hash differs from the Stage-2 contract")
    templates: dict[tuple[str, ...], str] = {}
    labels = set(contract["vocabulary"]["labels"])
    for row in manifest.get("reviewed_templates", []):
        glosses = row.get("glosses")
        english = row.get("english")
        if (
            not isinstance(glosses, list)
            or not glosses
            or any(not isinstance(item, str) or item not in labels for item in glosses)
            or not isinstance(english, str)
            or not english.strip()
        ):
            raise ValueError("Stage-3 manifest contains an invalid reviewed template")
        key = tuple(glosses)
        if key in templates:
            raise ValueError(f"duplicate Stage-3 reviewed template: {key}")
        templates[key] = english.strip()
    if not templates:
        raise ValueError("Stage-3 manifest must contain reviewed templates")
    return manifest


def literal_render(glosses: list[str], manifest: dict[str, Any]) -> str:
    if not glosses:
        return str(manifest["empty_output"])
    lexicon = manifest.get("literal_lexicon", {})
    words = [str(lexicon.get(gloss, gloss.lower())) for gloss in glosses]
    text = " ".join(words)
    return text[0].upper() + text[1:] + "."


def naturalize_stage2_output(
    stage2_output: dict[str, Any],
    *,
    manifest: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    manifest = manifest or load_naturalizer_manifest(manifest_path)
    contract = contract or load_contract()
    validate_stage2_output(stage2_output, contract=contract)
    glosses = list(stage2_output["glosses"])
    literal = literal_render(glosses, manifest)
    templates = {
        tuple(row["glosses"]): str(row["english"])
        for row in manifest["reviewed_templates"]
    }
    key = tuple(glosses)
    if not glosses:
        natural = str(manifest["empty_output"])
        mode = "empty"
        fallback = True
    elif key in templates:
        natural = templates[key]
        mode = "reviewed_template"
        fallback = False
    else:
        natural = literal
        mode = "literal_fallback"
        fallback = True
    return {
        "format": "slt_stage3_mobile_text_v17",
        "version": 1,
        "utterance_id": stage2_output["utterance_id"],
        "token_indices": list(stage2_output["token_indices"]),
        "glosses": glosses,
        "literal_english": literal,
        "natural_english": natural,
        "rendering_mode": mode,
        "safe_fallback_used": fallback,
        "naturalizer_manifest_sha256": sha256_file(manifest_path),
        "vocabulary_manifest_sha256": stage2_output["vocabulary_manifest_sha256"],
        "recognizer_checkpoint_sha256": stage2_output["recognizer_checkpoint_sha256"],
    }
