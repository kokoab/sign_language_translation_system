"""Stable locked-100 Stage-2 gloss-sequence boundary for future Stage 3 consumers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


CONTRACT_PATH = Path("active/v17/stage2_to_stage3_contract_v17.json")


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    contract = json.loads(path.read_text())
    if contract.get("format") != "slt_stage2_to_stage3_contract_v17":
        raise ValueError("invalid Stage-2-to-Stage-3 contract format")
    if contract.get("version") != 1:
        raise ValueError("unsupported Stage-2-to-Stage-3 contract version")
    labels = contract.get("vocabulary", {}).get("labels")
    if not isinstance(labels, list) or len(labels) != 100 or len(set(labels)) != 100:
        raise ValueError("the Stage-2-to-Stage-3 vocabulary is not the locked 100 labels")
    return contract


def collapse_ctc_tokens(
    logits: np.ndarray,
    *,
    window_count: int,
    contract: dict[str, Any] | None = None,
) -> list[int]:
    contract = contract or load_contract()
    ctc = contract["ctc"]
    maximum_windows = int(ctc["maximum_windows"])
    tokens_per_window = int(ctc["tokens_per_window"])
    blank = int(ctc["blank_index"])
    value = np.asarray(logits)
    if value.shape != (maximum_windows * tokens_per_window, 101):
        raise ValueError(f"expected logits [64,101], got {value.shape}")
    if not 1 <= window_count <= maximum_windows:
        raise ValueError("window_count must be in [1,8]")
    raw = value[: window_count * tokens_per_window].argmax(-1).tolist()
    output: list[int] = []
    previous: int | None = None
    for token in raw:
        token = int(token)
        if token != blank and token != previous:
            output.append(token)
        previous = token
    return output


def make_stage2_output(
    *,
    utterance_id: str,
    token_indices: list[int],
    window_count: int,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = contract or load_contract()
    labels = contract["vocabulary"]["labels"]
    if not utterance_id:
        raise ValueError("utterance_id cannot be empty")
    if not 1 <= window_count <= int(contract["ctc"]["maximum_windows"]):
        raise ValueError("window_count must be in [1,8]")
    if any(not 1 <= int(token) <= 100 for token in token_indices):
        raise ValueError("Stage-2 output contains an out-of-vocabulary token")
    output = {
        "format": "slt_stage2_gloss_sequence_v17",
        "version": 1,
        "utterance_id": utterance_id,
        "token_indices": [int(token) for token in token_indices],
        "glosses": [labels[int(token) - 1] for token in token_indices],
        "window_count": int(window_count),
        "blank_index": int(contract["ctc"]["blank_index"]),
        "vocabulary_manifest_sha256": contract["vocabulary"]["manifest_sha256"],
        "recognizer_checkpoint_sha256": contract["recognizer"]["checkpoint_sha256"],
    }
    validate_stage2_output(output, contract=contract)
    return output


def validate_stage2_output(
    output: dict[str, Any], *, contract: dict[str, Any] | None = None
) -> None:
    contract = contract or load_contract()
    required = {
        "format", "version", "utterance_id", "token_indices", "glosses",
        "window_count", "blank_index", "vocabulary_manifest_sha256",
        "recognizer_checkpoint_sha256",
    }
    if set(output) != required:
        raise ValueError(f"Stage-2 output keys differ from the frozen contract: {set(output) ^ required}")
    if output["format"] != "slt_stage2_gloss_sequence_v17" or output["version"] != 1:
        raise ValueError("Stage-2 output format/version mismatch")
    if output["blank_index"] != contract["ctc"]["blank_index"]:
        raise ValueError("Stage-2 output blank-index mismatch")
    if output["vocabulary_manifest_sha256"] != contract["vocabulary"]["manifest_sha256"]:
        raise ValueError("Stage-2 output vocabulary hash mismatch")
    if output["recognizer_checkpoint_sha256"] != contract["recognizer"]["checkpoint_sha256"]:
        raise ValueError("Stage-2 output checkpoint hash mismatch")
    expected = make_stage2_output_unchecked(
        utterance_id=output["utterance_id"],
        token_indices=output["token_indices"],
        window_count=output["window_count"],
        contract=contract,
    )
    if output != expected:
        raise ValueError("Stage-2 output token/gloss mapping or bounds are invalid")


def make_stage2_output_unchecked(
    *, utterance_id: str, token_indices: list[int], window_count: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(utterance_id, str) or not utterance_id:
        raise ValueError("utterance_id cannot be empty")
    if not isinstance(token_indices, list) or any(
        not isinstance(token, int) or isinstance(token, bool) or not 1 <= token <= 100
        for token in token_indices
    ):
        raise ValueError("token_indices must contain only integers in [1,100]")
    if not isinstance(window_count, int) or isinstance(window_count, bool) or not 1 <= window_count <= 8:
        raise ValueError("window_count must be an integer in [1,8]")
    labels = contract["vocabulary"]["labels"]
    return {
        "format": "slt_stage2_gloss_sequence_v17",
        "version": 1,
        "utterance_id": utterance_id,
        "token_indices": token_indices,
        "glosses": [labels[token - 1] for token in token_indices],
        "window_count": window_count,
        "blank_index": contract["ctc"]["blank_index"],
        "vocabulary_manifest_sha256": contract["vocabulary"]["manifest_sha256"],
        "recognizer_checkpoint_sha256": contract["recognizer"]["checkpoint_sha256"],
    }
