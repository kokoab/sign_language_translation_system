#!/usr/bin/env python3
"""Summarize iPhone Stage 3 latency results for real output lengths.

Output lengths come from the multi-sentence evaluation set (T5 tokenizer, EOS included):
one sentence p90 = 11 tokens, whole session median = 19, p90 = 28. Estimated latency is
first-graph median + N x per-token median, per model and compute-unit setting.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/reports/stage3_latency_bench_v17_20260929"
LENGTHS = {"sentence_p90_11": 11, "session_median_19": 19, "session_p90_28": 28}
PUBLISHED_PARAMS_M = {
    "deployed_v2": 15.6, "t5_tiny_v2": 15.6, "t5_small": 60.5, "flan_t5_small": 77.0,
    "flan_t5_base": 247.6, "smollm2_135m": 134.5, "smollm2_360m": 361.8,
    "gemma3_270m": 268.1, "qwen25_05b": 494.0,
}


def main():
    rows = []
    for path in sorted(REPORT.glob("results_*.json")):
        rows += json.loads(path.read_text())
    table = []
    for r in rows:
        base = next(k for k in PUBLISHED_PARAMS_M if r["tag"].startswith(k))
        entry = {"tag": r["tag"], "units": r["units"], "params_m": PUBLISHED_PARAMS_M[base],
                 "weight_bits": r.get("weight_bits"), "first_ms": round(r["first_ms_median"], 1),
                 "token_ms": round(r["step_ms_median"], 1), "token_ms_p90": round(r["step_ms_p90"], 1),
                 "measured_16_ms": round(r["total_16_steps_ms_median"]),
                 "measured_48_ms": round(r["total_48_steps_ms_median"])}
        for name, n in LENGTHS.items():
            entry[f"est_{name}_ms"] = round(r["first_ms_median"] + n * r["step_ms_median"])
        table.append(entry)
    table.sort(key=lambda e: (e["est_session_p90_28_ms"]))
    (REPORT / "summary.json").write_text(json.dumps(table, indent=2))
    head = ["tag", "units", "params_m", "first_ms", "token_ms", "est_sentence_p90_11_ms",
            "est_session_median_19_ms", "est_session_p90_28_ms", "measured_48_ms"]
    print("\t".join(head))
    for e in table:
        print("\t".join(str(e[h]) for h in head))


if __name__ == "__main__":
    main()
