#!/usr/bin/env python3
"""Index the big files under artifacts/ so an agent never reads one whole.

A single 4.2 MB result JSON is roughly a million tokens. Reading one ends a session.
This writes ONE bounded index describing what is in each oversized file — top-level
keys, scalar metrics, CSV headers — so a session reads the index and then `jq`s the
one field it needs.

    venv/bin/python scripts/index_large_artifacts_v17.py
    venv/bin/python scripts/index_large_artifacts_v17.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN = ROOT / "artifacts"
OUT = SCAN / "LARGE_FILES.md"
MIN_BYTES = 1_000_000
PARSE_CAP = 40_000_000  # above this, report size only; do not load
MAX_KEYS = 8
MAX_VAL = 60


def _scalar(v) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


def _fmt(v) -> str:
    s = json.dumps(v) if not isinstance(v, str) else v
    s = re.sub(r"\s+", " ", s).strip()
    return s[:MAX_VAL] + ("…" if len(s) > MAX_VAL else "")


def describe_json(text: str) -> str:
    """Top-level shape plus any scalar top-level values (usually the metrics)."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return "unparseable JSON"
    if isinstance(obj, list):
        inner = ""
        if obj and isinstance(obj[0], dict):
            inner = " of objects keyed " + ", ".join(f"`{k}`" for k in list(obj[0])[:MAX_KEYS])
        return f"array, {len(obj)} items{inner}"
    if not isinstance(obj, dict):
        return f"scalar {type(obj).__name__}"
    scalars = {k: v for k, v in obj.items() if _scalar(v)}
    containers = [k for k, v in obj.items() if not _scalar(v)]
    parts = []
    if scalars:
        parts.append("; ".join(f"`{k}`={_fmt(v)}" for k, v in list(scalars.items())[:MAX_KEYS]))
    if containers:
        shown = ", ".join(f"`{k}`" for k in containers[:MAX_KEYS])
        more = f" +{len(containers) - MAX_KEYS} more" if len(containers) > MAX_KEYS else ""
        parts.append(f"nested: {shown}{more}")
    return " — ".join(parts) or "empty object"


def describe_csv(text: str) -> str:
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return "empty CSV"
    rows = sum(1 for _ in reader)
    cols = ", ".join(f"`{c}`" for c in header[:MAX_KEYS])
    more = f" +{len(header) - MAX_KEYS} more" if len(header) > MAX_KEYS else ""
    return f"{rows} rows — columns: {cols}{more}"


def describe_md(text: str) -> str:
    heads = [l.strip("# ").strip() for l in text.splitlines() if l.startswith("#")]
    n = len(text.splitlines())
    return f"{n} lines — sections: " + ("; ".join(heads[:6]) or "no headings")


def describe(path: Path) -> str:
    size = path.stat().st_size
    if size > PARSE_CAP:
        return "too large to parse; inspect with `jq`/`head` only"
    text = path.read_text(errors="replace")
    if path.suffix == ".json":
        return describe_json(text)
    if path.suffix == ".csv":
        return describe_csv(text)
    return describe_md(text)


def build() -> str:
    hits = sorted(
        (p for p in SCAN.rglob("*")
         if p.is_file() and p.suffix in {".json", ".csv", ".md"}
         and p.stat().st_size >= MIN_BYTES),
        key=lambda p: -p.stat().st_size,
    )
    out = [
        "# Large artifact files — read this, not them",
        "",
        f"Every file under `artifacts/` at or above {MIN_BYTES // 1_000_000} MB — "
        f"{len(hits)} files, largest first. Smaller-but-still-costly files "
        f"(200 KB–1 MB) are not listed; find them with:",
        "",
        "```sh",
        "find artifacts -size +200k \\( -name '*.json' -o -name '*.csv' \\) | xargs ls -S",
        "```",
        "",
        "**Never read one of these whole.** The largest is ~1M tokens and will end your",
        "session. Find the field you need below, then extract just that field:",
        "",
        "```sh",
        "jq '.some_key' artifacts/reports/<dir>/result.json     # one field",
        "jq -r 'keys[]' artifacts/reports/<dir>/result.json      # what is in it",
        "head -5 artifacts/reports/<dir>/audit.csv               # CSV shape",
        "```",
        "",
        "Regenerate: `venv/bin/python scripts/index_large_artifacts_v17.py`",
        "",
        "---",
        "",
    ]
    out += ["| File | MB | Contents |", "|---|---:|---|"]
    for p in hits:
        mb = p.stat().st_size / 1e6
        desc = describe(p).replace("|", "\\|")
        if len(desc) > 260:
            desc = desc[:260].rstrip() + "…"
        out.append(f"| `{p.relative_to(ROOT)}` | {mb:.1f} | {desc} |")
    out.append("")
    return "\n".join(out) + "\n"


def selftest() -> None:
    assert "`acc`=0.9" in describe_json('{"acc":0.9,"rows":[1,2]}')
    assert "nested: `rows`" in describe_json('{"acc":0.9,"rows":[1,2]}')
    assert "array, 2 items" in describe_json("[1,2]")
    assert describe_json("{oops") == "unparseable JSON"
    assert "2 rows" in describe_csv("a,b\n1,2\n3,4\n")
    assert "columns: `a`, `b`" in describe_csv("a,b\n1,2\n")
    assert "sections: T" in describe_md("# T\nbody")
    assert describe_csv("") == "empty CSV"
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
    else:
        OUT.write_text(build())
        print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} B)")
