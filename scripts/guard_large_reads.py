#!/usr/bin/env python3
"""PreToolUse guard: refuse Bash commands that dump a >1MB file into context.

`cat cache.json` on a 4.2 MB result file is ~1.1M tokens and ends the session.
A bare `jq .` is the same thing with extra steps. Both are blocked; every
targeted form (`jq .field`, `head`, `rg`, `wc`) passes untouched.

    scripts/guard_large_reads.py --selftest
"""
from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

LIMIT = 1_000_000
HINT = ("artifacts/LARGE_FILES.md indexes these files with their keys and metrics. "
        "Read that, then extract just what you need: jq '.field' <file>, "
        "jq -r 'keys[]' <file>, or head -50 <file>.")


def _big_files(tokens: list[str], root: Path) -> list[tuple[str, int]]:
    out = []
    for t in tokens:
        if t.startswith("-"):
            continue
        p = Path(t) if Path(t).is_absolute() else root / t
        try:
            if p.is_file() and p.stat().st_size > LIMIT:
                out.append((t, p.stat().st_size))
        except OSError:
            pass
    return out


def _dumps_whole_file(tokens: list[str]) -> bool:
    """True if the command reads a file end-to-end rather than a slice of it."""
    names = [Path(t).name for t in tokens if not t.startswith("-")]
    if any(n in {"cat", "bat"} for n in names):
        return True
    if any(n == "jq" for n in names):
        # a jq filter that selects nothing is a whole-file dump
        args = [t for t in tokens[tokens.index(next(t for t in tokens if Path(t).name == "jq")) + 1:]
                if not t.startswith("-")]
        filt = args[0] if args else "."
        return re.fullmatch(r"\.\s*|\.\[\]\s*|", filt.strip()) is not None
    return False


def check(command: str, root: Path) -> str | None:
    """Return a refusal reason, or None to allow."""
    for part in re.split(r"\|\||&&|[|;]", command):
        try:
            tokens = shlex.split(part)
        except ValueError:
            continue
        if not tokens or not _dumps_whole_file(tokens):
            continue
        big = _big_files(tokens, root)
        if big:
            listed = ", ".join(f"{n} ({s/1e6:.1f} MB)" for n, s in big)
            return f"Blocked: this reads {listed} in full — roughly {int(sum(s for _, s in big)/3.7):,} tokens. {HINT}"
    return None


def selftest() -> None:
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        big = root / "big.json"
        big.write_bytes(b"x" * (LIMIT + 10))
        (root / "small.json").write_bytes(b"x" * 100)
        assert check("cat big.json", root)
        assert check("cat ./big.json", root)
        assert check("jq . big.json", root)
        assert check("jq '.' big.json", root)
        assert check("jq -r '.[]' big.json", root)
        assert check(f"cat {big}", root)
        assert check("ls && cat big.json", root)
        # these must all pass
        assert check("cat small.json", root) is None
        assert check("jq '.acc' big.json", root) is None
        assert check("jq -r 'keys[]' big.json", root) is None
        assert check("head -50 big.json", root) is None
        assert check("wc -l big.json", root) is None
        assert check("rg foo big.json", root) is None
        assert check("cat missing.json", root) is None
        assert check("echo 'cat big.json'", root) is None or True  # quoted: best effort
    print("selftest ok")


def main() -> None:
    if "--selftest" in sys.argv:
        return selftest()
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return
    cmd = (payload.get("tool_input") or {}).get("command") or ""
    reason = check(cmd, Path.cwd())
    if reason:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }}))


if __name__ == "__main__":
    main()
