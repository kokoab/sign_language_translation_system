#!/usr/bin/env python3
"""The app's session history: one folder per run, one index over all of them.

Sessions live under `artifacts/app_sessions/` rather than `artifacts/reports/`, which
holds experiment provenance.  A demo run is not an experiment result and must not be
mistaken for one.

The live page writes each session with the same `SessionRecorder` the standalone reel
script uses, so the on-disk format is unchanged.  This module only reads those files
back and keeps a small index, because parsing every history on every visit to the
history page would make it feel slow.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SESSION_ROOT = REPO / "artifacts/app_sessions"
INDEX_NAME = "index.json"
INDEX_FORMAT = "slt_v17_app_session_index"


def _relative(path: Path) -> str:
    """Repo-relative where possible; a session root may sit anywhere."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _parse(stamp: str | None) -> datetime | None:
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _duration(history: dict) -> float:
    started, finished = _parse(history.get("started_utc")), _parse(
        history.get("finished_utc")
    )
    if started and finished:
        return max(0.0, (finished - started).total_seconds())
    stamps = history.get("video_source_timestamps_seconds") or []
    if len(stamps) >= 2:
        try:
            return max(0.0, float(stamps[-1]) - float(stamps[0]))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _spoken(history: dict) -> list[dict]:
    """Utterances that actually reached the screen, in order."""
    return [
        item for item in history.get("utterances") or []
        if isinstance(item, dict) and item.get("displayed_and_spoken")
    ]


def _glosses(history: dict) -> list[str]:
    spoken = _spoken(history)
    if spoken:
        out: list[str] = []
        for item in spoken:
            out.extend(str(g) for g in item.get("glosses") or [])
        if out:
            return out
    # Nothing was finished: fall back to whatever the session committed.
    return [
        str(row["committed_gloss"])
        for row in history.get("predictions") or []
        if isinstance(row, dict) and row.get("committed_gloss")
    ]


def summarize(folder: Path) -> dict | None:
    """One history folder reduced to a history-page row, or None if unreadable."""
    history_path = folder / "history.json"
    try:
        history = json.loads(history_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(history, dict):
        return None

    spoken = _spoken(history)
    video = folder / "session_lowres.mp4"
    stat = history_path.stat()
    return {
        "stamp": folder.name,
        "path": _relative(folder),
        "started_utc": history.get("started_utc"),
        "finished_utc": history.get("finished_utc"),
        "complete": bool(history.get("finished_utc")),
        "duration_seconds": round(_duration(history), 3),
        "glosses": _glosses(history),
        "sentences": [str(item.get("sentence") or "") for item in spoken],
        "utterance_count": len(history.get("utterances") or []),
        "prediction_count": len(history.get("predictions") or []),
        "source": history.get("source"),
        "mode": history.get("mode"),
        "score_semantics": history.get("score_semantics"),
        "test_accessed": bool(history.get("test_accessed", False)),
        "video": _relative(video) if video.is_file() else None,
        "history_mtime": stat.st_mtime,
        "history_size": stat.st_size,
    }


def _rows(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [
        folder for folder in root.iterdir()
        if folder.is_dir() and (folder / "history.json").is_file()
    ]


def build_index(root: Path = SESSION_ROOT, previous: dict | None = None) -> dict:
    """Re-summarise only the sessions whose history changed since last time."""
    cached = {
        row.get("stamp"): row
        for row in (previous or {}).get("sessions", [])
        if isinstance(row, dict)
    }
    sessions = []
    for folder in _rows(root):
        known = cached.get(folder.name)
        stat = (folder / "history.json").stat()
        if (
            known
            and known.get("history_mtime") == stat.st_mtime
            and known.get("history_size") == stat.st_size
        ):
            sessions.append(known)
            continue
        row = summarize(folder)
        if row is not None:
            sessions.append(row)
    sessions.sort(key=lambda row: str(row.get("stamp") or ""), reverse=True)
    return {
        "format": INDEX_FORMAT,
        "version": 1,
        "root": _relative(root),
        "generated_utc": datetime.now().astimezone().isoformat(),
        "session_count": len(sessions),
        "sessions": sessions,
    }


def load_index(root: Path = SESSION_ROOT) -> dict:
    path = root / INDEX_NAME
    try:
        previous = json.loads(path.read_text())
    except (OSError, ValueError):
        previous = None
    if not isinstance(previous, dict) or previous.get("format") != INDEX_FORMAT:
        previous = None
    return previous or {"sessions": []}


def refresh(root: Path = SESSION_ROOT, *, write: bool = True) -> dict:
    """The one call the shell makes: reconcile the index with what is on disk."""
    index = build_index(root, load_index(root))
    if write and root.is_dir():
        tmp = root / f".{INDEX_NAME}.tmp"
        tmp.write_text(json.dumps(index, indent=1) + "\n")
        tmp.replace(root / INDEX_NAME)
    return index


def describe(row: dict) -> tuple[str, str, str]:
    """Row text for the history page: when, how long, and what was said."""
    started = _parse(row.get("started_utc"))
    when = started.astimezone().strftime("%d %b %Y · %H:%M") if started else row.get(
        "stamp", "unknown"
    )
    seconds = float(row.get("duration_seconds") or 0.0)
    length = f"{int(seconds) // 60}:{int(seconds) % 60:02d}"
    sentences = [text for text in row.get("sentences") or [] if text]
    if sentences:
        said = sentences[-1]
    elif row.get("glosses"):
        said = " ".join(row["glosses"])
    else:
        said = "No recognized signs"
    return when, length, said


if __name__ == "__main__":
    summary = refresh()
    print(f"{summary['session_count']} sessions in {summary['root']}")
    for row in summary["sessions"][:10]:
        when, length, said = describe(row)
        print(f"  {when}  {length}  {said}")
