#!/usr/bin/env python3
"""Serve the boundary reviewer and persist every edit to disk as it happens.

A file:// page cannot write to the filesystem, so the reviewer keeps its work in browser
storage only and has to be exported by hand. Serving it instead lets the page POST after
every change, so the corrected intervals land in a real JSON file with no save step.

  venv/bin/python scripts/serve_alignment_reviewer_v17.py

State is written atomically to
artifacts/reports/local_phrase_alignment_v17_20260923/boundary_review_corrected.json
and read back on load, so progress survives a browser restart, a cleared cache or a
different browser. Serves only this repository, on localhost.
"""
from __future__ import annotations

import json
import os
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'artifacts/reports/local_phrase_alignment_v17_20260923'
STATE = REPORT / 'boundary_review_corrected.json'
PAGE = '/artifacts/reports/local_phrase_alignment_v17_20260923/review.html'
PORT = int(os.environ.get('PORT', 8787))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):
        if '__save' not in (args[0] if args else ''):
            return                                   # keep the console readable

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/__load':
            if STATE.exists():
                try:
                    return self._json(200, json.loads(STATE.read_text()))
                except json.JSONDecodeError:
                    return self._json(200, {})
            return self._json(200, {})
        return super().do_GET()

    @staticmethod
    def _merge(payload):
        """Disk wins wherever the browser has forgotten something.

        A reload, a cleared cache or a second tab can hand us a payload whose clips are
        back to their unreviewed defaults. Writing that verbatim would erase finished work,
        so a clip already marked reviewed on disk is only replaced by another reviewed
        version of itself. Nothing here can silently reduce the number of reviewed clips.
        """
        if not STATE.exists():
            return payload
        try:
            previous = json.loads(STATE.read_text())
        except json.JSONDecodeError:
            return payload
        kept = {r['item']: r for r in previous.get('records', []) if isinstance(r, dict)}
        records = []
        for row in payload['records']:
            old = kept.get(row.get('item'))
            records.append(old if old and old.get('reviewed') and not row.get('reviewed') else row)
        return {**payload, 'records': records}

    def do_POST(self):
        if self.path != '/__save':
            return self._json(404, {'error': 'unknown endpoint'})
        try:
            raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            payload = json.loads(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            return self._json(400, {'error': str(exc)})
        if not isinstance(payload, dict) or 'records' not in payload:
            return self._json(400, {'error': 'expected an object with records'})
        REPORT.mkdir(parents=True, exist_ok=True)
        payload = self._merge(payload)
        tmp = STATE.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(payload, indent=1) + '\n')
        if STATE.exists():
            STATE.replace(STATE.with_suffix('.json.bak'))
        os.replace(tmp, STATE)                       # atomic; never a half-written file
        reviewed = sum(1 for r in payload['records'] if r.get('reviewed'))
        intervals = sum(1 for r in payload['records'] for i in r.get('intervals', []) if i)
        print('saved %d reviewed clips, %d intervals -> %s'
              % (reviewed, intervals, STATE.relative_to(ROOT)), flush=True)
        return self._json(200, {'ok': True, 'reviewed': reviewed, 'intervals': intervals})


def main():
    url = 'http://localhost:%d%s' % (PORT, PAGE)
    print('serving %s\nsaving to %s\n' % (ROOT, STATE.relative_to(ROOT)), flush=True)
    print('open: %s\n(ctrl-c to stop)\n' % url, flush=True)
    webbrowser.open(url)
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()


if __name__ == '__main__':
    main()
