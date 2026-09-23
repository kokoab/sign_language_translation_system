#!/usr/bin/env python3
"""Build the local-phrase boundary reviewer.

Every local phrase training clip, preloaded with the intervals the frozen DGS teacher
proposed and both isolated recognisers verified. 232 clips, 631 sign tokens, of which 328
carry a machine interval and 303 carry none. The reviewer is for correcting all of them.

Follows the convention established by the Luna pilot
(artifacts/reports/luna_boundary_annotation_pilot_20260921): the lexical sign interval
EXCLUDES preparatory movement and release, and a final hold may fall outside it. That
pilot measured a human reviewer against ASLLRP at 68 ms median start error and 145 ms
median end error, so perfect agreement is not the bar.

Emits a single self-contained page that reads the videos from disk by relative path.
Output is a corrected-intervals JSON in the boundary recipe's own interval format.
"""
from __future__ import annotations

import json
import os
import subprocess
from html import escape
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'artifacts/reports/local_phrase_alignment_v17_20260923'
COMBINED = ROOT / 'data/local/combined_dataset_v17_20260922/manifest.json'
CACHE = ROOT / 'artifacts/cache/local_phrase_teacher_segments_v17'
CLIPS = REPORT / 'clips'


def browser_copy(source, sha, fps, frames):
    """Transcode to H.264 for the browser, preserving frame count and rate exactly.

    The source clips are MPEG-4 Part 2 (fourcc mp4v, OpenCV's default writer). Safari
    plays it; Chrome and Brave do not support the codec at all, so the video element
    fails silently. Re-encoding to H.264 keeps every frame and the frame rate, which
    matters because the reviewer's interval timestamps are in seconds on this clock.
    """
    CLIPS.mkdir(parents=True, exist_ok=True)
    out = CLIPS / (sha + '.mp4')
    if not out.exists():
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(source),
                        '-c:v', 'libx264', '-crf', '20', '-preset', 'veryfast',
                        '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart', str(out)],
                       check=True)
    capture = cv2.VideoCapture(str(out))
    got_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    got_fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if got_frames != frames or abs(got_fps - fps) > 1e-3:
        raise ValueError('transcode changed the clock for %s: %d/%.4f vs %d/%.4f'
                         % (sha, got_frames, got_fps, frames, fps))
    return 'clips/' + sha + '.mp4'


def clip_rows():
    combined = json.loads(COMBINED.read_text())
    rows = sorted([r for r in combined['records']
                   if r.get('source') == 'local_phrases' and r['role'] == 'train'],
                  key=lambda r: r['source_item_id'])
    alignment = json.loads((REPORT / 'alignment.json').read_text())
    aligned = {r['video_sha256']: r for r in alignment['records']}
    out = []
    for row in rows:
        path = ROOT / row['video_path']
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        fps = fps if fps and fps > 1 else 30.
        playable = browser_copy(path, row['video_sha256'], fps, frames)
        match = aligned.get(row['video_sha256'])
        glosses = row['target_sequence']
        intervals = [None] * len(glosses)
        scores = [None] * len(glosses)
        if match:
            for a in match['aligned']:
                intervals[a['index']] = [round(a['start'], 3), round(a['end'], 3)]
                scores[a['index']] = dict(proposal=round(a['proposal_score'], 3),
                                          verifier=round(a['verifier_score'], 3))
        cached = CACHE / (row['video_sha256'] + '.json')
        candidates = []
        if cached.exists():
            data = json.loads(cached.read_text())
            if data.get('labels'):
                from scripts.align_local_phrases_v17 import decode_segments
                candidates = [[round(s['start'], 3), round(s['end'], 3)]
                              for s in decode_segments(data['labels'], data['fps'])]
        out.append(dict(item=row['source_item_id'], sha=row['video_sha256'],
                        video=playable, signer=row['signer_id'],
                        glosses=glosses, fps=round(fps, 4),
                        duration=round(frames / fps, 3) if frames else 0.,
                        intervals=intervals, scores=scores, candidates=candidates))
    return out


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local phrase boundary review</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22%3E%3Crect width=%2216%22 height=%2216%22 rx=%223%22 fill=%22%234d9bff%22/%3E%3C/svg%3E">
<style>
:root{color-scheme:dark;
--bg:#0e1420;--panel:#18202e;--panel2:#202b3c;--text:#f2f6fc;--muted:#9fb0c6;--line:#3a4a63;
--accent:#4d9bff;--machine:#ff9f3f;--ok:#2fbd7a;--warn:#ffc043;--bad:#ff6b76;--cand:#5c6b82}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,sans-serif}
.app{display:grid;grid-template-columns:280px 1fr;height:100vh}
aside{border-right:1px solid var(--line);background:var(--panel);overflow-y:auto}
aside h1{font-size:16px;margin:0;padding:16px 16px 6px}
.prog{padding:0 16px 12px;color:var(--muted);font-size:12.5px}
.clip{display:flex;justify-content:space-between;gap:8px;padding:9px 16px;cursor:pointer;
border-top:1px solid var(--line);font-size:13.5px;color:var(--text)}
.clip:hover{background:var(--panel2)}
.clip.on{background:#26456e;font-weight:650}
.clip .tag{color:var(--muted);font-size:11.5px}
.st{font-size:11.5px;padding:2px 8px;border-radius:999px;background:#2c3850;color:var(--muted);white-space:nowrap}
.st.full{background:#12503a;color:#7ee9b4}.st.part{background:#54410c;color:#ffd97a}
.st.none{background:#5a2126;color:#ffadb3}.st.done{background:var(--accent);color:#04101f}
main{overflow-y:auto;padding:16px 22px 40px}
.top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}
h2{margin:0;font-size:24px;letter-spacing:.2px}
.sub{color:var(--muted);font-size:13px;margin-top:3px}
.split{display:grid;grid-template-columns:minmax(360px,42%) minmax(0,1fr);gap:20px;margin-top:14px;align-items:start}
.split>div{min-width:0}
video{width:100%;max-height:46vh;background:#000;border-radius:10px;display:block}
.clock{font:13.5px ui-monospace,SFMono-Regular,monospace;color:var(--text);margin-top:8px}
.clock b{color:var(--accent)}
.zoom{display:flex;align-items:center;gap:10px;margin:2px 0 10px;color:var(--muted);font-size:13px}
.zoom input{flex:1}
.ruler{position:relative;height:18px;margin-bottom:2px;color:var(--muted);font-size:11px;width:calc(100% * var(--zoom,1))}
.ruler i{position:absolute;top:0;width:1px;height:6px;background:var(--line)}
.ruler span{position:absolute;top:6px;transform:translateX(-50%);white-space:nowrap}
.sign{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:12px 14px;margin-bottom:11px}
.sign.sel{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.sign header{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:9px;flex-wrap:wrap}
.sign h3{margin:0;font-size:17px}
.sign h3 em{font-style:normal;color:var(--muted);font-weight:400;font-size:13.5px;margin-left:8px}
.tlwrap{overflow-x:auto;overflow-y:hidden;min-width:0;padding-bottom:4px}
.tl{position:relative;height:46px;background:var(--panel2);border:1px solid var(--line);
border-radius:8px;user-select:none;cursor:crosshair;width:calc(100% * var(--zoom,1))}
.cand{position:absolute;top:4px;bottom:4px;background:#5c6b8240;border:1px dashed var(--cand);border-radius:5px}
.bar{position:absolute;top:3px;bottom:3px;background:#4d9bffcc;border:1px solid var(--accent);
border-radius:6px;cursor:grab}
.bar.machine{background:#ff9f3fbb;border-color:var(--machine)}
.bar.short{background:#ffc043cc;border-color:var(--warn)}
.warn{color:var(--warn);font-weight:700}
.bar b{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);font-size:12.5px;
color:#04101f;font-weight:700;pointer-events:none;white-space:nowrap}
.h{position:absolute;top:-3px;bottom:-3px;width:20px;cursor:ew-resize;display:flex;align-items:center;justify-content:center}
.h::after{content:"";width:5px;height:60%;background:#fff;border-radius:3px;box-shadow:0 0 0 1px #0006}
.h.l{left:-10px}.h.r{right:-10px}
.ph{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--bad);z-index:3;cursor:ew-resize}
.ph::before{content:"";position:absolute;left:-9px;right:-9px;top:0;bottom:0}
.ph::after{content:"";position:absolute;left:-5px;top:-5px;width:12px;height:12px;border-radius:50%;
background:var(--bad);box-shadow:0 1px 3px #0008}
.ruler{cursor:ew-resize}
.nums{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:9px;font-size:13px;color:var(--muted)}
.nums label{display:flex;align-items:center;gap:6px;color:var(--text)}
.nums input{width:88px;font:13.5px ui-monospace,monospace;padding:5px 7px;border:1px solid var(--line);
border-radius:7px;background:var(--panel2);color:var(--text)}
.v{display:flex;gap:5px}
.v button{font:inherit;font-size:12.5px;padding:5px 11px;border:1px solid var(--line);
background:var(--panel2);color:var(--text);border-radius:7px;cursor:pointer}
.v button:hover{border-color:var(--accent)}
.v button.on[data-v=accept]{background:var(--ok);color:#04140d;border-color:var(--ok);font-weight:700}
.v button.on[data-v=reject]{background:var(--bad);color:#210609;border-color:var(--bad);font-weight:700}
.v button.on[data-v=unsure]{background:var(--warn);color:#241a00;border-color:var(--warn);font-weight:700}
.acts{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 12px}
.io{align-self:center;font-size:12.5px;padding:5px 11px;border-radius:999px;
background:#2c3850;color:var(--muted);white-space:nowrap}
.io.ok{background:#12503a;color:#7ee9b4}.io.busy{background:#54410c;color:#ffd97a}
.io.err{background:#5a2126;color:#ffadb3}
button.act{font:inherit;font-size:13.5px;padding:8px 13px;border:1px solid var(--line);
background:var(--panel);color:var(--text);border-radius:9px;cursor:pointer}
button.act:hover{border-color:var(--accent)}
button.act.primary{background:var(--accent);color:#04101f;border-color:var(--accent);font-weight:700}
button.act.danger{border-color:#7a3038;color:var(--bad)}
kbd{font:12px ui-monospace,monospace;background:#2c3850;border:1px solid var(--line);
border-bottom-width:2px;border-radius:4px;padding:1px 5px;color:var(--text)}
.help{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:13px 15px;
margin-top:12px;font-size:13.5px;color:var(--text)}
.help p{margin:0 0 9px}.help .keys{color:var(--muted);line-height:2.1}
textarea{width:100%;font:inherit;padding:9px;border:1px solid var(--line);border-radius:9px;
background:var(--panel2);color:var(--text)}
@media(max-width:1100px){.split{grid-template-columns:1fr}}
</style></head><body>
<div class="app">
<aside><h1>Boundary review</h1><div class="prog" id="prog"></div><div id="list"></div></aside>
<main>
  <div class="top">
    <div><h2 id="title"></h2><div class="sub" id="sub"></div></div>
    <div class="acts" style="margin:0">
      <button class="act" id="prev">&larr; prev</button>
      <button class="act" id="next">next &rarr;</button>
      <span id="io" class="io">browser storage only</span>
      <button class="act" id="save">Download JSON</button>
    </div>
  </div>
  <div class="split">
    <div>
      <video id="v" playsinline></video>
      <div class="clock" id="clock"></div>
      <div class="acts">
        <button class="act" id="loop">Loop <kbd>L</kbd></button>
        <button class="act" id="reset">Reset clip to machine</button>
        <button class="act danger" id="wipe">Clear all my edits</button>
      </div>
      <textarea id="note" rows="2" placeholder="note for this clip (optional)"></textarea>
      <div class="help">
        <p><b>Convention</b> (Luna pilot): exclude preparatory movement and release; a final hold
        may fall outside the interval. A human reviewer differed from ASLLRP by 68&nbsp;ms median
        at start, 145&nbsp;ms at end &mdash; exact agreement is not the bar.</p>
        <div class="keys">
          <kbd>Space</kbd> play/pause &middot; <kbd>&larr;</kbd><kbd>&rarr;</kbd> step frame
          (<kbd>Shift</kbd> &times;5) &middot; <kbd>1</kbd>&ndash;<kbd>9</kbd> select sign<br>
          <kbd>[</kbd> start at playhead &middot; <kbd>]</kbd> end <b>and jump to the next unmarked sign</b>
          &middot; <kbd>T</kbd> snap to teacher segment<br>
          <kbd>,</kbd><kbd>.</kbd> nudge start &middot; <kbd>;</kbd><kbd>'</kbd> nudge end &middot;
          <kbd>A</kbd>/<kbd>R</kbd>/<kbd>U</kbd> accept/reject/unsure<br>
          <kbd>L</kbd> loop &middot; <kbd>N</kbd>/<kbd>P</kbd> next/prev clip &middot;
          <kbd>Del</kbd> clear selected sign
        </div>
      </div>
    </div>
    <div>
      <div id="right"><div class="zoom"><span>zoom</span><input type="range" id="zoomr" min="1" max="8" step="0.5" value="1">
        <span id="zoomv"></span></div>
      <div class="tlwrap"><div class="ruler" id="ruler"></div></div>
      <div id="rows"></div>
    </div></div>
  </div>
</main></div>
<script>
const DATA = __DATA__;
const KEY = 'local_phrase_boundary_review_v17';
let state = JSON.parse(localStorage.getItem(KEY) || '{}');
let ci = 0, si = 0, looping = false, zoom = 1;
const v = document.getElementById('v');
const $ = id => document.getElementById(id);
const clip = () => DATA[ci];
const MINF = 2;                       // hard floor only, not a working default
const DEFAULT_W = 0.40;               // near the median local sign; never the 3-frame floor
const SHORT = 0.25;                   // anything under this is flagged, not silently accepted

function rec(i){
  const c = DATA[i];
  if(!state[c.item]) state[c.item] = {
    intervals: c.intervals.map(x => x ? x.slice() : null),
    verdicts: c.glosses.map(() => null), note: '', touched: false };
  const r = state[c.item];
  // repair anything an earlier build collapsed to zero length
  r.intervals = r.intervals.map(iv => {
    if(!iv) return null;
    let [a, b] = iv.map(Number);
    if(!isFinite(a) || !isFinite(b)) return null;
    a = Math.max(0, Math.min(c.duration, a));
    b = Math.max(0, Math.min(c.duration, b));
    if(b - a < MINF / c.fps){ b = Math.min(c.duration, a + MINF / c.fps);
      a = Math.max(0, b - MINF / c.fps); }
    return [a, b]; });
  return r;
}
const fr = t => Math.round(t * clip().fps);
const snap = t => Math.max(0, Math.min(clip().duration, Math.round(t * clip().fps) / clip().fps));
function save(){ localStorage.setItem(KEY, JSON.stringify(state)); drawList(); autosave(); }

function status(i){
  const c = DATA[i], r = state[c.item];
  if(r && r.touched) return ['done', 'reviewed'];
  const n = c.intervals.filter(Boolean).length;
  if(n === 0) return ['none', '0/' + c.glosses.length];
  return n < c.glosses.length ? ['part', n + '/' + c.glosses.length] : ['full', n + '/' + c.glosses.length];
}
function drawList(){
  const done = DATA.filter(c => state[c.item] && state[c.item].touched).length;
  let iv = 0, tot = 0;
  DATA.forEach(c => { const r = state[c.item]; tot += c.glosses.length;
    iv += (r ? r.intervals : c.intervals).filter(Boolean).length; });
  $('prog').textContent = done + ' of ' + DATA.length + ' clips reviewed \u00b7 ' + iv + ' of ' + tot + ' intervals set';
  $('list').innerHTML = DATA.map((c, i) => { const [cls, txt] = status(i);
    return '<div class="clip' + (i === ci ? ' on' : '') + '" data-i="' + i + '">' +
      '<span><b>' + (i + 1) + '</b> ' + c.glosses.join(' ') +
      ' <span class="tag">' + c.item.split(':').pop().slice(0, 8) + '</span></span>' +
      '<span class="st ' + cls + '">' + txt + '</span></div>'; }).join('');
}
function drawRuler(){
  const c = clip(), n = Math.max(5, Math.round(7 * zoom));
  let h = '';
  for(let i = 0; i <= n; i++){
    const x = 100 * i / n;
    h += '<i style="left:' + x + '%"></i><span style="left:' + x + '%">' +
         (c.duration * i / n).toFixed(2) + 's</span>'; }
  $('ruler').innerHTML = h;
  $('zoomv').textContent = '\u00d7' + zoom;
}
function drawRows(){
  const c = clip(), r = rec(ci);
  $('rows').innerHTML = c.glosses.map((g, k) => {
    const iv = r.intervals[k], sc = c.scores[k];
    const machine = c.intervals[k];
    const edited = iv && machine && (Math.abs(iv[0] - machine[0]) > 1e-6 || Math.abs(iv[1] - machine[1]) > 1e-6);
    const cand = c.candidates.map(s => '<div class="cand" style="left:' + (100 * s[0] / c.duration) +
      '%;width:' + (100 * (s[1] - s[0]) / c.duration) + '%"></div>').join('');
    const short = iv && (iv[1] - iv[0]) < SHORT;
    const bar = iv ? '<div class="bar' + (short ? ' short' : machine && !edited ? ' machine' : '') + '" data-k="' + k +
      '" style="left:' + (100 * iv[0] / c.duration) + '%;width:' + (100 * (iv[1] - iv[0]) / c.duration) +
      '%"><div class="h l" data-e="0"></div><b>' + g + '</b><div class="h r" data-e="1"></div></div>' : '';
    const meta = iv ? (short ? '<span class="warn">' + (iv[1] - iv[0]).toFixed(3) + 's \u2014 very short</span>'
        : (iv[1] - iv[0]).toFixed(3) + 's') + ' \u00b7 frames ' + fr(iv[0]) + '\u2013' + fr(iv[1])
      + (sc ? ' \u00b7 verifier ' + sc.verifier : '') + (edited ? ' \u00b7 edited' : machine ? ' \u00b7 machine' : ' \u00b7 added')
      : 'no interval \u2014 press T to snap to a teacher segment, or [ and ]';
    return '<div class="sign' + (k === si ? ' sel' : '') + '" data-k="' + k + '">' +
      '<header><h3>' + (k + 1) + '. ' + g + '<em>' + meta + '</em></h3>' +
      '<div class="v">' + ['accept','reject','unsure'].map(x =>
        '<button data-v="' + x + '" data-k="' + k + '" class="' + (r.verdicts[k] === x ? 'on' : '') +
        '">' + x + '</button>').join('') + '</div></header>' +
      '<div class="tlwrap"><div class="tl" data-k="' + k + '">' +
      cand + bar + '<div class="ph"></div></div></div>' +
      '<div class="nums"><label>start <input type="number" step="0.001" name="s' + k +
      '" data-k="' + k + '" data-e="0" value="' + (iv ? iv[0].toFixed(3) : '') + '"></label>' +
      '<label>end <input type="number" step="0.001" name="e' + k + '" data-k="' + k +
      '" data-e="1" value="' + (iv ? iv[1].toFixed(3) : '') + '"></label></div></div>'; }).join('');
}
function load(i){
  ci = ((i % DATA.length) + DATA.length) % DATA.length; si = 0;
  const c = clip(); rec(ci);
  $('title').textContent = c.glosses.join(' \u00b7 ');
  $('sub').textContent = c.item + ' \u00b7 ' + c.signer + ' \u00b7 ' + c.fps.toFixed(2) + ' fps \u00b7 ' +
    c.duration.toFixed(3) + 's \u00b7 ' + Math.round(c.duration * c.fps) + ' frames \u00b7 ' +
    c.candidates.length + ' teacher segments';
  v.preload = 'auto'; v.src = c.video;
  v.onerror = () => { $('sub').innerHTML += ' <b style="color:var(--bad)">video failed to load</b>'; };
  $('note').value = state[c.item].note || '';
  zoom = Math.min(8, Math.max(1, Math.round(2.6 / Math.max(c.duration, .3))));
  $('zoomr').value = zoom;
  $('right').style.setProperty('--zoom', zoom);
  drawRows(); drawRuler(); drawList();
}
function tick(){
  const c = clip(), x = 100 * v.currentTime / c.duration;
  document.querySelectorAll('.tl .ph').forEach(p => p.style.left = x + '%');
  $('clock').innerHTML = '<b>' + v.currentTime.toFixed(3) + 's</b> \u00b7 frame ' + fr(v.currentTime) +
    ' / ' + Math.round(c.duration * c.fps);
  requestAnimationFrame(tick);
}
function setEdge(k, e, t, quiet){
  const c = clip(), r = rec(ci), lim = MINF / c.fps;
  let iv = r.intervals[k];
  if(!iv){ const a = snap(t); iv = r.intervals[k] = [a, snap(Math.min(c.duration, a + DEFAULT_W))]; }
  const val = snap(t);
  // Marking an edge past the opposite one means a NEW span is being marked, not a reversal.
  // The old code clamped to a 3-frame floor here, which silently produced 0.1s "signs".
  if(e === 0){
    if(val >= iv[1] - lim) r.intervals[k] = iv = [val, snap(Math.min(c.duration, val + DEFAULT_W))];
    else iv[0] = val;
  } else {
    if(val <= iv[0] + lim) r.intervals[k] = iv = [snap(Math.max(0, val - DEFAULT_W)), val];
    else iv[1] = val;
  }
  r.touched = true; save();
  if(!quiet) drawRows();
}
function nudge(k, e, d){ const iv = rec(ci).intervals[k]; if(iv) setEdge(k, e, iv[e] + d / clip().fps); }
function advance(){
  // after closing a sign, land on the next one that still has no interval
  const r = rec(ci), n = clip().glosses.length;
  for(let i = 1; i <= n; i++){
    const k = (si + i) % n;
    if(!r.intervals[k]){ si = k; drawRows(); return; }
  }
  drawRows();
}
function verdict(k, val){
  const r = rec(ci); r.verdicts[k] = r.verdicts[k] === val ? null : val; r.touched = true;
  save(); drawRows();
}
function snapCand(){
  const c = clip(), r = rec(ci); if(!c.candidates.length) return;
  const at = r.intervals[si] ? (r.intervals[si][0] + r.intervals[si][1]) / 2 : v.currentTime;
  const best = c.candidates.reduce((a, b) =>
    Math.abs((b[0] + b[1]) / 2 - at) < Math.abs((a[0] + a[1]) / 2 - at) ? b : a);
  r.intervals[si] = best.slice(); r.touched = true; save(); drawRows();
}
function selectSign(k){
  if(si === k) return;
  si = k;
  document.querySelectorAll('.sign').forEach(el => el.classList.toggle('sel', +el.dataset.k === k));
}
$('list').onclick = e => { const d = e.target.closest('.clip'); if(d) load(+d.dataset.i); };
$('rows').addEventListener('click', e => {
  const b = e.target.closest('button[data-v]');
  if(b){ verdict(+b.dataset.k, b.dataset.v); return; }
  const sign = e.target.closest('.sign'); if(sign) selectSign(+sign.dataset.k);
});
$('rows').addEventListener('change', e => {
  const inp = e.target.closest('input[type=number]'); if(!inp) return;
  if(inp.value.trim() === '' || !isFinite(Number(inp.value))){ drawRows(); return; }
  setEdge(+inp.dataset.k, +inp.dataset.e, Number(inp.value));
});
$('rows').addEventListener('pointerdown', e => {
  const card = e.target.closest('.sign');
  if(card) selectSign(+card.dataset.k);
  const tl = e.target.closest('.tl'); if(!tl) return;
  const k = +tl.dataset.k;
  const handle = e.target.closest('.h');
  const bar = e.target.closest('.bar');
  const box = tl.getBoundingClientRect();
  const at = ev => clip().duration * (ev.clientX - box.left) / box.width;
  if(handle){
    const edge = +handle.dataset.e;
    const move = ev => setEdge(k, edge, at(ev));
    const up = () => { removeEventListener('pointermove', move); removeEventListener('pointerup', up); };
    addEventListener('pointermove', move); addEventListener('pointerup', up);
    e.preventDefault(); return; }
  if(bar){
    const iv = rec(ci).intervals[k], w = iv[1] - iv[0], grab = at(e) - iv[0];
    const move = ev => { const c = clip(), r = rec(ci);
      let a = Math.max(0, Math.min(c.duration - w, at(ev) - grab));
      r.intervals[k] = [snap(a), snap(a + w)]; r.touched = true; save(); drawRows(); };
    const up = () => { removeEventListener('pointermove', move); removeEventListener('pointerup', up); };
    addEventListener('pointermove', move); addEventListener('pointerup', up);
    e.preventDefault(); return; }
  scrub(e, box);                       // empty track or the playhead itself: drag to scrub
});
function scrub(e, box){
  const seek = ev => { v.currentTime = Math.max(0, Math.min(clip().duration, at2(ev, box))); };
  const at2 = (ev, b) => clip().duration * (ev.clientX - b.left) / b.width;
  seek(e);
  const up = () => { removeEventListener('pointermove', seek); removeEventListener('pointerup', up); };
  addEventListener('pointermove', seek); addEventListener('pointerup', up);
  e.preventDefault();
}
$('ruler').addEventListener('pointerdown', e => {
  scrub(e, $('ruler').getBoundingClientRect());
});
$('note').oninput = () => { const r = rec(ci); r.note = $('note').value; r.touched = true; save(); };
$('prev').onclick = () => load(ci - 1);
$('next').onclick = () => load(ci + 1);
$('loop').onclick = () => { looping = !looping; $('loop').classList.toggle('primary', looping); };
$('reset').onclick = () => { const c = clip();
  state[c.item] = { intervals: c.intervals.map(x => x ? x.slice() : null),
    verdicts: c.glosses.map(() => null), note: '', touched: false };
  save(); drawRows(); };
$('wipe').onclick = () => { if(confirm('Discard every edit you have made and reload the machine alignments?')){
  state = {}; localStorage.removeItem(KEY); load(ci); } };
$('zoomr').oninput = e => { zoom = Number(e.target.value);
  $('right').style.setProperty('--zoom', zoom); drawRuler(); };
v.addEventListener('timeupdate', () => {
  if(!looping) return; const iv = rec(ci).intervals[si]; if(!iv) return;
  if(v.currentTime >= iv[1] || v.currentTime < iv[0] - .05) v.currentTime = iv[0];
});
function payload(){
  return { format: 'local_phrase_boundary_review_v17',
    convention: 'lexical sign interval; excludes preparatory movement and release',
    reviewed_at: new Date().toISOString(),
    records: DATA.map(c => { const r = state[c.item] || {};
      return { item: c.item, video_sha256: c.sha, signer: c.signer, fps: c.fps,
        target_sequence: c.glosses, machine_intervals: c.intervals,
        intervals: r.intervals || c.intervals,
        verdicts: r.verdicts || c.glosses.map(() => null),
        reviewed: !!r.touched, note: r.note || '' }; }) };
}
$('save').onclick = () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(payload(), null, 1)], {type: 'application/json'}));
  a.download = 'boundary_review_corrected.json'; a.click();
};

// ---- autosave -------------------------------------------------------------
// Served over http the page writes straight to disk after every edit. Opened as a
// file:// URL there is no server to write to, so it falls back to browser storage and
// says so rather than pretending the work is saved.
const served = location.protocol.startsWith('http');
let timer = null, inflight = false, again = false;
function io(cls, text){ const el = $('io'); el.className = 'io ' + cls; el.textContent = text; }
async function push(){
  if(!served) return;
  if(inflight){ again = true; return; }
  inflight = true; io('busy', 'saving\u2026');
  try {
    const res = await fetch('/__save', { method: 'POST',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload()) });
    if(!res.ok) throw new Error('HTTP ' + res.status);
    const info = await res.json();
    io('ok', 'saved \u00b7 ' + info.reviewed + ' clips \u00b7 ' + info.intervals + ' intervals');
  } catch (err) {
    io('err', 'NOT saved to disk: ' + err.message);
  } finally {
    inflight = false;
    if(again){ again = false; push(); }
  }
}
function autosave(){ if(!served) return; clearTimeout(timer); timer = setTimeout(push, 400); }
addEventListener('beforeunload', () => {
  if(served) navigator.sendBeacon('/__save', new Blob([JSON.stringify(payload())],
    {type: 'application/json'}));
});
addEventListener('keydown', e => {
  const t = e.target;
  if(t && t.nodeType === 1 && t.matches('textarea,input,select')) return;
  const c = clip(), f = 1 / c.fps;
  if(/^[1-9]$/.test(e.key)){ if(+e.key <= c.glosses.length){ si = +e.key - 1; drawRows(); } e.preventDefault(); return; }
  const map = {
    ' ': () => v.paused ? v.play() : v.pause(),
    'ArrowRight': () => v.currentTime = snap(v.currentTime + (e.shiftKey ? 5 : 1) * f),
    'ArrowLeft': () => v.currentTime = snap(v.currentTime - (e.shiftKey ? 5 : 1) * f),
    '[': () => setEdge(si, 0, v.currentTime),
    ']': () => { setEdge(si, 1, v.currentTime, true); advance(); },
    ',': () => nudge(si, 0, -1), '.': () => nudge(si, 0, 1),
    ';': () => nudge(si, 1, -1), "'": () => nudge(si, 1, 1),
    'a': () => verdict(si, 'accept'), 'r': () => verdict(si, 'reject'), 'u': () => verdict(si, 'unsure'),
    'l': () => $('loop').click(), 't': snapCand,
    'n': () => load(ci + 1), 'p': () => load(ci - 1),
    'Delete': () => { rec(ci).intervals[si] = null; rec(ci).touched = true; save(); drawRows(); },
    'Backspace': () => { rec(ci).intervals[si] = null; rec(ci).touched = true; save(); drawRows(); }};
  const fn = map[e.key]; if(fn){ fn(); e.preventDefault(); }
});
(async () => {
  if(served){
    try {
      const saved = await (await fetch('/__load')).json();
      if(saved && saved.records){
        for(const r of saved.records)
          if(r.reviewed) state[r.item] = { intervals: r.intervals, verdicts: r.verdicts,
                                           note: r.note || '', touched: true };
        localStorage.setItem(KEY, JSON.stringify(state));
      }
      io('ok', 'saving to boundary_review_corrected.json');
    } catch (err) { io('err', 'server unreachable \u2014 browser storage only'); }
  } else {
    io('', 'browser storage only \u2014 run serve_alignment_reviewer_v17.py to save to disk');
  }
  load(0); tick();
})();
</script></body></html>
"""


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    rows = clip_rows()
    out = REPORT / 'review.html'
    out.write_text(PAGE.replace('__DATA__', json.dumps(rows, separators=(',', ':'))))
    machine = sum(1 for r in rows for i in r['intervals'] if i)
    tokens = sum(len(r['glosses']) for r in rows)
    print('clips %d | sign tokens %d | machine intervals %d | missing %d'
          % (len(rows), tokens, machine, tokens - machine))
    print('wrote', out, '(%.0f KB)' % (out.stat().st_size / 1024))


if __name__ == '__main__':
    main()
