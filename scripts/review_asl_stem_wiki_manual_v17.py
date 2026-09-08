#!/usr/bin/env python3
"""Run the local ASL STEM Wiki admission-review UI."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
from pathlib import Path
import re
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


DECISIONS = {"", "yes", "no", "unsure"}
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


def video_probe(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frames < 1 or fps <= 0:
        raise ValueError(f"invalid video metadata: {path}")
    return {"frames": frames, "fps": fps, "duration": frames / fps}


def as_bool(value):
    return value is True or str(value).lower() == "true"


class ReviewStore:
    def __init__(
        self, queue: Path, citizen_train_root: Path, probe=video_probe,
        auto_annotations=None,
    ):
        self.queue = Path(queue)
        self.train_root = Path(citizen_train_root).resolve()
        if self.train_root.name != "train":
            raise ValueError("Citizen references must use the raw/train directory")
        with self.queue.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.fields = reader.fieldnames or []
            self.rows = list(reader)
        if not self.rows:
            raise ValueError("review queue is empty")
        self.probe = probe
        self._metadata = {}
        self._lock = threading.Lock()
        self.automatic = {}
        automatic_path = Path(auto_annotations) if auto_annotations else None
        if automatic_path and automatic_path.exists():
            payload = json.loads(automatic_path.read_text())
            if not payload.get("complete"):
                raise ValueError("automatic annotation run is incomplete")
            for annotation in payload["annotations"]:
                index = int(annotation["queue_index"])
                if not 0 <= index < len(self.rows):
                    raise ValueError("automatic annotation has an unknown queue row")
                row = self.rows[index]
                identity = ("participant", "filename", "raw_gloss", "citizen_asl_lex_code")
                if any(annotation[key] != row[key] for key in identity):
                    raise ValueError(f"automatic annotation identity mismatch at row {index}")
                self.automatic[index] = annotation

    def _source_path(self, index):
        path = Path(self.rows[index]["video_path"])
        return (path if path.is_absolute() else Path.cwd() / path).resolve()

    def _references(self, index):
        folder = (self.train_root / self.rows[index]["raw_gloss"]).resolve()
        if folder.parent != self.train_root or not folder.is_dir():
            return []
        return sorted(
            path.resolve() for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".m4v"}
        )

    def _probe(self, path):
        if path not in self._metadata:
            self._metadata[path] = self.probe(path)
        return self._metadata[path]

    def public_rows(self):
        output = []
        for index, stored in enumerate(self.rows):
            row = dict(stored)
            source = self._source_path(index)
            row.update({
                "index": index,
                "expected_gloss": stored["raw_gloss"],
                "source_video": {
                    "url": f"/media/source/{index}",
                    "path": str(source),
                    **self._probe(source),
                },
                "reference_videos": [
                    {
                        "url": f"/media/reference/{index}/{reference_index}",
                        "path": str(path),
                        "name": path.name,
                    }
                    for reference_index, path in enumerate(self._references(index))
                ],
                "signer_quality_verified": as_bool(stored["signer_quality_verified"]),
                "variant_verified": as_bool(stored["variant_verified"]),
                "boundary_verified": as_bool(stored["boundary_verified"]),
                "training_eligible": as_bool(stored["training_eligible"]),
                "automatic_annotation": self.automatic.get(index),
            })
            output.append(row)
        return output

    def media_path(self, kind, index, reference_index=None):
        if not 0 <= index < len(self.rows):
            raise ValueError("unknown review row")
        if kind == "source":
            return self._source_path(index)
        if kind == "reference" and reference_index is not None:
            references = self._references(index)
            if 0 <= reference_index < len(references):
                return references[reference_index]
        raise ValueError("unknown media item")

    def apply(self, index, update):
        if not 0 <= index < len(self.rows):
            raise ValueError("unknown review row")
        signer = str(update.get("signer_quality_decision", "")).lower()
        variant = str(update.get("variant_decision", "")).lower()
        if signer not in DECISIONS or variant not in DECISIONS:
            raise ValueError("decisions must be yes, no, unsure, or blank")
        start_raw = update.get("verified_start_frame", "")
        end_raw = update.get("verified_end_frame", "")
        if (start_raw == "") != (end_raw == ""):
            raise ValueError("set both start and end frames")
        boundary_verified = False
        start = end = ""
        if start_raw != "":
            try:
                start, end = int(start_raw), int(end_raw)
            except (TypeError, ValueError):
                raise ValueError("start and end must be whole frame numbers")
            frames = self._probe(self._source_path(index))["frames"]
            if start < 0 or end <= start or end >= frames:
                raise ValueError(f"boundaries must satisfy 0 <= start < end < {frames}")
            if end - start + 1 > 256:
                raise ValueError("verified span cannot exceed 256 frames")
            boundary_verified = True

        row = self.rows[index]
        source_excluded = row["signer_status"] == "l2_excluded"
        signer_verified = signer == "yes" and not source_excluded
        variant_verified = variant == "yes"
        eligible = signer_verified and variant_verified and boundary_verified
        row.update({
            "signer_quality_decision": signer,
            "variant_decision": variant,
            "verified_start_frame": str(start),
            "verified_end_frame": str(end),
            "reviewer_notes": str(update.get("reviewer_notes", "")).strip(),
            "signer_quality_verified": str(signer_verified),
            "variant_verified": str(variant_verified),
            "boundary_verified": str(boundary_verified),
            "training_eligible": str(eligible),
        })
        self._save()
        return self.public_rows()[index]

    def _save(self):
        with self._lock:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", newline="", dir=self.queue.parent,
                prefix=self.queue.name + ".", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=self.fields)
                writer.writeheader()
                writer.writerows(self.rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.queue)


INDEX_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASL admission review</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--card:#141b30;--line:#33405e;--text:#eef3ff;--muted:#9ca9c4;--blue:#68b7ff;--green:#58d68d;--red:#ff7b86;--amber:#f7c65f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,-apple-system,sans-serif}button,input,select,textarea{font:inherit}
header{position:sticky;top:0;z-index:5;background:#0b1020ee;border-bottom:1px solid var(--line);padding:12px 18px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
main{max-width:1500px;margin:auto;padding:18px}.grow{flex:1}.muted{color:var(--muted)}.pill{border:1px solid var(--line);border-radius:999px;padding:5px 10px}
button,select,input,textarea{background:#10172a;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px 10px}button{cursor:pointer}button:hover{border-color:var(--blue)}button:disabled{opacity:.45;cursor:not-allowed}
.primary{background:#1768a9;border-color:#399ce7}.good{color:var(--green)}.bad{color:var(--red)}.warn{color:var(--amber)}
.hero{display:grid;grid-template-columns:1fr auto;gap:16px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}.gloss{font-size:clamp(30px,5vw,64px);font-weight:800;line-height:1}.code{text-align:right}
.videos{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}.panel h2{margin:0 0 10px;font-size:18px}video{display:block;width:100%;aspect-ratio:4/3;background:#000;border-radius:8px;object-fit:contain}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}.controls input[type=range]{flex:1;min-width:180px}.wide{width:100%}
.review{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}.decision{display:flex;gap:8px}.decision label{flex:1}.decision input{position:absolute;opacity:0}.decision span{display:block;text-align:center;border:1px solid var(--line);border-radius:8px;padding:10px;cursor:pointer}.decision input:checked+span{border-color:var(--blue);background:#183a5b}
.field{margin:12px 0}.field>label,.field>strong{display:block;margin-bottom:7px}.bounds{display:grid;grid-template-columns:1fr 1fr;gap:12px}.bounds input[type=number]{width:100%}textarea{width:100%;min-height:92px;resize:vertical}
.footer{display:flex;gap:10px;justify-content:flex-end;margin-top:16px}.notice{padding:10px;border-radius:8px;background:#241f13;border:1px solid #675523}.status{min-height:22px;margin-top:8px}
@media(max-width:850px){.videos,.review{grid-template-columns:1fr}.code{text-align:left}.hero{grid-template-columns:1fr}}
</style></head><body>
<header><strong>ASL admission review</strong><span id="progress" class="pill"></span><select id="filter" aria-label="Filter"><option value="all">All rows</option><option value="auto-high">Automatic: high evidence</option><option value="auto-review">Automatic: needs review</option><option value="auto-abstain">Automatic: abstained</option><option value="pending">All pending</option><option value="eligible">Eligible</option><option value="excluded">Rejected / excluded</option></select><span class="grow"></span><button id="prev">← Previous</button><span id="position"></span><button id="next">Next →</button></header>
<main>
<section class="hero"><div><div class="muted">Expected gloss</div><div id="gloss" class="gloss"></div><div id="canonical" class="muted"></div></div><div class="code"><div class="muted">Locked Citizen ASL-LEX code</div><strong id="asllex"></strong><div id="signer"></div></div></section>
<section class="videos">
 <div class="panel"><h2>Source sentence</h2><video id="source" controls preload="metadata"></video><div class="controls"><button data-step="-10">−10</button><button data-step="-1">−1 frame</button><input id="timeline" type="range" min="0" value="0"><button data-step="1">+1 frame</button><button data-step="10">+10</button><strong id="frameNow"></strong></div><div class="controls"><button id="jumpProposal">Jump to proposal</button><label>Playback speed <select id="speed"><option>.25</option><option>.5</option><option selected>1</option><option>1.5</option><option>2</option></select></label><label><input id="loop" type="checkbox"> Loop selection</label></div><div id="proposal" class="muted status"></div></div>
 <div class="panel"><h2>Citizen training reference</h2><video id="reference" controls loop preload="metadata"></video><div class="controls"><button id="prevRef">← Reference</button><span id="refPosition" class="grow"></span><button id="nextRef">Reference →</button></div><div class="muted status">Exact locked raw-gloss class from Citizen <strong>training split only</strong>.</div></div>
</section>
<section class="review">
 <div class="panel"><h2>Frame boundaries</h2><div id="autoPanel" class="notice"><strong>Automatic model proposal</strong><div id="autoDetails" class="status"></div><div class="controls"><button id="useAuto">Use automatic bounds</button><button id="playAuto">Play automatic span</button></div></div><div class="controls"><button id="setStart">Set start</button><button id="setEnd">Set end</button><button id="clearBounds">Clear</button><span id="spanLength" class="grow"></span></div><div class="field"><label for="startRange">Start frame</label><input id="startRange" class="wide" type="range" min="0" value="0"><input id="startFrame" type="number" min="0"></div><div class="field"><label for="endRange">End frame</label><input id="endRange" class="wide" type="range" min="0" value="0"><input id="endFrame" type="number" min="0"></div><p class="muted">Use inclusive bounds. Approved spans must contain 256 frames or fewer.</p></div>
 <div class="panel"><h2>Review decisions</h2><div id="l2Notice" class="notice" hidden>This participant is source-excluded as possible L2 and cannot become training eligible.</div><div class="field"><strong>Signer quality</strong><div id="signerDecision" class="decision"></div></div><div class="field"><strong>Matches the locked Citizen reference variant</strong><div id="variantDecision" class="decision"></div></div><div class="field"><label for="notes">Reviewer notes</label><textarea id="notes"></textarea></div><div id="eligibility" class="status"></div></div>
</section>
<div class="footer"><span id="saveStatus" class="grow status"></span><button id="save">Save</button><button id="saveNext" class="primary">Save &amp; next</button></div>
</main>
<script>
let rows=[],current=0,refIndex=0,visible=[];const $=id=>document.getElementById(id);
function decisions(id,name){$(id).innerHTML=['yes','no','unsure'].map(v=>`<label><input type="radio" name="${name}" value="${v}"><span>${v[0].toUpperCase()+v.slice(1)}</span></label>`).join('')}
decisions('signerDecision','signer_quality_decision');decisions('variantDecision','variant_decision');
function value(name){return document.querySelector(`input[name="${name}"]:checked`)?.value||''}
function setValue(name,v){document.querySelectorAll(`input[name="${name}"]`).forEach(x=>x.checked=x.value===v)}
function done(r){return r.signer_status==='l2_excluded'||r.signer_quality_decision==='no'||r.variant_decision==='no'||r.training_eligible}
function refreshVisible(){const f=$('filter').value;visible=rows.map((_,i)=>i).filter(i=>f==='all'||(f==='auto-high'&&rows[i].automatic_annotation?.confidence_tier==='high'&&!done(rows[i]))||(f==='auto-review'&&rows[i].automatic_annotation?.confidence_tier==='review'&&!done(rows[i]))||(f==='auto-abstain'&&rows[i].automatic_annotation?.confidence_tier==='abstain'&&!done(rows[i]))||(f==='pending'&&!done(rows[i]))||(f==='eligible'&&rows[i].training_eligible)||(f==='excluded'&&done(rows[i])&&!rows[i].training_eligible));if(!visible.includes(current))current=visible[0]??0}
function frame(){return Math.max(0,Math.round($('source').currentTime*(rows[current].source_video.fps||30)))}
function seek(n){const r=rows[current],f=Math.max(0,Math.min(r.source_video.frames-1,Number(n)||0));$('source').currentTime=f/r.source_video.fps;$('timeline').value=f;$('frameNow').textContent=`Frame ${f} / ${r.source_video.frames-1}`}
function ref(){const refs=rows[current].reference_videos;$('reference').src=refs[refIndex]?.url||'';$('reference').hidden=!refs.length;$('refPosition').textContent=refs.length?`${refIndex+1} of ${refs.length} · ${refs[refIndex].name}`:'No training reference found';$('prevRef').disabled=refIndex<=0;$('nextRef').disabled=refIndex>=refs.length-1}
function span(){const a=$('startFrame').value,b=$('endFrame').value;if(a===''||b===''){$('spanLength').textContent='No verified span';return}$('spanLength').textContent=`${Number(b)-Number(a)+1} frames`}
function render(){refreshVisible();const r=rows[current],m=r.source_video,max=m.frames-1;refIndex=0;$('gloss').textContent=r.expected_gloss;$('canonical').textContent=r.canonical_label===r.expected_gloss?'':`Displayed label: ${r.canonical_label}`;$('asllex').textContent=r.citizen_asl_lex_code;$('signer').textContent=`${r.participant} · ${r.signer_status}`;$('source').src=m.url;$('timeline').max=max;$('startRange').max=max;$('endRange').max=max;$('startFrame').max=max;$('endFrame').max=max;
 const proposal=r.proposed_frame||r.proposed_start_frame;$('proposal').textContent=proposal!==''?`Machine proposal: frame ${proposal}${r.pseudo_confidence?` · confidence ${r.pseudo_confidence}`:''}. Review aid only.`:'No machine position proposal.';$('jumpProposal').disabled=proposal==='';
 const auto=r.automatic_annotation;$('autoPanel').hidden=!auto;if(auto){const top=(auto.full_top3||[]).map(x=>`${x.gloss} ${(100*x.probability).toFixed(0)}%`).join(' · ');$('autoDetails').textContent=`${auto.confidence_tier.toUpperCase()} · ${(100*auto.confidence).toFixed(1)}% uncalibrated evidence · frames ${auto.start_frame}–${auto.end_frame} · Stage 1 ${auto.full_top1_matches?'agrees':'disagrees'} · CTC ${auto.ctc_agrees?'agrees':'disagrees'}${top?` · ${top}`:''}. This proposal cannot confirm the exact Citizen variant.`;$('useAuto').disabled=$('playAuto').disabled=auto.start_frame===undefined||auto.end_frame===undefined}
 $('startFrame').value=r.verified_start_frame;$('endFrame').value=r.verified_end_frame;$('startRange').value=r.verified_start_frame||0;$('endRange').value=r.verified_end_frame||max;$('notes').value=r.reviewer_notes||'';setValue('signer_quality_decision',r.signer_quality_decision);setValue('variant_decision',r.variant_decision);$('l2Notice').hidden=r.signer_status!=='l2_excluded';document.querySelectorAll('input[name="signer_quality_decision"]').forEach(x=>x.disabled=r.signer_status==='l2_excluded');
 $('eligibility').className='status '+(r.training_eligible?'good':'warn');$('eligibility').textContent=r.training_eligible?'Eligible after save':'Not training eligible';$('position').textContent=visible.length?`${visible.indexOf(current)+1} / ${visible.length}`:'0 / 0';const reviewed=rows.filter(done).length;$('progress').textContent=`${reviewed}/${rows.length} reviewed · ${rows.filter(x=>x.training_eligible).length} eligible`;$('prev').disabled=visible.indexOf(current)<=0;$('next').disabled=visible.indexOf(current)>=visible.length-1;span();ref();seek(r.verified_start_frame||auto?.start_frame||r.proposed_frame||r.proposed_start_frame||0)}
async function save(advance=false){$('saveStatus').textContent='Saving…';const before=[...visible],at=before.indexOf(current);const payload={signer_quality_decision:value('signer_quality_decision'),variant_decision:value('variant_decision'),verified_start_frame:$('startFrame').value,verified_end_frame:$('endFrame').value,reviewer_notes:$('notes').value};const response=await fetch(`/api/review/${current}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const body=await response.json();if(!response.ok){$('saveStatus').textContent=body.error;return}rows[current]=body;refreshVisible();if(advance)current=before[at+1]??visible.find(i=>i>current)??visible[0]??current;render();$('saveStatus').textContent='Saved'}
function move(delta){refreshVisible();const at=visible.indexOf(current),next=visible[at+delta];if(next!==undefined){current=next;render()}}
$('filter').onchange=render;$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('save').onclick=()=>save();$('saveNext').onclick=()=>save(true);$('timeline').oninput=e=>seek(e.target.value);$('source').ontimeupdate=()=>{const f=frame();$('timeline').value=f;$('frameNow').textContent=`Frame ${f} / ${rows[current].source_video.frames-1}`;if($('loop').checked&&$('endFrame').value!==''&&f>=Number($('endFrame').value))seek($('startFrame').value)};document.querySelectorAll('[data-step]').forEach(b=>b.onclick=()=>seek(frame()+Number(b.dataset.step)));$('speed').onchange=e=>$('source').playbackRate=Number(e.target.value);$('jumpProposal').onclick=()=>seek(rows[current].proposed_frame||rows[current].proposed_start_frame);$('useAuto').onclick=()=>{const a=rows[current].automatic_annotation;$('startFrame').value=$('startRange').value=a.start_frame;$('endFrame').value=$('endRange').value=a.end_frame;span();seek(a.start_frame)};$('playAuto').onclick=()=>{$('useAuto').click();$('loop').checked=true;$('source').play()};$('setStart').onclick=()=>{$('startFrame').value=frame();$('startRange').value=frame();span()};$('setEnd').onclick=()=>{$('endFrame').value=frame();$('endRange').value=frame();span()};$('clearBounds').onclick=()=>{$('startFrame').value=$('endFrame').value='';span()};[['startRange','startFrame'],['endRange','endFrame']].forEach(([range,num])=>{$(range).oninput=e=>{$(num).value=e.target.value;span()};$(num).oninput=e=>{$(range).value=e.target.value||0;span()}});$('prevRef').onclick=()=>{refIndex--;ref()};$('nextRef').onclick=()=>{refIndex++;ref()};
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowLeft'){e.preventDefault();seek(frame()+(e.shiftKey?-10:-1))}if(e.key==='ArrowRight'){e.preventDefault();seek(frame()+(e.shiftKey?10:1))}if(e.key===' '){e.preventDefault();$('source').paused?$('source').play():$('source').pause()}if(e.key.toLowerCase()==='s')$('setStart').click();if(e.key.toLowerCase()==='e')$('setEnd').click();if(e.key.toLowerCase()==='l')$('loop').click();if(e.key==='Enter'&&(e.ctrlKey||e.metaKey))save(true)});
fetch('/api/rows').then(r=>r.json()).then(data=>{rows=data;render()}).catch(e=>$('saveStatus').textContent=e);
</script></body></html>'''


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            print(f"{self.address_string()} - {format % args}")

        def _json(self, payload, status=200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                body = INDEX_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/rows":
                self._json(store.public_rows())
            elif self.path.startswith("/media/"):
                self._media(False)
            else:
                self.send_error(404)

        def do_HEAD(self):
            if self.path.startswith("/media/"):
                self._media(True)
            else:
                self.send_error(404)

        def do_POST(self):
            match = re.fullmatch(r"/api/review/(\d+)", self.path)
            if not match:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 65536:
                    raise ValueError("review payload is too large")
                update = json.loads(self.rfile.read(length))
                self._json(store.apply(int(match.group(1)), update))
            except (ValueError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, 400)

        def _media(self, head):
            try:
                parts = self.path.strip("/").split("/")
                if len(parts) == 3 and parts[1] == "source":
                    path = store.media_path("source", int(parts[2]))
                elif len(parts) == 4 and parts[1] == "reference":
                    path = store.media_path("reference", int(parts[2]), int(parts[3]))
                else:
                    raise ValueError("unknown media item")
                size = path.stat().st_size
                start, end, status = 0, size - 1, 200
                requested = self.headers.get("Range")
                if requested:
                    match = RANGE_RE.fullmatch(requested)
                    if not match:
                        raise ValueError("invalid byte range")
                    start = int(match.group(1) or 0)
                    end = int(match.group(2) or size - 1)
                    if start > end or end >= size:
                        raise ValueError("byte range outside media")
                    status = 206
                self.send_response(status)
                self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if head:
                    return
                with path.open("rb") as handle:
                    handle.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = handle.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (OSError, ValueError) as error:
                self.send_error(404, str(error))

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=Path(
        "artifacts/reports/asl_stem_wiki_manual_admission_v17/expert_review_queue.csv"
    ))
    parser.add_argument("--citizen-train-root", type=Path, default=Path(
        "data/local/citizen100_v17/raw/train"
    ))
    parser.add_argument("--auto-annotations", type=Path, default=Path(
        "artifacts/reports/asl_stem_wiki_auto_annotation_v17/annotations.json"
    ))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    store = ReviewStore(
        args.queue, args.citizen_train_root,
        auto_annotations=args.auto_annotations,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Review UI: {url}\nQueue: {args.queue}\nPress Ctrl-C to stop.")
    if not args.no_open:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
