from pathlib import Path
import json, hashlib, statistics
p = Path(__file__).resolve().parent
families = {f: json.loads((p / (f + '.json')).read_text()) for f in ['apple', 'mediapipe']}
ids = [[r['path'] for r in d['rows']] for d in families.values()]
assert ids[0] == ids[1] and len(ids[0]) == 378
assert not any(r['failed'] for d in families.values() for r in d['rows'])
summary = {f: {stage: statistics.median(r['total_ms'][stage] for r in d['rows']) for stage in ['base','fusion','span']} for f,d in families.items()}
hashes = {name: hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ['benchmark.py','apple.json','mediapipe.json']}
(p/'summary.json').write_text(json.dumps({'count':len(ids[0]),'failures':0,'median_staged_ms':summary,'sha256':hashes},indent=2)+'\n')
lines = ['# Matched Mac pipeline processing — 2026-10-07', '',
'## Outcome', '',
'Both pipelines completed all 378 identical canonical validation recordings with zero extraction failures. No protected test, training, model selection, dataset changes or acquisition. These timings are development-computer measurements, not mobile operating-system rankings.', '',
'| Stage | Apple Vision (ms/clip) | MediaPipe (ms/clip) |', '| --- | ---: | ---: |']
for k,label in [('base','Landmark-only'),('fusion','Combined inputs'),('span','Interval-adapted')]:
 lines.append(f"| {label} | {summary['apple'][k]:.3f} | {summary['mediapipe'][k]:.3f} |")
lines += ['', '## Measurement contract', '',
'Apple M4 development Mac. Each family ran separately to avoid competing benchmark processes. Both used the same ordered input videos and recorded orientation decisions, production sampling contracts, a shared FP32 MobileCLIP2 image encoder, and FP32 PyTorch recognition on MPS. Models were warmed before measurement. Recognition cost is the median of two synchronized calls per graph per clip.', '',
'Per-clip costs sum measured stages: landmark extraction and feature preparation, tensor transfer, applicable hand-crop preparation/hand-image encoding, and recognition. The reported aggregate is the median of these per-clip stage sums, not the sum of population medians or a live streaming response time. Initial video decoding/resizing, model loading, detector renewal, camera capture, English generation and speech/UI work are excluded. Hand-crop preparation includes its selected-frame decoding. Landmark-only totals include the same small shared tensor-transfer measurement; the separate hand-image extraction cost is excluded from that stage.', '',
'The three recognition accuracy values are sourced from mediapipe_rebuild_v17_20261004/REPORT.md, not recomputed by this timing script. Same validation membership; no additional MediaPipe-only evaluation dataset. The prior rebuild reports matched training lists; this benchmark does not modify them. Timed features are freshly extracted and encoded; recognition accuracy evidence uses the saved evaluation contract. Accuracy and timing are separate measurements of the named stages.', '',
'## Failed first attempt and recovery', '',
'Initial MediaPipe execution exceeded MPS memory capacity because of the documented macOS GPU pixel-buffer leak. Its partial JSON and failure log are preserved and excluded. The complete retry renews the detector between clips once its call counter reaches 1000, outside timed work. This follows the existing detector renewal mechanism and per-sequence tracking contract. No allocator limit was disabled. Apple completed without this failure. These are steady-state staged costs, not sustained deployment/resource readiness evidence.', '',
'## Provenance', '',
'benchmark.py contains exact checkpoint and source paths. apple.json and mediapipe.json retain every per-clip component timing. summary.json records hashes and checks. chart_values.json drives the manuscript figure. Device comparison remains in ../phone_precision_v17_20261007/REPORT.md.']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary))
