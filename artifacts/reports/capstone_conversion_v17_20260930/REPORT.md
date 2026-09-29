# Matched recognition conversion check — 2026-09-30

Author authorized validation-only conversion evaluation. No training or test-set access.

## Recognizer

Script: scripts/check_capstone_conversion_v17.py. Uses original FP32 PyTorch checkpoint
and application fixed-batch FP16 Core ML package; exact identities in recognizer_summary.json
and package_hashes.json. Compute units ALL on development Mac, not a device accuracy test.
All378validation examples included; final two examples padded to batch8 and trimmed after
prediction. Labels from checkpoint label_to_index; identical cached landmark/hand features.
Top1:360/378=95.238095% before,359/378=94.973545% after. Top5:374/378=98.941799% both.
Top1agreement377/378=99.735450%. One LIKE example changed from correct LIKE to MY.
No checkpoint/threshold/input adjustment made. This measures recognizer conversion,
not MobileCLIP encoder precision changes, sequence WER or end-to-end phone accuracy.

## Boundary

Reused saved boundary_export.json:0/1944top1 timing-state mismatches on40tuning clips,
100%agreement; maximum probability difference0.0010393262. Original checkpoint SHA256
verified against recorded export hash. This is saved conversion agreement, not a new
boundary run or human-annotation accuracy. Citation/provenance in boundary_evidence.json.

## Audit note

Saved fixed-batch recognizer exporter reports378parity samples but its loop stops at the
last complete batch (376). New check covers the final partial batch too. It confirms one
changed prediction for this deployed export, unlike the different batch1 export's0mismatch.
Do not transfer the batch1export's perfect agreement to the application package.

## Presentation

Replaced phone-profile table/chart with matched accuracy/agreement table. Kept28ms physical
phone preparation/recognition median as one sentence. No total speech latency inferred.

## Matched model timings — 2026-09-30

Script:scripts/benchmark_capstone_conversion_v17.py; raw samples/provenance:latency.json.
Hardware:Apple M4; PyTorch CPU one thread, CoreML ALL. Full recognition graphs
matched, including all exported computation.16batches of8validation inputs,80timed calls
per backend. Boundary32windows from4tuning clips,160timed calls/backend.10warmups and
5passes, alternating backend order. Prepared arrays/tensors; loading and preprocessing
excluded. No test access. Recognizer median:27.64331250→11.32177050ms/batch8;
boundary:1.39489600→2.00045800ms/window. Boundary CoreML slower in this check.
Runtime+hardware allocation comparison, not isolated precision speedup and not iPhone timing.

## Storage sizes

Reproducible script:measure_capstone_conversion_sizes_v17.py (repository scripts/).
Saved FP32 full inference state, excluding training metadata/optimizer, compared with
full uncompiled CoreML source package. Decimal MB. Recognizer49.616359→25.506008MB;
boundary9.254398→4.682127MB. Package bytes include graph/manifest/weights. Not resident
RAM, compiled bundle size or application size. Exact bytes and paths in sizes.json.
