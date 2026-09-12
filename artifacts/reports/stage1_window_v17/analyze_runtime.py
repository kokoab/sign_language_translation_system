"""Run from the repository root after runtime_epoch01 replay and epoch evaluation."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from active.v17.train_stage_2_other_ctc_v17 import _edit_operations

REPORT = ROOT/'artifacts/reports/stage1_window_v17'
pool = {r['source_item_id']: r for r in json.loads((REPORT/'evaluation_manifest.json').read_text())['rows']}
cached = json.loads((REPORT/'evaluations/epoch_01.json').read_text())
cached_rows = {r['item_id']: r for r in cached['rows']}
replays = json.loads((REPORT/'runtime_epoch01/results.json').read_text())
assert len(replays) == len(json.loads((REPORT/'runtime_recordings.json').read_text()))
assert all(r['exit_code'] == 0 for r in replays)
rows, delays, inference_ms, all_operations = [], [], [], Counter()
for replay in replays:
    history = json.loads((ROOT/replay['history']).read_text())
    events = [e for e in history['events'] if e['type'] == 'stage1_window_transcript_update']
    updates = [dict(observed=e['observed_seconds'], elapsed=e['elapsed_seconds'], hypothesis=e['hypothesis']) for e in events]
    finish = replay['finish']
    if events:
        elapsed = events[-1]['elapsed_seconds'] + (datetime.fromisoformat(finish['recorded_utc']) - datetime.fromisoformat(events[-1]['recorded_utc'])).total_seconds()
    else:
        elapsed = history['capture_stats']['capture_elapsed_seconds'] + finish['finish_decode_ms']/1000
    request = next(e for e in history['events'] if e['type'] == 'finish_requested')
    updates.append(dict(observed=request['seconds'], elapsed=elapsed, hypothesis=finish['selected']))
    inference_ms.extend(e['latency_ms']['total'] for e in events)
    item = replay['item_id']
    row = dict(item_id=item, role=replay['role'], history=replay['history'], final=finish['selected'],
               finish_decode_ms=finish['finish_decode_ms'], windows=len(events),
               capture=history['capture_stats'], partial_tail=finish['final_visual_redecode']['partial_tail_included'],
               online_word_replacements=sum(bool(e['replaced_tail']) for e in events))
    if item in pool:
        source = pool[item]
        row['cache_matches'] = row['final'] == cached_rows[item]['final']
        row['reference'] = source['reference']
        operations = Counter(o['operation'] for o in _edit_operations(source['reference'], row['final']))
        row['operations'] = dict(operations)
        all_operations.update(operations)
        if source['verified_reference_intervals']:
            intervals = [i for i in source['intervals'] if i['label'] != '__OTHER__']
            first = [None] * len(intervals)
            for update in updates:
                available = [i for i, interval in enumerate(intervals) if interval['start_seconds'] <= update['observed']]
                for operation in _edit_operations([intervals[i]['label'] for i in available], update['hypothesis']):
                    if operation['operation'] == 'match':
                        index = available[operation['reference_position']]
                        if first[index] is None:
                            first[index] = update['elapsed']
            row['total_signs'] = len(intervals)
            row['missed_signs'] = first.count(None)
            row['first_correct_elapsed_seconds'] = first
            row['delays_seconds'] = [t-i['end_seconds'] for t, i in zip(first, intervals) if t is not None]
            delays.extend(row['delays_seconds'])
    rows.append(row)

total = sum(r.get('total_signs', 0) for r in rows)
missed = sum(r.get('missed_signs', 0) for r in rows)
stats = dict(checkpoint_sha256=cached['checkpoint_sha256'], rows=rows, runtime_recordings=len(rows),
             manifest_sha256=cached['manifest_sha256'],
             annotated_item_ids=[r['item_id'] for r in rows if 'total_signs' in r],
             annotated_recordings=sum('total_signs' in r for r in rows), total_signs=total, missed_signs=missed,
             missed_sign_frequency=missed/total, median_delay_seconds=float(np.median(delays)) if delays else None,
             p95_delay_seconds=float(np.percentile(delays, 95)) if delays else None,
             window_inference_median_ms=float(np.median(inference_ms)),
             window_inference_p95_ms=float(np.percentile(inference_ms, 95)),
             raw_cache_matching_recordings=sum(r.get('cache_matches', False) for r in rows),
             compared_recordings=sum('cache_matches' in r for r in rows),
             complete_annotated_pool=False, latency_includes_runtime=True,
             limitation='Subset diagnostic; fresh-process capture startup and Finish included; concurrent offline evaluation used CPU. Not a warmed-camera or full-pool eligibility measurement. HUNGRY has no transcript/latency truth.')
(REPORT/'runtime_summary.json').write_text(json.dumps(stats, indent=2)+'\n')
print(json.dumps({k: v for k, v in stats.items() if k != 'rows'}, indent=2))
