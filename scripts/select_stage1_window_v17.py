#!/usr/bin/env python3
"""Select only a candidate that passes every frozen complete-streaming gate."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.evaluate_stage1_window_v17 import failed_gates
from active.v17.train_stage_2_other_ctc_v17 import sha256


def select(report_root, evaluations, runtime_path=None):
    freeze = json.loads((report_root/'development_freeze.json').read_text())
    if freeze.get('status') != 'baselines_frozen':
        raise ValueError('baselines are not frozen')
    manifest = report_root/'evaluation_manifest.json'
    pool = json.loads(manifest.read_text())
    identities = {r['source_item_id'] for r in pool['rows']}
    annotated_identities = {r['source_item_id'] for r in pool['rows']
                            if r.get('verified_reference_intervals')}
    manifest_hash = sha256(manifest)
    ctc = json.loads((ROOT/freeze['ctc_baseline']).read_text())['metrics']['asllrp_other_ctc']
    familiar = json.loads((report_root/'older_reel_familiar/summary.json').read_text())
    transitions = json.loads((report_root/'transitions/baseline.json').read_text())
    runtime = json.loads(runtime_path.read_text()) if runtime_path else {}
    rows = []
    for path in sorted(evaluations.glob('epoch_*.json')):
        report = json.loads(path.read_text())
        observed = {r['item_id'] for r in report['rows']}
        connected = report['metrics']['asllrp_other_ctc']
        own_start = report['isolated_baseline']
        baseline = dict(connected_wer=ctc['wer_percent'], connected_insertions=ctc['insertions'],
                        connected_deletions=ctc['deletions'], familiar_wer=familiar['wer_percent'],
                        citizen_accuracy=own_start['citizen'], semlex_accuracy=own_start['semlex'],
                        transition_false_emissions=transitions['ctc_false_emissions'])
        measured = runtime.get(report['checkpoint_sha256'], {})
        measured_ids = measured.get('annotated_item_ids')
        runtime_matches = (measured.get('manifest_sha256') == manifest_hash
                           and isinstance(measured_ids, list)
                           and len(measured_ids) == len(annotated_identities)
                           and set(measured_ids) == annotated_identities)
        candidate = dict(connected_wer=connected['wer_percent'], connected_insertions=connected['insertions'],
                         connected_deletions=connected['deletions'],
                         familiar_wer=report['metrics']['local_phrases']['wer_percent'],
                         citizen_accuracy=report['isolated']['citizen'], semlex_accuracy=report['isolated']['semlex'],
                         transition_false_emissions=report['transition_false_emissions'],
                         median_delay_seconds=measured.get('median_delay_seconds'),
                         matched_pool=(observed == identities and len(report['rows']) == len(identities)
                                       and report['manifest_sha256'] == manifest_hash),
                         complete_streaming=report.get('complete_streaming') is True,
                         latency_includes_runtime=(measured.get('complete_annotated_pool') is True
                                                   and runtime_matches))
        failures = failed_gates(candidate, baseline)
        unknown = [name for name in failures if candidate.get(name) is None
                   or (name == 'latency_includes_runtime' and not candidate[name])]
        rows.append(dict(epoch=report['epoch'], checkpoint=report['checkpoint'],
                         checkpoint_sha256=report['checkpoint_sha256'], candidate=candidate, baseline=baseline,
                         failed_gates=failures, unverified_gates=unknown,
                         measured_failed_gates=[f for f in failures if f not in unknown], eligible=not failures))
    if len(rows) != 12 or {r['epoch'] for r in rows} != set(range(1, 13)):
        raise ValueError('all 12 complete epoch evaluations are required')
    eligible = [r for r in rows if r['eligible']]
    def order(row):
        c = row['candidate']
        delay = c['median_delay_seconds']
        return c['connected_wer'], c['connected_deletions'], float('inf') if delay is None else delay, row['epoch']
    selected = min(eligible, key=order) if eligible else None
    return dict(eligible=selected is not None, selected=selected,
                diagnostic_checkpoint=min(rows, key=order), rows=rows,
                development_freeze_sha256=sha256(report_root/'development_freeze.json'),
                seed=17111, confirmation_seed_authorized=selected is not None,
                defaults_changed=False, protected_test_accessed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'artifacts/reports/stage1_window_v17')
    parser.add_argument('--evaluations', type=Path, required=True)
    parser.add_argument('--runtime-metrics', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = select(args.report, args.evaluations, args.runtime_metrics)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(eligible=result['eligible'], diagnostic_epoch=result['diagnostic_checkpoint']['epoch'])))


if __name__ == '__main__':
    main()
