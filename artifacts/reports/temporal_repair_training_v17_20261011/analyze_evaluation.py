"""Compare completed native outputs; coarse visual cores are development diagnostics."""
from pathlib import Path
import collections, csv, json

O = Path(__file__).resolve().parent
reviews = json.loads((O.parent / 'temporal_repair_contract_v17_20261011/phone_visual_review.json').read_text())['records']

def phases(path):
    records = json.loads(path.read_text())['records']
    result = {}
    for prefix in ['sept30', 'oct11', 'have_take11']:
        rows = [r for r in records if r['id'].startswith(prefix)]
        if prefix == 'sept30':
            result[prefix] = {'exact': sum(r['hypothesis'] == r['reference'] for r in rows), 'n': len(rows)}
            continue
        annotations = [a for a in reviews if a['label'] == ('MY' if prefix == 'oct11' else 'HAVE')]
        events = []
        all_correct = {r['id']: True for r in rows}
        all_following = {r['id']: True for r in rows}
        for annotation in annotations:
            outcomes = collections.Counter()
            followed_correctly = 0
            for row in rows:
                lo, hi = annotation['core_seconds']
                indexes = [i for i,w in enumerate(row['words']) if min(hi, w['end_seconds']) - max(lo, w['start_seconds']) > 1e-5]
                words = [row['words'][i]['gloss'] for i in indexes]
                outcomes[','.join(words) or 'no_overlapping_output'] += 1
                all_correct[row['id']] &= words == [annotation['label']]
                follows = words == [annotation['label']] and indexes[0]+1 < len(row['words']) and row['words'][indexes[0]+1]['gloss'] == annotation['following_sign']
                followed_correctly += follows
                all_following[row['id']] &= follows
            events.append({'id': annotation['id'], 'outcomes': dict(outcomes), 'target_only': outcomes[annotation['label']], 'followed_by_expected_next_sign': followed_correctly, 'n': len(rows)})
        result[prefix] = {'events': events, 'target_only': sum(e['target_only'] for e in events), 'n': len(events) * len(rows), 'all_events_target_only_runs': sum(all_correct.values()), 'target_then_following_sign': sum(e['followed_by_expected_next_sign'] for e in events), 'all_events_target_then_following_runs': sum(all_following.values())}
    return result

def controls(path):
    rows = json.loads(path.read_text())['records']
    return {label: {'n': len(rr), 'exact': sum(r['hypothesis'] == r['reference'] for r in rr), 'silent': sum(not r['hypothesis'] for r in rr), 'contains_MY': sum('MY' in r['hypothesis'] for r in rr), 'contains_LIKE': sum('LIKE' in r['hypothesis'] for r in rr)} for label in ['MY', 'LIKE'] for rr in [[r for r in rows if r['reference'] == [label]]]}

def validation(path):
    rows = json.loads(path.read_text())['records']
    base = {r['id']: r for r in json.loads((O / 'incumbent_validation.json').read_text())['records']}
    result = {}
    class_rows = []
    for dataset in sorted({r['dataset'] for r in rows}):
        rr = [r for r in rows if r['dataset'] == dataset]
        result[dataset] = {'n': len(rr), 'exact': sum(r['hypothesis'] == r['reference'] for r in rr), 'silent': sum(not r['hypothesis'] for r in rr), 'changed': sum(r['hypothesis'] != base[r['id']]['hypothesis'] for r in rr)}
        for label in sorted({r['reference'][0] for r in rr if len(r['reference']) == 1}):
            group = [r for r in rr if r['reference'] == [label]]
            before = sum(base[r['id']]['hypothesis'] == r['reference'] for r in group)
            after = sum(r['hypothesis'] == r['reference'] for r in group)
            class_rows.append(dict(dataset=dataset, label=label, n=len(group), before=before, after=after, delta=after-before))
    with path.with_suffix('.per_class.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(class_rows[0])); writer.writeheader(); writer.writerows(class_rows)
    result['citizen_class_changes'] = dict(collections.Counter('better' if r['delta'] > 0 else 'worse' if r['delta'] < 0 else 'same' for r in class_rows if r['dataset'] == 'citizen_val'))
    return result

if __name__ == '__main__':
    result = {'note': 'Source-video native Mac replays; correlated schedules of known development videos, not independent phone accuracy. Core-overlap counts use conservative reviewed interiors; no labels inferred for ambiguous later MY/LIKE or aborted HAVE.'}
    for name in ['incumbent', 'selected', 'diagnostic']:
        arm = {}
        for runtime in ['control', 'continuous_contact']:
            path = O / f'{name}_{runtime}_replay.json'
            if path.exists(): arm[runtime] = phases(path)
        for suffix, function in [('validation', validation), ('genuine_controls', controls)]:
            path = O / f'{name}_{suffix}.json'
            if path.exists(): arm[suffix] = function(path)
        if arm: result[name] = arm
    (O / 'evaluation_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
