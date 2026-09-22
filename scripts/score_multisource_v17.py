"""Multi-source phrase scoring: weighted aggregate plus per-source no-regression gates.

Replaces single-set selection. Local alone is 15 glosses / 6 phrase types / 3 signers and
cannot resolve candidates; pooling by token count lets the largest source decide; and
isolated (1,356 samples) would swamp any aggregate it entered. So:

  - phrase sources are combined by WEIGHTED MACRO average, not pooled counts
  - local carries the largest single weight as the deployment domain, but the other
    sources together outweigh it, so a local-only win cannot carry a candidate
  - isolated is a RETENTION CONSTRAINT, never an aggregate term
  - every source additionally gates on no-regression against the incumbent

Read-only over recorded result.json validation. Nothing trained, nothing promoted.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'artifacts/models'

# Deployment domain gets the largest single share; the rest together outweigh it.
WEIGHTS = {'local_phrases': 0.40, 'asllrp_contiguous': 0.20,
           'asllrp_other_ctc': 0.20, 'ncslgr_strict': 0.20}
ISOLATED = ('isolated:val', 'isolated:landmarks_v17', 'isolated_validation:val',
            'isolated_validation:landmarks_v17')
REGRESSION_LIMIT = 0.03       # max WER points a single source may lose vs incumbent
ISOLATED_LIMIT = 0.01         # max exact-accuracy points isolated may lose
INCUMBENT = 'unified_streaming_grounded_ctc_v17_v1'


def normalize(entry):
    """Older result.json files record wer/edits/target_tokens; newer use known_*."""
    if not isinstance(entry, dict) or 'samples' not in entry:
        return None
    wer = entry.get('known_wer', entry.get('wer'))
    if wer is None:
        return None
    return dict(samples=entry['samples'], exact=entry.get('exact', 0), known_wer=wer,
                known_edits=entry.get('known_edits', entry.get('edits', 0)),
                known_target_tokens=entry.get('known_target_tokens', entry.get('target_tokens', 0)))


def load(name):
    path = MODELS / name / 'result.json'
    if not path.exists():
        return None
    validation = (json.loads(path.read_text()).get('validation') or {})
    by_source = validation.get('by_source')
    if not isinstance(by_source, dict):
        return None
    phrase = {k: n for k, n in ((k, normalize(v)) for k, v in by_source.items() if k in WEIGHTS) if n}
    isolated = [v for k, v in by_source.items() if k in ISOLATED]
    if not phrase:
        return None
    samples = sum(v['samples'] for v in isolated)
    exact = sum(v['exact'] for v in isolated)
    return dict(name=name, phrase=phrase,
                isolated_exact=exact / samples if samples else None,
                isolated_samples=samples,
                overall=validation.get('exact_phrases', {}))


def score(entry):
    """Weighted macro WER over covered phrase sources, renormalized to what exists."""
    covered = {k: v for k, v in entry['phrase'].items() if v.get('samples')}
    total = sum(WEIGHTS[k] for k in covered)
    if not total:
        return None
    weighted = sum(WEIGHTS[k] * covered[k]['known_wer'] for k in covered) / total
    macro = sum(covered[k]['known_wer'] for k in covered) / len(covered)
    tokens = sum(covered[k]['known_target_tokens'] for k in covered)
    micro = (sum(covered[k]['known_edits'] for k in covered) / tokens) if tokens else None
    return dict(weighted=weighted, macro=macro, micro=micro,
                coverage=sorted(covered), weight_covered=total)


def gates(entry, incumbent):
    """A candidate must not regress any source, or isolated retention."""
    failures = []
    for source, value in entry['phrase'].items():
        base = incumbent['phrase'].get(source)
        if base and value['known_wer'] > base['known_wer'] + REGRESSION_LIMIT:
            failures.append('%s +%.1fpt WER' % (source, 100 * (value['known_wer'] - base['known_wer'])))
    missing = [s for s in incumbent['phrase'] if s not in entry['phrase']]
    if missing:
        failures.append('no coverage: ' + ','.join(sorted(missing)))
    if (entry['isolated_exact'] is not None and incumbent['isolated_exact'] is not None
            and entry['isolated_exact'] < incumbent['isolated_exact'] - ISOLATED_LIMIT):
        failures.append('isolated -%.1fpt' % (100 * (incumbent['isolated_exact'] - entry['isolated_exact'])))
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--incumbent', default=INCUMBENT)
    args = parser.parse_args()

    entries = [e for e in (load(p.name) for p in sorted(MODELS.iterdir()) if p.is_dir()) if e]
    incumbent = next((e for e in entries if e['name'] == args.incumbent), None)
    if incumbent is None:
        raise SystemExit('incumbent not found: ' + args.incumbent)

    rows = []
    for entry in entries:
        s = score(entry)
        if s is None:
            continue
        rows.append((entry, s, gates(entry, incumbent)))
    rows.sort(key=lambda r: r[1]['weighted'])

    print('weights: ' + ', '.join('%s=%.2f' % kv for kv in sorted(WEIGHTS.items())))
    print('isolated is a retention constraint, never an aggregate term')
    print('incumbent: %s\n' % args.incumbent)
    print('%-52s %9s %8s %8s %9s %6s %s' % (
        'checkpoint', 'WEIGHTED', 'macro', 'micro', 'isolated', 'srcs', 'gates'))
    for entry, s, failures in rows:
        mark = 'PASS' if not failures else 'FAIL: ' + '; '.join(failures[:2])
        if entry['name'] == args.incumbent:
            mark = 'incumbent'
        print('%-52s %8.2f%% %7.2f%% %7.2f%% %8s %6d %s' % (
            entry['name'][:52], 100 * s['weighted'], 100 * s['macro'],
            100 * s['micro'] if s['micro'] is not None else float('nan'),
            ('%.1f%%' % (100 * entry['isolated_exact'])) if entry['isolated_exact'] is not None else '—',
            len(s['coverage']), mark))

    best = next((r for r in rows if not r[2] and r[0]['name'] != args.incumbent), None)
    print()
    print('incumbent weighted WER: %.2f%%' % (100 * next(s for e, s, _ in rows if e['name'] == args.incumbent)['weighted']))
    if best:
        print('best passing candidate: %s at %.2f%%' % (best[0]['name'], 100 * best[1]['weighted']))
    else:
        print('no candidate passes every gate; incumbent retained')

    out = ROOT / 'artifacts/reports/multisource_selection_v17_20260922'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ranking.json').write_text(json.dumps(dict(
        weights=WEIGHTS, regression_limit=REGRESSION_LIMIT, isolated_limit=ISOLATED_LIMIT,
        incumbent=args.incumbent, note='weighted macro over phrase sources; isolated held out as a '
        'retention constraint; recorded validation only, nothing retrained or promoted',
        rankings=[dict(name=e['name'], **s, isolated_exact=e['isolated_exact'], gates=f)
                  for e, s, f in rows]), indent=1) + '\n')
    print('wrote', out / 'ranking.json')


if __name__ == '__main__':
    main()
