"""Audit saved decisions; no inference, fitting, threshold tuning or runtime edits."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_temporal_boundary_v17 import edit_counts, summarize

REPORT = Path(__file__).parent

def audit(records, baseline, oracle=False):
    counts, rows = Counter(), []
    for record in records:
        predictions = record['predictions'] if oracle else json.loads(Path(record['history']).read_text())['predictions']
        hypothesis = []
        for p in predictions:
            if p.get('ignored_after_reset'):
                continue
            proposal, verifier = p.get('proposal', {}), p.get('verifier', {})
            counts['completed_candidates'] += 1
            counts['proposal_rejected'] += not proposal.get('accepted', False)
            counts['verifier_rejected'] += not verifier.get('accepted', False)
            # Fixed existing commit score, no tuning. Counterfactual only: make the accepted full verifier the score
            # authority and remove the proposal veto. This can also drop an original
            # commit whose score was boosted by proposal agreement.
            commit = bool(verifier.get('accepted')) and verifier.get('model_score', 0) >= .45
            if commit:
                hypothesis.append(verifier['candidate_gloss'])
            counts['newly_emitted'] += commit and not bool(p.get('committed_gloss'))
            counts['removed_original_commits'] += not commit and bool(p.get('committed_gloss'))
        rows.append(dict(id=record['id'], reference=record['reference'], hypothesis=hypothesis,
                         metrics=edit_counts(record['reference'], hypothesis)))
    return dict(decisions=dict(counts), verifier_only_counterfactual=summarize(rows, baseline), records=rows)


def main():
    baseline = json.loads((REPORT / 'baseline_oracle.json').read_text())
    learned = json.loads((REPORT / 'learned_results.json').read_text())
    old = baseline['results']['reel']
    result = {'scope': 'Saved-candidate counterfactual; not new live accuracy. Same .45 commit score; unchanged intervals and verifier acceptance. No evaluation-based threshold selection.'}
    result['oracle_context100'] = audit(baseline['results']['oracle_trim_0_context_100'], old, True)
    result['runs'] = [dict(seed=r['training']['seed'], hand_geometry=r['training']['hand_geometry'],
                           original=r['summary'], **audit(r['records'], old)) for r in learned['runs']]
    assert len(result['runs']) == 4
    assert all(r['verifier_only_counterfactual']['references'] == 24 for r in result['runs'])
    assert sum(r['metrics']['correct'] for r in old) == 4
    (REPORT / 'decision_diagnostic.json').write_text(json.dumps(result, indent=2)+'\n')
    for r in [result['oracle_context100'], *result['runs']]:
        print(r.get('seed','oracle'),r.get('hand_geometry'),r['decisions'],r['verifier_only_counterfactual'])

if __name__ == '__main__':
    main()
