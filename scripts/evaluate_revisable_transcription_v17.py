#!/usr/bin/env python3
"""Measure revisable sequence errors and annotated-gap emissions on matched dev data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.model_stage2_v17 import load_stage2_other_preserving
from active.v17.model_stage2_live_adapt_v17 import load_stage2_live_adapted
from active.v17.train_stage_2_v17 import RealPhraseDataset
from active.v17.train_stage_2_other_ctc_v17 import _edit_operations, sha256
from scripts.live_stage2_ctc_v17 import collapse_ctc_path, supported_ctc_path
from active.v17.continuous_decode_v17 import CTCPrefixDecoder


def revision_size(previous, current):
    common = 0
    for old, new in zip(previous, current):
        if old != new:
            break
        common += 1
    return len(previous) - common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--adapted', action='store_true')
    parser.add_argument('--supervision', type=Path, default=ROOT/'artifacts/reports/stage2_v17_revisable_v1/supervision.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--final-beam', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, payload = (load_stage2_live_adapted if args.adapted else load_stage2_other_preserving)(args.checkpoint)
    model.eval()
    cache = json.loads((ROOT/'artifacts/reports/stage2_v17_live_matched_v1/cache.json').read_text())
    details = {r['item_id']: r for r in cache['rows']}
    annotations = json.loads(args.supervision.read_text())['items']
    if args.adapted:
        assert payload['input_contract'] == cache['contract']
    dataset = RealPhraseDataset(ROOT/'data/local/stage2_v17_live_matched_v1/frozen_features', 'validation')
    results = []
    with torch.inference_mode():
        for row in dataset:
            timing = [w['end'] for w in details[row.item_id]['windows'] if w['accepted']]
            annotation = annotations.get(row.item_id, {})
            if annotation:
                assert annotation['role'] == 'validation'
            gaps = annotation.get('blank_positions', [])
            updates, previous = [], []
            for windows in range(1, len(row.features) + 1):
                x = torch.from_numpy(row.features[:windows].astype(np.float32))[None]
                mask = torch.ones((1, windows), dtype=torch.bool)
                started = time.perf_counter()
                logits, lengths = model(x, mask)
                head_ms = 1000 * (time.perf_counter() - started)
                length = int(lengths[0])
                tokens, positions = collapse_ctc_path(logits.numpy(), length)
                tokens, positions = supported_ctc_path(tokens, positions)
                hypothesis = list(tokens)
                path = logits[0, :length].argmax(-1).tolist()
                current_gaps = [p for p in gaps if p < length]
                updates.append(dict(windows=windows, source_end_seconds=timing[windows-1],
                    hypothesis=hypothesis, positions=list(positions), head_ms=head_ms,
                    revised_words=revision_size(previous, hypothesis),
                    annotated_gap_steps=len(current_gaps),
                    gap_known_emission_steps=sum(0 < path[p] <= 100 for p in current_gaps),
                    gap_nonblank_steps=sum(path[p] != 0 for p in current_gaps)))
                previous = hypothesis
            reference = [int(v) for v in row.targets if v <= 100]
            final_greedy = list(previous)
            beam_alternatives = []
            if args.final_beam:
                beam = CTCPrefixDecoder(beam_width=8, token_topk=12)
                for step in logits[0, :int(lengths[0])].log_softmax(-1).numpy():
                    # Existing beam requires finite values; vetoed paths remain negligible.
                    beam.step(np.maximum(step, -1e4))
                beam_alternatives = [dict(tokens=[int(t) for t in tokens if t <= 100], score=score)
                                     for tokens, score in beam.alternatives()]
                previous = beam_alternatives[0]['tokens']
            operations = Counter(v['operation'] for v in _edit_operations(reference, previous))
            first = next((v['source_end_seconds'] for v in updates if v['hypothesis']), None)
            results.append(dict(item_id=row.item_id, source=row.source, reference=reference,
                final=previous, operations=dict(operations), first_output_source_seconds=first,
                final_greedy=final_greedy, beam_alternatives=beam_alternatives,
                updates=updates, revision_events=sum(v['revised_words'] > 0 for v in updates),
                revised_words=sum(v['revised_words'] for v in updates)))
    metrics = {}
    for source in sorted({r['source'] for r in results}):
        rows = [r for r in results if r['source'] == source]
        tokens = sum(len(r['reference']) for r in rows)
        counts = Counter()
        for r in rows:
            counts.update(r['operations'])
        first = [r['first_output_source_seconds'] for r in rows if r['first_output_source_seconds'] is not None]
        finals = [r['updates'][-1] for r in rows]
        metrics[source] = dict(samples=len(rows), reference_tokens=tokens,
            substitutions=counts['substitution'], deletions=counts['deletion'], insertions=counts['insertion'],
            wer_percent=100*(counts['substitution']+counts['deletion']+counts['insertion'])/max(1,tokens),
            exact_phrases=sum(r['reference'] == r['final'] for r in rows),
            revision_events=sum(r['revision_events'] for r in rows),
            clips_with_revisions=sum(r['revision_events'] > 0 for r in rows),
            revised_words=sum(r['revised_words'] for r in rows),
            clips_with_output=len(first), median_first_output_source_seconds=float(np.median(first)) if first else None,
            annotated_gap_steps=sum(v['annotated_gap_steps'] for v in finals),
            gap_known_emission_steps=sum(v['gap_known_emission_steps'] for v in finals),
            gap_nonblank_steps=sum(v['gap_nonblank_steps'] for v in finals))
    output = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
        supervision_sha256=sha256(args.supervision), metrics=metrics, rows=results,
        protected_test_accessed=False, final_beam=args.final_beam,
        limitations='Repeated development evaluation. Time is source-window end, excludes extraction/scheduling. Gap metrics are annotation-to-output-bin proxies, not manual judgments of physical movements. No word-level semantic or translation accuracy claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
