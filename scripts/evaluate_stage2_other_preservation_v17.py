#!/usr/bin/env python3
"""Verify preserved-emission checkpoints on the frozen v2 development gates."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.model_stage2_v17 import load_stage2_other_preserving
from active.v17.train_stage_2_other_ctc_v17 import (
    CombinedDataset, EXPECTED_EXPERIMENT_MANIFEST_SHA256, EXPECTED_ENCODER_SHA256,
    _validate_feature_root, _validate_measured_baseline, _validate_stem_manifest,
    eligibility, evaluate_transition, sha256, validate_transition_manifest,
)
from active.v17.train_stage_2_accuracy_repair_v17 import IsolatedPoolDataset
from active.v17.train_stage_2_v17 import RealPhraseDataset, collate


def runtime_parity(dataset, checkpoint):
    import coremltools as ct
    from active.v17.export_stage2_coreml_v17 import fixed_input
    from active.v17.model_stage2_v17 import preserve_ctc_emission_runs
    from active.v17.train_stage_2_other_ctc_v17 import collapse_ctc
    from scripts.live_stage2_ctc_v17 import select_general_ctc_logits, validate_preservation_runtime
    model, payload = load_stage2_other_preserving(checkpoint)
    paths = [ROOT / f'artifacts/coreml/Stage2Selector{name}V17FP32.mlpackage' for name in ('Primary', 'Specialist')]
    selector = json.loads((ROOT / 'artifacts/reports/stage2_v17_general_ctc_selector_v1/validation.json').read_text())['selector_config']
    validate_preservation_runtime(SimpleNamespace(
        stage2_selector=ROOT / 'artifacts/models/stage2_v17_general_ctc_selector_v1/model.pth',
        stage2_primary=paths[0], stage2_specialist=paths[1],
        stage2_encoder=ROOT / 'artifacts/coreml/Stage2FrozenEncoderV17FP32.mlpackage',
        image_encoder=ROOT / 'artifacts/coreml/MobileCLIP2S0ImageEncoderV17FP32.mlpackage',
    ), payload, selector)
    heads = [ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL) for path in paths]
    names = [head.get_spec().description.output[0].name for head in heads]
    mismatches = []
    times = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            features, mask = fixed_input(row.features.astype(np.float32), 8)
            provider = dict(frozen_features=features, window_mask=mask)
            start = time.perf_counter()
            primary, specialist = [np.asarray(head.predict(provider)[name]) for head, name in zip(heads, names)]
            chosen, *_ = select_general_ctc_logits(
                primary, specialist, len(row.features) * 8,
                **{k: selector[k] for k in ('blend_weight', 'blank_bias', 'score_margin', 'minimum_tokens')},
            )
            base_done = time.perf_counter()
            odds, lengths = model.other_log_odds(torch.from_numpy(features), torch.from_numpy(mask > .5))
            actual = preserve_ctc_emission_runs(torch.from_numpy(chosen.reshape(1, -1, 101).copy()), odds, lengths, model.margin)
            done = time.perf_counter()
            expected, _ = model(torch.from_numpy(features), torch.from_numpy(mask > .5))
            a = collapse_ctc(actual[0, :lengths[0]].argmax(-1).numpy())
            b = collapse_ctc(expected[0, :lengths[0]].argmax(-1).numpy())
            if a != b:
                mismatches.append(dict(item_id=row.item_id, runtime=a, pytorch=b))
            times.append(dict(baseline_ms=1000 * (base_done-start), evidence_ms=1000 * (done-base_done), total_ms=1000 * (done-start)))
            if index % 100 == 0:
                print('runtime parity', index, 'mismatches', len(mismatches), flush=True)
    return dict(samples=len(dataset), mismatches=mismatches,
                latency_ms={key: dict(p50=float(np.median([r[key] for r in times])), p95=float(np.quantile([r[key] for r in times], .95))) for key in times[0]},
                disclosure='Mac cached-feature stage-2 timing; excludes video extraction, full Finish latency and iPhone thermals.')


def validation_dataset():
    manifest = validate_transition_manifest(
        ROOT / 'artifacts/reports/stage2_v17_transition_adapt_v2/manifest.json',
        expected_sha256=EXPECTED_EXPERIMENT_MANIFEST_SHA256,
    )
    roots = [
        ('stage2_v17_frozen_features', {'local_phrases', 'asllrp_contiguous'}),
        ('stage2_v17_asllrp_other_frozen_features', {'asllrp_other_ctc'}),
        ('stage2_v17_asllrp_segmented_validation_frozen_features', {'asllrp_segmented_validation'}),
        ('stage2_v17_transition_adapt_v2/frozen_features', {'asl_stem_wiki_verified_interval'}),
    ]
    datasets = []
    for name, sources in roots:
        root = ROOT / 'data/local' / name
        _validate_feature_root(root, sources, EXPECTED_ENCODER_SHA256)
        datasets.append(RealPhraseDataset(root, 'validation'))
    _validate_stem_manifest(ROOT / 'data/local/stage2_v17_transition_adapt_v2/frozen_features', manifest)
    datasets.append(IsolatedPoolDataset(
        ROOT / 'data/local/stage2_v17_isolated_replay/citizen_validation.npz',
        'isolated_citizen_validation', augment_boundaries=False,
    ))
    return CombinedDataset(datasets)


def run(args):
    torch.set_num_threads(2)
    dataset = validation_dataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate)
    results = []
    baseline = None
    for path in args.checkpoints:
        model, payload = load_stage2_other_preserving(path)
        if payload['encoder_sha256'] != EXPECTED_ENCODER_SHA256:
            raise ValueError('checkpoint encoder mismatch')
        if baseline is None:
            baseline = evaluate_transition(model.accepted, loader, torch.device('cpu'))
            _validate_measured_baseline(baseline['summary'])
        result = evaluate_transition(model, loader, torch.device('cpu'))
        passes = eligibility(result['summary'], baseline['summary']) and result['summary']['stem_correct'] >= 15
        results.append(dict(checkpoint=str(path), checkpoint_sha256=sha256(path),
                            seed=payload['seed'], eligible=passes, validation=result))
        print(payload['seed'], passes, result['summary'], flush=True)
    both_seeds = len(results) == 2 and {row['seed'] for row in results} == {1701, 1702}
    selected = min(results, key=lambda row: row['validation']['summary']['target_edits']) if both_seeds and all(row['eligible'] for row in results) else None
    report = dict(format='slt_stage2_other_preservation_validation_v17', baseline=baseline,
                  results=results, selected_checkpoint=selected['checkpoint'] if selected else None,
                  citizen_test_accessed=False, semlex_test_accessed=False, local_test_accessed=False,
                  disclosure='Repeated development selection; not independent test or iPhone performance.')
    if args.runtime and selected:
        report['runtime_parity'] = runtime_parity(dataset, Path(selected['checkpoint']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', type=Path, default=[
        ROOT / f'artifacts/models/stage2_v17_transition_repair_v3/seed_{seed}.pth' for seed in (1701, 1702)
    ])
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/reports/stage2_v17_transition_repair_v3/validation.json')
    parser.add_argument('--runtime', action='store_true')
    run(parser.parse_args())
