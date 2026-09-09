#!/usr/bin/env python3
"""Package both preserved-emission candidates with a shared train-calibrated margin."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.model_stage2_v17 import load_stage2_other_preserving
from active.v17.train_stage_2_other_ctc_v17 import directory_sha256


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    torch.set_num_threads(2)
    cache = torch.load(args.train_cache, map_location='cpu', weights_only=False)
    if cache['role'] != 'train':
        raise ValueError('margin calibration requires training data')
    expected_sources = {'asllrp_other_ctc', 'asllrp_contiguous', 'local_phrases',
                        'isolated_citizen_train', 'asl_stem_wiki_verified_interval',
                        'asllrp_segmented_train'}
    if {row['source'] for row in cache['rows']} != expected_sources:
        raise ValueError('unexpected calibration sources')
    for record in cache['models'].values():
        if sha256(ROOT / record['path']) != record['sha256']:
            raise ValueError('cached model provenance changed')
    accepted = torch.load(ROOT / cache['models']['accepted']['path'], map_location='cpu', weights_only=False)
    candidates = []
    maxima = {}
    for seed in (1701, 1702):
        evidence = torch.load(ROOT / cache['models'][str(seed)]['path'], map_location='cpu', weights_only=False)
        head_path = args.head_root / f'head_{seed}.pth'
        head = torch.load(head_path, map_location='cpu', weights_only=False)
        state = evidence['model_state_dict']
        maximum = -math.inf
        for row, hidden, known in zip(cache['rows'], cache['models'][str(seed)]['values'], cache['models']['accepted']['values']):
            if row['source'] == 'asllrp_other_ctc':
                continue
            path = known.argmax(-1)
            normalizer = torch.nn.functional.linear(hidden, state['ctc_head.weight'][:101], state['ctc_head.bias'][:101]).logsumexp(-1)
            odds = torch.nn.functional.linear(hidden, head['head_state_dict']['weight'], head['head_state_dict']['bias']).squeeze(-1) - normalizer
            scores = known.log_softmax(-1)
            starts = [0] + (torch.where(path[1:] != path[:-1])[0] + 1).tolist() + [len(path)]
            for start, end in zip(starts, starts[1:]):
                token = int(path[start])
                if token:
                    maximum = max(maximum, float((odds[start:end] - scores[start:end, token]).mean()))
        maxima[str(seed)] = maximum
        candidates.append(dict(
            format='slt_stage2_other_preserving_ctc_v17', version=1, seed=seed,
            accepted_checkpoint=accepted, evidence_checkpoint=evidence,
            other_head_state_dict=head['head_state_dict'], head_epoch=head['epoch'],
            accepted_sha256=cache['models']['accepted']['sha256'],
            evidence_sha256=cache['models'][str(seed)]['sha256'],
            other_head_sha256=sha256(head_path), training_cache_sha256=sha256(args.train_cache),
            encoder_sha256=evidence['encoder_sha256'],
            citizen_test_accessed=False, semlex_test_accessed=False, local_test_accessed=False,
        ))
    # The factor was selected during development; it is not a test-independent claim.
    margin = max(0., *maxima.values()) + math.log(2.)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for payload in candidates:
        payload.update(margin=margin, negative_train_maxima=maxima,
                       margin_disclosure='Shared negative-training maximum plus log(2); factor and run policy selected after development diagnostics.')
        for name in ('primary', 'specialist'):
            payload[f'runtime_{name}_sha256'] = directory_sha256(ROOT / f'artifacts/coreml/Stage2Selector{name.title()}V17FP32.mlpackage')
        payload['runtime_encoder_sha256'] = directory_sha256(ROOT / 'artifacts/coreml/Stage2FrozenEncoderV17FP32.mlpackage')
        payload['runtime_image_encoder_sha256'] = directory_sha256(ROOT / 'artifacts/coreml/MobileCLIP2S0ImageEncoderV17FP32.mlpackage')
        destination = args.output / f"seed_{payload['seed']}.pth"
        torch.save(payload, destination)
        load_stage2_other_preserving(destination)
        records.append(dict(path=str(destination), sha256=sha256(destination), seed=payload['seed']))
    result = dict(margin=margin, negative_train_maxima=maxima, candidates=records)
    (args.output / 'packaging.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache', type=Path, default=ROOT / 'artifacts/models/stage2_v17_other_preservation_v1/train_cache.pt')
    parser.add_argument('--head-root', type=Path, default=ROOT / 'artifacts/models/stage2_v17_other_preservation_v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/models/stage2_v17_transition_repair_v3')
    print(json.dumps(run(parser.parse_args()), indent=2))
