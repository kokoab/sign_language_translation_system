"""Recipe-scoped native-clock ASL start/end training; no phrase/CTC or gap labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from active.v17.approved_phrase_data_v17 import digest, verify_manifest
from active.v17.stage1_window_v17 import RAW_FORMAT
from active.v17.temporal_boundary_v17 import (
    FORMAT, TemporalBoundary, boundary_features, boundary_targets, masked_boundary_loss,
)

REPORT = ROOT / 'artifacts/reports/asl_temporal_boundary_v17_20260922'
RECIPE = ROOT / 'active/v17/temporal_boundary_manifest_20260922.json'
CONFIDENT = ROOT / 'artifacts/reports/confident_supervision_v17_20260920/confident_supervision.json'
COMBINED = ROOT / 'data/local/combined_dataset_v17_20260922/manifest.json'
SOURCE = ROOT / 'artifacts/reports/o5s5_citizen100_v17/combined_supervision.json'
CURATED = ROOT / 'artifacts/reports/clean_boundary_subset_20260920/curated_manifest.json'
MODELS = ROOT / 'artifacts/models/asl_temporal_boundary_v17_20260922'


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def code_hashes():
    return {name: digest(ROOT / name) for name in (
        'active/v17/temporal_boundary_v17.py', 'scripts/train_temporal_boundary_v17.py')}


def safe_path(name):
    path = (ROOT / name).resolve()
    if ROOT not in path.parents or {'test', 'external_evaluation_reserved'} & {p.casefold() for p in path.parts}:
        raise ValueError('unapproved/protected source path')
    return path


def event_key(row, source_rows):
    if row['source'] == 'o5s5':
        return (row['role'], row['source_item_id'], round(row['start_seconds'], 3), round(row['end_seconds'], 3), row['label'])
    parent = row['source_item_id'].split(':')[1].removesuffix('.mp4')
    source = source_rows[(row['source'], row['role'], row['source_item_id'])]
    event = source['intervals'][int(row['identity'].rsplit(':', 1)[-1])]
    if event['label'] != row['label']:
        raise ValueError('source event identity mismatch')
    return (parent, event['annotation_start_frame_global'], event['annotation_end_frame_global'], row['label'])


def read_raw(path):
    with np.load(path, allow_pickle=False) as z:
        if str(z['raw_format'].item()) != RAW_FORMAT:
            raise ValueError('raw Apple Vision schema mismatch')
        raw, times = z['raw_features'].astype(np.float32), z['timestamps_seconds'].astype(np.float64)
        meta = json.loads(str(z['metadata_json'].item()))
    boundary_features(raw, times)  # finite/clock/shape validation, without time normalization
    return raw, times, meta


def validate_event_coverage(times, intervals):
    for start, end in intervals:
        if not np.isfinite([start, end]).all() or end <= start or start < times[0] - 1e-8 or end > times[-1] + 1e-8:
            raise ValueError('accepted sign interval is outside the raw clock')
        if any(not np.any(np.abs(times - edge) <= .050001) for edge in (start, end)):
            raise ValueError('accepted boundary has no observed sample in its target band')


def prepare():
    if RECIPE.exists():
        raise FileExistsError('recipe exists; use --preflight rather than overwrite provenance')
    canonical = verify_manifest()
    confident, combined, source = (json.loads(p.read_text()) for p in (CONFIDENT, COMBINED, SOURCE))
    if digest(SOURCE) != confident['provenance']['source_manifest_sha256'] or digest(CURATED) != confident['provenance']['strict_manifest_sha256']:
        raise ValueError('confident annotation provenance changed')
    source_rows = {(r['source'], r['role'], r['source_item_id']): r for r in source['rows']}
    o5 = {(r['role'], r['source_item_id'], round(r['start_seconds'], 3), round(r['end_seconds'], 3), r['canonical_label']): r
          for r in combined['records'] if r['source'] == 'o5s5'}
    unique = {}
    for r in sorted(confident['rows'], key=lambda r: (r['source'] != 'asllrp_contiguous', r['archive_path'], r['identity'])):
        if r['role'] not in ('train', 'validation'):
            raise ValueError('unsupported data role')
        if r['source'] not in ('asllrp_contiguous', 'asllrp_other_ctc', 'o5s5'):
            continue
        key = event_key(r, source_rows)
        if r['source'] == 'o5s5' and (r['target_kind'] != 'known' or key not in o5):
            continue
        if key in unique and (unique[key]['role'], unique[key]['signer_id']) != (r['role'], r['signer_id']):
            raise ValueError('duplicate source event crosses roles or signer identities')
        unique.setdefault(key, r)
    grouped = defaultdict(list)
    for row in unique.values():
        grouped[(row['role'], row['source'], row['source_item_id'], row['archive_path'])].append(row)
    records, counts = [], Counter()
    physical_roles = defaultdict(set)
    for (role, src, item, archive), events in sorted(grouped.items()):
        events = sorted(events, key=lambda e: (e['start_seconds'], e['end_seconds']))
        raw, times, metadata = read_raw(safe_path(archive))
        contract = metadata['observer_contract']
        if contract != source['observer_contract']:
            raise ValueError('raw observer contract mismatch: ' + archive)
        video = events[0]['video_path']
        if metadata['source_item_id'] != item or metadata['video_path'] != video:
            raise ValueError('raw source/video identity mismatch')
        if digest(safe_path(video)) != metadata['video_sha256']:
            raise ValueError('video hash mismatch: ' + video)
        if src == 'o5s5':
            for event in events:
                current = o5[event_key(event, source_rows)]
                if digest(safe_path(archive)) != current['raw_feature_sha256']:
                    raise ValueError('current O5S5 raw archive mismatch')
        parent = item if src == 'o5s5' else item.split(':')[1].removesuffix('.mp4')
        physical_roles[(src == 'o5s5', parent)].add(role)
        # Calibration is a deterministic parent-held subset of training, never validation.
        split = 'calibration' if role == 'train' and int(hashlib.sha256(parent.encode()).hexdigest()[:8], 16) % 7 == 0 else role
        accepted = [[float(e['start_seconds']), float(e['end_seconds'])] for e in events]
        validate_event_coverage(times, accepted)
        y = boundary_targets(times, accepted, [], False)
        counts[f'{role}:{src}:events'] += len(events)
        counts[f'{split}:supervised_values'] += int((y >= 0).sum())
        records.append(dict(role=role, split=split, source=src, item=item, parent=parent,
                            signer=events[0]['signer_id'], raw_path=archive, raw_sha256=digest(safe_path(archive)),
                            video_path=video, video_sha256=metadata['video_sha256'], frames=len(times),
                            intervals=accepted, event_ids=[e['identity'] for e in events]))
    if any(len(roles) != 1 for roles in physical_roles.values()):
        raise ValueError('parent video crosses training/validation roles')
    recipe = dict(format=FORMAT, training_entrypoint='scripts/train_temporal_boundary_v17.py',
                  training_ready=True, scope='boundary edges only; no lexical targets, background or phrase CTC',
                  supervision='start/end positives; negatives only inside accepted intervals; all other values ignored',
                  canonical_verification=canonical, code_sha256=code_hashes(),
                  inputs={str(p.relative_to(ROOT)): digest(p) for p in (CONFIDENT, COMBINED, SOURCE, CURATED)},
                  seeds=[17621, 17622], hand_geometry_arms=[False, True], epochs=8, lr=.001,
                  hidden=64, lookahead=4, chunk_frames=128, batch_size=16,
                  observer_contract=source['observer_contract'],
                  selection='minimum parent-held training-calibration edge BCE; validation never selects epochs',
                  counts=dict(counts), records=records, test_accessed=False,
                  limitation='No independently annotated low-motion/hold/repeat phone set; long events over1.2s excluded by inherited quality contract; no supervised outside class.')
    atomic(RECIPE, recipe)
    atomic(REPORT / 'preparation.json', dict(records=len(records), counts=dict(counts), recipe_sha256=digest(RECIPE)))
    print(json.dumps(dict(records=len(records), counts=dict(counts))), flush=True)


def load_recipe():
    recipe = json.loads(RECIPE.read_text())
    if recipe.get('format') != FORMAT or recipe.get('training_entrypoint') != 'scripts/train_temporal_boundary_v17.py' or not recipe.get('training_ready'):
        raise ValueError('wrong/unready dedicated boundary recipe')
    if recipe['code_sha256'] != code_hashes():
        raise ValueError('boundary training code hash changed')
    for name, sha in recipe['inputs'].items():
        if digest(safe_path(name)) != sha:
            raise ValueError('source manifest hash changed: ' + name)
    if digest(ROOT / 'active/v17/approved_phrase_manifest_20260921_v2.json') != recipe['canonical_verification']['sha256']:
        raise ValueError('canonical phrase contract changed')
    return recipe


def data(recipe, geometry):
    output = defaultdict(list)
    for record in recipe['records']:
        path = safe_path(record['raw_path'])
        if digest(path) != record['raw_sha256']:
            raise ValueError('raw sequence changed: ' + str(path))
        raw, times, metadata = read_raw(path)
        if metadata['observer_contract'] != recipe['observer_contract']:
            raise ValueError('prepared observer contract changed')
        validate_event_coverage(times, record['intervals'])
        x = boundary_features(raw, times, geometry)
        y = boundary_targets(times, record['intervals'], [], False)
        # Never allow frames beyond a raw timestamp discontinuity into one receptive field.
        breaks = [0, *(np.flatnonzero(np.diff(times) > .26) + 1).tolist(), len(times)]
        for first, last in zip(breaks, breaks[1:]):
            for start in range(first, last, recipe['chunk_frames']):
                stop = min(start + recipe['chunk_frames'], last)
                left = max(first, start - 30)
                right = min(last, stop + recipe['lookahead'])
                targets = y[left:right - recipe['lookahead']].copy()
                targets[:start - left] = -1
                if len(targets) and (targets >= 0).any():
                    output[record['split']].append((x[left:right], targets))
    if any(not output[split] for split in ('train', 'calibration', 'validation')):
        raise ValueError('empty supervised split')
    return output


def batch(rows, device):
    length, dim = max(len(x) for x, _ in rows), rows[0][0].shape[1]
    xs = np.zeros((len(rows), length, dim), np.float32)
    ys = np.full((len(rows), length - 4, 2), -1, np.float32)
    for i, (x, y) in enumerate(rows):
        xs[i, :len(x)], ys[i, :len(y)] = x, y
    return torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device)


def positive_weights(rows, device):
    y = np.concatenate([y for _, y in rows])
    positive = (y == 1).sum(0)
    if (positive == 0).any():
        raise ValueError('missing start/end training positives')
    return torch.as_tensor(np.clip((y == 0).sum(0) / positive, 1, 10), dtype=torch.float32, device=device)


@torch.inference_mode()
def evaluate_loss(model, rows, weight, device, size=16):
    model.eval()
    total = count = 0
    for start in range(0, len(rows), size):
        x, y = batch(rows[start:start + size], device)
        logits = model(x)
        loss = masked_boundary_loss(logits, y, weight)
        n = int((y >= 0).sum())
        total += float(loss) * n; count += n
    return total / count


def preflight():
    recipe = load_recipe()
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required for this training recipe')
    torch.set_num_threads(2)
    rows = data(recipe, True)
    model = TemporalBoundary(450, recipe['hidden'], recipe['lookahead']).to('mps')
    weight = positive_weights(rows['train'], 'mps')
    x, y = batch(rows['train'][:16], 'mps')
    loss = masked_boundary_loss(model(x), y, weight)
    loss.backward()
    if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('nonfinite boundary gradients')
    value = dict(status='passed', optimizer_steps=0, device='mps', loss=float(loss),
                 chunks={k: len(v) for k, v in rows.items()}, parameters=sum(p.numel() for p in model.parameters()),
                 recipe_sha256=digest(RECIPE))
    atomic(REPORT / 'preflight.json', value)
    print(json.dumps(value), flush=True)


def train():
    recipe = load_recipe()
    precheck = json.loads((REPORT / 'preflight.json').read_text())
    if precheck['status'] != 'passed' or precheck['recipe_sha256'] != digest(RECIPE):
        raise ValueError('matching preflight required')
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required')
    MODELS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); results = []
    for geometry in recipe['hand_geometry_arms']:
        rows = data(recipe, geometry)
        weight = positive_weights(rows['train'], 'mps')
        for seed in recipe['seeds']:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = TemporalBoundary(450 if geometry else 366, recipe['hidden'], recipe['lookahead']).to('mps')
            optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['lr'], weight_decay=.0001)
            name = f'seed_{seed}_geometry_{int(geometry)}'
            path = MODELS / (name + '.pth')
            if path.exists():
                raise FileExistsError('refusing to overwrite trained checkpoint: ' + str(path))
            best, history = float('inf'), []
            for epoch in range(1, recipe['epochs'] + 1):
                model.train(); order = list(range(len(rows['train']))); random.shuffle(order)
                total = 0.
                for start in range(0, len(order), recipe['batch_size']):
                    selected = [rows['train'][i] for i in order[start:start + recipe['batch_size']]]
                    x, y = batch(selected, 'mps')
                    optimizer.zero_grad(set_to_none=True)
                    loss = masked_boundary_loss(model(x), y, weight)
                    if not torch.isfinite(loss): raise ValueError('nonfinite training loss')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                    total += float(loss) * len(selected)
                calibration = evaluate_loss(model, rows['calibration'], weight, 'mps')
                history.append(dict(epoch=epoch, train_loss=total / len(order), calibration_loss=calibration))
                if calibration < best:
                    best = calibration
                    payload = dict(format=FORMAT, model_config=model.config, model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                   hand_geometry=geometry, seed=seed, epoch=epoch, recipe_path=str(RECIPE.relative_to(ROOT)),
                                   recipe_sha256=digest(RECIPE), promoted=False, output_names=['start', 'end'])
                    tmp = path.with_suffix('.tmp'); torch.save(payload, tmp); tmp.replace(path)
                atomic(REPORT / (name + '_history.json'), history)
                print(name, history[-1], flush=True)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(saved['model_state_dict'])
            results.append(dict(seed=seed, hand_geometry=geometry, selected_epoch=saved['epoch'], checkpoint=str(path.relative_to(ROOT)),
                                checkpoint_sha256=digest(path), calibration_loss=best,
                                validation_loss=evaluate_loss(model, rows['validation'], weight, 'mps')))
    result = dict(status='complete', runs=results, elapsed_seconds=time.perf_counter() - started,
                  recipe_sha256=digest(RECIPE), promoted=False,
                  conclusion='Edge losses are not recognition accuracy; run the paired whole-video evaluation.')
    atomic(REPORT / 'training_results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'preflight', 'train'])
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'preflight': preflight()
    else:
        try:
            result = train()
            atomic(REPORT / 'completion.json', result)
            message = 'ASL boundary training complete; whole-video evaluation still required.'
        except Exception:
            atomic(REPORT / 'completion.json', dict(status='failed', traceback=traceback.format_exc()))
            message = 'ASL boundary training failed; see completion report.'
            raise
        finally:
            subprocess.run(['osascript', '-e', 'display notification ' + json.dumps(message) + ' with title "SLT temporal boundary"'], check=False)


if __name__ == '__main__': main()
