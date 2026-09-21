#!/usr/bin/env python3
"""Audited, matched Flores OTHER supplement; no protected test access."""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.train_youtube_motion_pilot_v17 as pilot
from scripts.prepare_grounded_streaming_data_v17 import write_archive
from active.v17.model_unified_streaming_ctc_v17 import load_unified_streaming_head
from active.v17.approved_phrase_data_v17 import APPROVED_ROOT, require_training_manifest
import numpy as np
import torch

DEVICE = 'auto'
REPORT = ROOT / 'artifacts/reports/flores_other_v17_20260921'
MODELS = ROOT / 'artifacts/models/flores_other_v17_20260921'
CACHE = ROOT / 'data/local/stage2_v17_flores_other_20260921'
PHRASES = APPROVED_ROOT / 'phrases'
INITIAL = ROOT / 'artifacts/models/youtube_motion_pretrain_v17_20260921'
MANIFEST = ROOT / 'active/v17/stage2_2m_flores_training_manifest_v17.json'
SOURCE = ROOT / 'data/local/stage2_v17_2m_flores_multimodal/train/two_m_flores_asl'


def save(name, value):
    (REPORT / name).write_text(json.dumps(value, indent=2) + '\n')


def map_targets(raw, labels):
    targets = []
    for word in raw.split():
        token = word.strip('.,!?;:"()[]{}').upper()
        target = labels[token] + 1 if token in labels else 101
        if not targets or target != 101 or targets[-1] != 101:
            targets.append(target)
    if not targets:
        raise ValueError('empty gloss transcript')
    return targets


def ctc_min_steps(targets):
    return len(targets) + sum(a == b for a, b in zip(targets, targets[1:]))


def device_setup(requested=None):
    torch.set_num_threads(2)
    requested = DEVICE if requested is None else requested
    device = torch.device(('mps' if torch.backends.mps.is_available() else 'cpu') if requested == 'auto' else requested)
    if device.type == 'mps':
        torch.mps.set_per_process_memory_fraction(.35)
    return device


def stage1():
    payload = torch.load(pilot.BASE, map_location='cpu', weights_only=False)
    model = pilot.SLTStage1V17(pilot.Stage1V17Config(**payload['model_config']))
    model.load_state_dict(payload['model_state_dict'], strict=True)
    return model, payload['label_to_index']


def prepare():
    if CACHE.exists():
        raise FileExistsError(CACHE)
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['two_m_flores_devtest_accessed'] is False
    rows = {r['source_item_id']: r for r in manifest['rows']}
    _, labels = stage1()
    assert len(labels) == 100
    seen = set(); hashes = set(); retained = []; excluded = []; tokens = Counter()
    reference_hashes = set()
    args = pilot.ctc_parser().parse_args([])
    for root in (PHRASES, args.other_root):
        for archive in root.glob('*/*/*.npz'):
            with np.load(archive, allow_pickle=False) as d:
                m = json.loads(str(d['metadata_json'].item()))
                if m.get('video_sha256'):
                    reference_hashes.add(m['video_sha256'])
    for archive in sorted(SOURCE.glob('*.npz')):
        with np.load(archive, allow_pickle=False) as d:
            m = json.loads(str(d['metadata_json'].item()))
            identity = m['source_item_id']; row = rows[identity]
            assert identity not in seen; seen.add(identity)
            assert row['role'] == m['role'] == 'train' and identity.startswith('2m_flores_dev:')
            assert m['schema']['landmark_schema_fingerprint'] == 'b872fa3dcc16aab5'
            assert m['video_sha256'] == row['video_sha256'] == pilot.base.sha256(ROOT / row['video_path'])
            assert row['video_sha256'] not in reference_hashes | hashes
            hashes.add(row['video_sha256'])
            features = d['landmarks']; ranges = d['window_source_ranges']
            assert features.shape[1:] == (32, 61, 5) and np.isfinite(features).all()
            assert len(ranges) == len(features) and ranges[0, 0] == 0
            assert np.all(ranges[:, 1] > ranges[:, 0]) and np.all(ranges[1:, 0] == ranges[:-1, 1])
            if m['dropped_tail_frames'] or ranges[-1, 1] != row['frame_count']:
                excluded.append({'id': identity, 'reason': 'incomplete source-frame coverage', 'tail_frames': m['dropped_tail_frames']})
                continue
            targets = map_targets(row['raw_gloss'], labels)
            frames = pilot.base.restore_source_frames(features, ranges)
            windows = pilot.base.rolling_windows(frames, 4, 8)
            assert len(frames) == row['frame_count'] and ctc_min_steps(targets) <= len(windows)
            tokens.update(targets)
            m.update(source='flores_other', raw_gloss=row['raw_gloss'],
                     target_sequence=[next(k for k,v in labels.items() if v == t-1) if t != 101 else '__OTHER__' for t in targets],
                     mapping_contract='exact casefolded gloss after outer punctuation; consecutive OTHER spans collapsed; known repeats retained',
                     source_archive_sha256=pilot.base.sha256(archive), original_manifest_sha256=pilot.base.sha256(MANIFEST))
            target = CACHE / 'train/flores_other' / archive.name
            write_archive(target, features, ranges, np.array(targets)-1, m)
            with np.load(target, allow_pickle=False) as check:
                assert np.array_equal(check['landmarks'], features)
                assert np.array_equal(check['target_indices'] + 1, targets)
            retained.append({'id': identity, 'raw_gloss': row['raw_gloss'], 'targets': targets,
                             'archive': str(target.relative_to(ROOT)), 'archive_sha256': pilot.base.sha256(target),
                             'frames': len(frames), 'ctc_steps': len(windows)})
    assert seen == set(rows) and retained
    audit = {'status': 'passed', 'acquired': len(rows), 'admitted': len(retained), 'excluded': excluded,
             'known_tokens': sum(v for k,v in tokens.items() if k != 101), 'other_spans': tokens[101],
             'known_label_coverage': len(set(tokens)-{101}), 'base_sha256': pilot.base.sha256(pilot.BASE),
             'raw_annotation_preserved': True, 'full_frame_coverage': True, 'official_test_accessed': False,
             'cross_dataset_signer_identity': 'unknown; local Flores signer IDs are not global IDs',
             'phrase_content_overlap': False, 'retained': retained}
    save('audit.json', audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('retained','excluded')}, indent=2), flush=True)


def preflight():
    approved_args = pilot.ctc_parser().parse_args([])
    approved_args.phrase_root = PHRASES
    require_training_manifest(approved_args)
    device = device_setup()
    audit = json.loads((REPORT / 'audit.json').read_text())
    assert audit['status'] == 'passed' and audit['base_sha256'] == pilot.base.sha256(pilot.BASE)
    for row in audit['retained']:
        assert pilot.base.sha256(ROOT / row['archive']) == row['archive_sha256']
    samples = pilot.base.phrase_sequences(CACHE, 'train', 4, 8)
    assert len(samples) == audit['admitted']
    model, _ = stage1()
    longest = sorted(samples, key=lambda s:len(s.windows), reverse=True)[:32]
    encoded = pilot.base.encode(model, longest, device, 32, 'window')
    payload = torch.load(INITIAL / 'baseline_17321_initial.pth', map_location='cpu', weights_only=False)
    head = pilot.UnifiedStreamingCTCHeadV17(pilot.UnifiedStreamingCTCConfig(**payload['head_config'])).to(device)
    for seed in pilot.SEEDS:
        initial = torch.load(INITIAL / f'baseline_{seed}_initial.pth', map_location='cpu', weights_only=False)
        assert initial['head_config'] == payload['head_config']
        head.load_state_dict(initial['head_state_dict'], strict=True)
    batch = pilot.base.collate(encoded)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.0001)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = head(batch['evidence'].to(device))
        loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)(logits.log_softmax(-1).transpose(0,1),batch['targets'].to(device),batch['lengths'],batch['target_lengths'])
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
        optimizer.step()
    pilot.PHRASE_ROOT = PHRASES
    inputs = pilot.behavior_inputs(device)
    assert len(inputs[0]) == 289
    save('preflight.json', {'status':'passed','longest_flores_steps':int(batch['lengths'][0]), 'real_ctc_loss':float(loss.detach()),
         'real_optimizer_step':True,'optimizer_steps':3,'full_batch_size':len(encoded),'mps_memory_fraction':.35 if device.type == 'mps' else None,'evaluation_sequences':289,'device':str(device),'official_test_accessed':False})
    print('Full cache audit, strict initialization, real Flores backward/optimizer and behavior input checks passed.', flush=True)


@torch.inference_mode()
def rejection(checkpoint, samples, device):
    model = load_unified_streaming_head(torch.load(checkpoint,map_location='cpu',weights_only=False),device=device)
    counts = Counter()
    for sample in samples:
        path = model(torch.from_numpy(sample.evidence.astype(np.float32))[None].to(device))[0].argmax(-1).cpu().tolist()
        known, _ = pilot.emission_steps(path)
        counts['samples'] += 1
        counts['other_emitted'] += 101 in path
        counts['other_without_expected_sign'] += 101 in path and sample.targets[0] not in known
    return dict(counts)


def run():
    approved_args = pilot.ctc_parser().parse_args([])
    approved_args.phrase_root = PHRASES
    require_training_manifest(approved_args)
    for name in ('audit.json', 'preflight.json', 'integration_smoke.json'):
        if json.loads((REPORT/name).read_text())['status'] != 'passed':
            raise ValueError(f'passed check required: {name}')
    if MODELS.exists():
        raise FileExistsError(MODELS)
    MODELS.mkdir()
    device = device_setup(); pilot.PHRASE_ROOT = PHRASES
    evaluation = isolated = None; results = {}
    for seed in pilot.SEEDS:
        for arm in ('without_flores','with_flores'):
            args = pilot.ctc_parser().parse_args([])
            args.base = pilot.BASE; args.phrase_root = PHRASES
            args.initial_head = INITIAL / f'baseline_{seed}_initial.pth'
            args.seed = seed; args.epochs = 18; args.embedding_batch_size = 32; args.device = str(device)
            args.output_dir = MODELS / f'{arm}_{seed}'
            args.supplement_root = CACHE if arm == 'with_flores' else None
            args.supplement_mass = .1
            save('status.json', {'state':'training','arm':arm,'seed':seed,'pid':os.getpid()})
            result = pilot.ctc_run(args)
            if evaluation is None:
                evaluation = pilot.behavior_inputs(device)
                model, labels = stage1()
                raw = pilot.base.isolated_sequences([args.citizen_root/'val',args.semlex_val_root/'landmarks_v17'],labels)
                isolated = pilot.base.encode(model,raw,device,32,'window')
            details = pilot.behavior(result['output'],evaluation,device)
            for row in details['predictions']:
                assert sum(row[k] for k in ('substitutions','deletions','insertions')) == pilot.base.edit_distance(row['expected'],row['predicted'])
            save(f'behavior_{arm}_{seed}.json',details)
            result['device'] = str(device)
            result['behavior'] = {k:v for k,v in details.items() if k != 'predictions'}
            result['known_sign_other_rejection'] = rejection(result['output'],isolated,device)
            results[f'{arm}_{seed}'] = {k:v for k,v in result.items() if k != 'history'}
            save('comparison.json',results)
    lines = ['# Flores OTHER matched comparison','','Same baseline initialization per seed; only added Flores supervision differs. No YouTube pretraining.','',
             '| Seed / arm | Local WER | ASLLRP WER | NCSLGR WER | Isolated exact | OTHER without expected isolated sign |',
             '|---|---:|---:|---:|---:|---:|']
    gates = []
    for key,result in results.items():
        v=result['validation']; s=v['by_source']; r=result['known_sign_other_rejection']
        lines.append(f"| {key} | {s['local_phrases']['known_wer']:.2%} | {s['asllrp_contiguous']['known_wer']:.2%} | {s['ncslgr_strict']['known_wer']:.2%} | {v['isolated']['exact_accuracy']:.2%} | {r.get('other_without_expected_sign',0)}/{r['samples']} |")
    for seed in pilot.SEEDS:
        a,b=[results[f'{arm}_{seed}'] for arm in ('without_flores','with_flores')]
        av,bv=a['validation'],b['validation']
        gates.append(bv['by_source']['local_phrases']['known_wer'] < av['by_source']['local_phrases']['known_wer'] and
                     all(bv['by_source'][s]['known_wer'] <= av['by_source'][s]['known_wer'] for s in ('asllrp_contiguous','ncslgr_strict')) and
                     bv['isolated']['exact_accuracy'] >= av['isolated']['exact_accuracy']-.01 and
                     b['known_sign_other_rejection'].get('other_without_expected_sign',0) <= a['known_sign_other_rejection'].get('other_without_expected_sign',0))
    lines += ['',f'Paired WER/retention/rejection gates passed: {sum(gates)}/2. No automatic promotion.',
              '', 'Full error counts, duplicate outputs, synthetic hold/repeat accuracy and conditional ASLLRP emission delay: comparison.json and behavior_*.json.',
              'No real held/repeat test or global Flores signer identities are established. Existing development sets are reused; this is not a new unbiased test result.']
    (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save('status.json', {'state':'complete','pid':os.getpid(),'paired_gates':gates})


def main():
    global DEVICE, REPORT, MODELS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true'); parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--device', choices=('auto','cpu','mps'), default='auto')
    parser.add_argument('--report-dir', type=Path, default=REPORT)
    parser.add_argument('--model-dir', type=Path, default=MODELS)
    args=parser.parse_args()
    DEVICE, REPORT, MODELS = args.device, args.report_dir.resolve(), args.model_dir.resolve()
    if args.prepare:
        prepare(); return
    if args.preflight:
        preflight(); return
    state='failed'
    try:
        run(); state='complete'
    except Exception:
        save('status.json',{'state':'failed','error':traceback.format_exc(),'pid':os.getpid()})
        (REPORT/'REPORT.md').write_text('# Flores OTHER experiment failed\n\nNo complete paired conclusion is available. Completed arms, if any, remain in comparison.json.\n\n```\n'+traceback.format_exc()+'```\n')
        raise
    finally:
        stamp=datetime.now().astimezone().isoformat(timespec='seconds')
        summary=f'Flores OTHER experiment {state}; reports: {REPORT.relative_to(ROOT)}; no promotion or protected test access.'
        history=ROOT/'docs/ground_truth/data-sources/log.md'
        history.write_text(history.read_text().replace('\n---\n','\n---\n\n## '+stamp+' — Flores OTHER completion\n\n'+summary+'\n',1))
        ground=ROOT/'PROJECT_GROUND_TRUTH.md'
        ground.write_text('\n'.join('**Flores OTHER 2026-09-21:** '+summary if line.startswith('**Flores OTHER 2026-09-21:**') else line for line in ground.read_text().splitlines())+'\n')
        notice=subprocess.run(['/usr/bin/osascript','-e','display notification '+json.dumps(summary)+' with title "SLT Flores experiment"'],capture_output=True,text=True)
        save('training_notification.json',{'state':state,'returncode':notice.returncode,'stderr':notice.stderr})
        subprocess.run([sys.executable,str(ROOT/'scripts/index_large_artifacts_v17.py')],check=False)

if __name__ == '__main__':
    main()
