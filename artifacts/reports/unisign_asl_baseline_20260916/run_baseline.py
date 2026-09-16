"""One frozen, pretrained ASL comparison. No training or status polling.

--launch detaches; the worker writes REPORT.md and sends one exit notification.
--self-check checks the native pose adapter without downloading model weights.
"""
import argparse
import ast
import copy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'data/local/tools/Uni-Sign'
DATA = ROOT / 'data/local/unisign_asl_baseline_20260916'
DEPS = ROOT / 'data/local/tools/unisign_native_deps'
CODE_REV = 'eed438bcb49e30405cd6ccdfcccca330c134e830'
MODEL_REV = 'eab251b7fe7e8521afc0e67be98add670ea40a0d'
TEXT_REV = '2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f'
DATA_REV = '1231830fc1e8d77555a245ca22353b144e168111'
WEIGHT_SHA = '1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683fb00d'
sys.path[:0] = [str(DEPS), str(SOURCE), str(SOURCE / 'demo/rtmlib-main'), str(ROOT)]
from scripts.acquire_how2sign_transition_subset_v17 import download_file, sha256


def save(name, value):
    path = HERE / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def download(url, path, expected_sha=None):
    if not path.exists():
        print('Acquire', path.name, flush=True)
        download_file(url, path, timeout=90, retries=2)
    if expected_sha and sha256(path) != expected_sha:
        raise ValueError('Hash mismatch: ' + str(path))
    return path


def native_dataset():
    """Execute unchanged upstream pose-only definitions, excluding CUDA training imports."""
    import numpy as np
    import torch
    from torchvision import transforms
    from torch.nn.utils.rnn import pad_sequence
    names = {'crop_scale', 'load_part_kp', 'Base_Dataset', 'S2T_Dataset_online'}
    path = SOURCE / 'datasets.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if getattr(node, 'name', '') in names]
    assert {node.name for node in tree.body} == names
    namespace = dict(torch=torch, np=np, copy=copy, random=random,
                     Dataset=torch.utils.data, transforms=transforms,
                     pad_sequence=pad_sequence)
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['S2T_Dataset_online'], namespace['load_part_kp']


def self_check():
    import numpy as np
    import torch
    cls, load = native_dataset()
    rng = np.random.default_rng(42)
    keypoints = rng.uniform(.1, .9, (12, 1, 133, 2))
    scores = np.ones((12, 1, 133))
    ds = cls(SimpleNamespace(rgb_support=False, max_length=256))
    ds.pose_data = dict(keypoints=keypoints, scores=scores)
    src, tgt = ds.collate_fn([ds[0]])
    direct = load(keypoints, scores)
    for part, joints in [('body', 9), ('left', 21), ('right', 21), ('face_all', 18)]:
        assert src[part].shape == (1, 12, joints, 3)
        assert torch.equal(src[part][0], direct[part])
        assert torch.isfinite(src[part]).all()
    assert src['attention_mask'].shape == (1, 12)
    assert tgt['gt_sentence'] == [''], 'English references must never enter inference'
    masked = load(keypoints, np.zeros_like(scores))
    assert all(torch.count_nonzero(v) == 0 for v in masked.values())
    import onnxruntime
    from rtmlib import Wholebody
    assert 'lightweight' in Wholebody.MODE
    assert onnxruntime.__version__ == '1.19.2'
    print('PASS: native adapter shapes, direct equivalence, confidence masking, no reference input')


def prepare():
    import cv2
    DATA.mkdir(parents=True, exist_ok=True)
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == CODE_REV
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'diff', '--name-only'], text=True).strip()
    base = f'https://huggingface.co/datasets/aipieces/How2Sign/resolve/{DATA_REV}/'
    # Raw videos avoid the known mismatch between original sentence cuts and realigned labels.
    archive = download(base + 'val_raw_videos/shard_001_014.zip', DATA / 'val_raw_shard001.zip',
                       expected_sha='6608418616d5879e585799ec908c513ce469d901b42b439d7d614e61171f164f')
    labels = list(csv.DictReader((HERE / 'how2sign_realigned_val.csv').open(), delimiter='\t'))
    recordings = []
    with zipfile.ZipFile(archive) as zipped:
        members = {Path(n).stem: n for n in zipped.namelist() if n.endswith('.mp4')}
        chosen = []
        signer_counts = {}
        for video_name in sorted(members):
            signer = video_name.rsplit('-', 2)[-2]
            if signer_counts.get(signer, 0) >= 2:
                continue
            eligible = sorted([r for r in labels if r['VIDEO_NAME'] == video_name
                               and 2 <= float(r['END_REALIGNED']) - float(r['START_REALIGNED']) <= 12],
                              key=lambda r: r['SENTENCE_ID'])
            if len(eligible) >= 3:
                chosen.append((video_name, eligible[:3]))
                signer_counts[signer] = signer_counts.get(signer, 0) + 1
            if len(chosen) == 4:
                break
        if len(chosen) < 4:
            raise ValueError('First validation raw shard lacks four eligible source videos; do not silently change sample')
        # Selection is fixed before any model output. No sentence quality/recognizability selection.
        save('selection.json', dict(policy='First four sorted eligible raw validation video names in shard001, at most two videos per filename signer ID; first three sorted sentence IDs with duration 2–12s each',
                                    selected=[r for _, rows in chosen for r in rows],
                                    signer_source_counts=signer_counts,
                                    dataset_revision=DATA_REV, archive_sha256=sha256(archive)))
        for video_name, rows in chosen:
            raw = DATA / 'raw' / (video_name + '.mp4')
            raw.parent.mkdir(exist_ok=True)
            if not raw.exists():
                with zipped.open(members[video_name]) as src, raw.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
            raw_hash = sha256(raw)
            for row in rows:
                target = DATA / 'clips' / (row['SENTENCE_NAME'] + '.mp4')
                target.parent.mkdir(exist_ok=True)
                start, end = float(row['START_REALIGNED']), float(row['END_REALIGNED'])
                if not target.exists():
                    subprocess.run(['ffmpeg', '-v', 'error', '-n', '-ss', str(start), '-i', str(raw),
                                    '-t', str(end-start), '-an', '-c:v', 'libx264', '-crf', '18',
                                    '-pix_fmt', 'yuv420p', str(target)], check=True)
                cap = cv2.VideoCapture(str(target))
                fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                assert fps > 0 and abs(count/fps - (end-start)) < .15
                recordings.append(dict(item_id=row['SENTENCE_NAME'], video=str(target),
                                       video_sha256=sha256(target), reference=row['SENTENCE'],
                                       split='How2Sign validation', source_video=video_name,
                                       start=start, end=end, raw_sha256=raw_hash))
    previous = json.loads((HERE.parent / 'stage2_held_sign_diagnostics_20260916/manifest.json').read_text())
    for row in previous['recordings']:
        # The 20s unsegmented rollover trace is not a completed utterance.
        if row['item_id'] == 'webcam_long_context':
            continue
        assert sha256(Path(row['video'])) == row['video_sha256']
        recordings.append(dict(item_id=row['item_id'], video=row['video'],
                               video_sha256=row['video_sha256'], reference=None,
                               gloss_annotation=row['reference'], split='qualitative diagnostic',
                               annotation_status=row['annotation_status']))
    manifest = dict(recordings=recordings, protected_test_accessed=False,
                    paired_references='Realigned English validation CSV; raw frontal videos cut at corrected times',
                    limitation='12 selected development utterances from four sources, not a benchmark or signer-disjoint claim. Nine earlier clips have no English references.')
    save('manifest.json', manifest)
    return manifest


def evaluate(manifest):
    import gc
    import cv2
    import numpy as np
    import torch
    import transformers
    from transformers import MT5Config, MT5ForConditionalGeneration
    from rtmlib import Wholebody
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    text_dir = DATA / 'mt5-base'
    for name in ['config.json', 'generation_config.json', 'special_tokens_map.json',
                 'spiece.model', 'tokenizer_config.json']:
        download(f'https://huggingface.co/google/mt5-base/resolve/{TEXT_REV}/{name}', text_dir / name)
    checkpoint = download(f'https://huggingface.co/ZechengLi19/Uni-Sign/resolve/{MODEL_REV}/how2sign_pose_only_slt.pth',
                          DATA / 'how2sign_pose_only_slt.pth', WEIGHT_SHA)
    import config
    config.mt5_path = str(text_dir)
    from models import Uni_Sign
    args = SimpleNamespace(hidden_dim=256, dataset='How2Sign', rgb_support=False,
                           max_length=256, label_smoothing=.2)
    text_config = MT5Config.from_pretrained(str(text_dir), local_files_only=True)
    # Full Uni-Sign state includes mT5. Avoid downloading unused base weights.
    with patch.object(MT5ForConditionalGeneration, 'from_pretrained',
                      side_effect=lambda *a, **kw: MT5ForConditionalGeneration(text_config)):
        model = Uni_Sign(args)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
    loaded = model.load_state_dict(state, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    del state
    gc.collect()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.eval().to(device=device, dtype=torch.float32)
    os.environ['TORCH_HOME'] = str(DATA / 'rtmlib_cache')
    extractor = Wholebody(to_openpose=False, mode='lightweight', backend='onnxruntime', device='cpu')
    cls, _ = native_dataset()
    native_hashes = {str(p.relative_to(SOURCE)):sha256(p) for p in SOURCE.rglob('*.py')
                     if '.git' not in p.parts}
    provenance = dict(code_revision=CODE_REV, checkpoint_revision=MODEL_REV,
        checkpoint_sha256=sha256(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
        parameter_count=sum(p.numel() for p in model.parameters()), strict_load=True,
        tokenizer_revision=TEXT_REV, tokenizer_files={p.name:sha256(p) for p in text_dir.iterdir() if p.is_file()},
        dataset_revision=DATA_REV, native_source_sha256=native_hashes,
        extractor_files={p:sha256(Path(p)) for p in [extractor.det_model.onnx_model, extractor.pose_model.onnx_model]},
        worker_sha256=sha256(Path(__file__)), manifest_sha256=sha256(HERE/'manifest.json'),
        device=device, dtype='float32', torch=torch.__version__, transformers=transformers.__version__,
        preprocessing='Unchanged native lightweight Wholebody; all decoded frames; first detected person; x/W,y/H; native confidence/region normalization; seeded native 256-frame sampling',
        adaptations=['CPU ONNX instead of CUDA pose extraction', 'sequential frame extraction',
                     'MPS/CPU float32 instead of CUDA bfloat16', 'mT5 constructed from config then full strict checkpoint load',
                     'unchanged pose-only dataset definitions loaded without unused training/RGB imports'],
        beam_width=4, max_new_tokens=100, protected_test_accessed=False,
        training_performed=False)
    save('provenance.json', provenance)
    results = []
    for row in manifest['recordings']:
        print('Inference', row['item_id'], flush=True)
        begin = time.monotonic()
        cap = cv2.VideoCapture(row['video'])
        if not cap.isOpened():
            raise ValueError('Cannot open ' + row['video'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        keypoints, scores, persons = [], [], []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            kp, sc = extractor(frame)
            assert kp.shape[1:] == (133, 2) and sc.shape[1:] == (133,)
            assert len(kp) and np.isfinite(kp).all() and np.isfinite(sc).all()
            persons.append(len(kp))
            keypoints.append(kp[:1] / np.array([frame.shape[1], frame.shape[0]])[None,None])
            scores.append(sc[:1])
        cap.release()
        assert len(keypoints) > 0 and fps > 0
        pose_seconds = time.monotonic()-begin
        pose_path = HERE / 'poses' / (row['item_id'] + '.npz')
        pose_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(pose_path, keypoints=np.asarray(keypoints), scores=np.asarray(scores), fps=fps)
        ds = cls(args)
        ds.pose_data = dict(keypoints=keypoints, scores=scores)
        ds.rgb_data = row['video']
        random.seed(42)  # independent reproducible sampling for every utterance
        src, tgt = ds.collate_fn([ds[0]])
        assert tgt['gt_sentence'] == ['']
        for k, v in src.items():
            if isinstance(v, torch.Tensor):
                src[k] = v.to(device=device, dtype=torch.float32)
        if device == 'mps':
            torch.mps.synchronize()
        begin = time.monotonic()
        with torch.inference_mode():
            stack = model(src, tgt)
            tokens = model.generate(stack, max_new_tokens=100, num_beams=4)
        if device == 'mps':
            torch.mps.synchronize()
        text_seconds = time.monotonic()-begin
        prediction = model.mt5_tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        results.append(dict(**row, prediction=prediction, frame_count=len(keypoints), fps=fps,
                            duration_seconds=len(keypoints)/fps, pose_seconds=pose_seconds,
                            translation_seconds=text_seconds,
                            multi_person_frames=sum(n > 1 for n in persons),
                            mean_pose_confidence=float(np.asarray(scores).mean())))
        save('results.json', results)
    from sacrebleu.metrics import BLEU, CHRF
    paired = [r for r in results if r['reference'] is not None]
    predictions, refs = [r['prediction'] for r in paired], [[r['reference'] for r in paired]]
    bleu, chrf = BLEU(tokenize='13a'), CHRF()
    metrics = dict(paired_count=len(paired), diagnostic_count=len(results)-len(paired),
                   bleu=bleu.corpus_score(predictions, refs).score,
                   bleu_signature=str(bleu.get_signature()),
                   chrf=chrf.corpus_score(predictions, refs).score,
                   chrf_signature=str(chrf.get_signature()),
                   exact_match_count=sum(p.strip().lower()==r.strip().lower() for p,r in zip(predictions,refs[0])),
                   peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   median_pose_seconds=float(np.median([r['pose_seconds'] for r in results])),
                   median_translation_seconds=float(np.median([r['translation_seconds'] for r in results])))
    save('summary.json', metrics)
    lines = ['# Released Uni-Sign ASL baseline', '',
             'Completed pretrained inference with strict checkpoint loading. No training or live-system changes.', '',
             f"12 realigned How2Sign validation utterances: BLEU {metrics['bleu']:.2f}, chrF {metrics['chrf']:.2f}; exact text matches {metrics['exact_match_count']}/12.", '',
             'This is a small selected development slice, not a benchmark, a signer-disjoint claim, or evidence of improved webcam recognition. Automatic scores do not establish semantic correctness.', '',
             f"Checkpoint: {provenance['checkpoint_bytes']/1e9:.2f} GB; {provenance['parameter_count']:,} parameters. Native pose extraction on CPU; translation on {device} float32.",
             f"Median pose extraction {metrics['median_pose_seconds']:.2f}s and translation {metrics['median_translation_seconds']:.2f}s per completed clip. Desktop timing only; no iPhone measurements.", '',
             '## Paired validation outputs', '']
    for row in paired:
        lines += [f"### {row['item_id']}", '', f"Reference: {row['reference']}", '',
                  f"Prediction: {row['prediction']}", '', f"[Video]({row['video']})", '']
    lines += ['## Earlier difficult recordings', '',
              'These clips lack verified English translations. Gloss annotations are shown separately; no English accuracy is assigned. No count-correctness claim is made from fluent English.', '']
    previous = json.loads((HERE.parent/'stage2_held_sign_diagnostics_20260916/results.json').read_text())['results']
    previous = {r['item_id']: r['corrected_final'] for r in previous if r['phase'] == 0}
    for row in results[len(paired):]:
        lines += [f"### {row['item_id']}", '', f"Prediction: {row['prediction']}", '',
                  f"Previous CTC replay, phase zero: {previous[row['item_id']]}", '',
                  f"Existing gloss annotation: {row.get('gloss_annotation')}", '',
                  row.get('annotation_status',''), '', f"[Video]({row['video']})", '']
    lines += ['## Decision boundary', '',
              'Keep this as a separate challenger. Do not promote or start training automatically. Review source-grounded meaning and repetition in the displayed outputs; compare with the prior CTC replay. The current 100-gloss pipeline has no matching open-vocabulary English validation benchmark, so these scores are not a head-to-head improvement percentage.', '',
              'Native online-demo preprocessing was used; this is not a reproduction of the paper’s pre-extracted-pose benchmark. The released model’s pretraining and checkpoint-selection overlap beyond declared splits is not independently audited.', '',
              '[Official code](https://github.com/ZechengLi19/Uni-Sign), [released weights](https://huggingface.co/ZechengLi19/Uni-Sign), [validation mirror](https://huggingface.co/datasets/aipieces/How2Sign).']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')


def worker():
    started = datetime.now(timezone.utc).isoformat()
    status = 'failed'
    try:
        self_check()
        manifest = prepare()
        evaluate(manifest)
        status = 'completed'
    except Exception:
        (HERE/'FAILURE.md').write_text('# Uni-Sign baseline failed\n\nNo model promotion or training.\n\n```\n'+traceback.format_exc()+'```\n')
        traceback.print_exc()
    finally:
        save('completion.json', dict(status=status, started_at=started,
                                     finished_at=datetime.now(timezone.utc).isoformat()))
        log_path = ROOT/'docs/ground_truth/live-streaming/log.md'
        outcome = f"Detached Uni-Sign baseline {status}; see `artifacts/reports/{HERE.name}/" + ('REPORT.md' if status=='completed' else 'FAILURE.md') + '`.'
        if status == 'completed':
            metrics = json.loads((HERE/'summary.json').read_text())
            outcome += f" Measured 12 paired development utterances and nine diagnostics: BLEU {metrics['bleu']:.2f}, chrF {metrics['chrf']:.2f}."
        entry = '\n## ' + datetime.now().strftime('%Y-%m-%d') + ' — Uni-Sign worker exit\n\n' + outcome + '\nNo training or production promotion. Next action: inspect outputs and limitations before deciding whether this challenger helps.\n'
        content = log_path.read_text()
        log_path.write_text(content.replace('\n---\n', '\n---\n'+entry, 1))
        ground = ROOT/'PROJECT_GROUND_TRUTH.md'
        content = ground.read_text()
        for old in ['prepared for detached launch.', 'running; wait for its exit notification.']:
            content = content.replace('Uni-Sign baseline status: ' + old,
                                      f'Uni-Sign baseline status: {status}; inspect its exit report before further action.')
        if status == 'completed':
            content = content.replace('Native adapter self-check and compilation pass; model quality is not yet measured.',
                                      f"Native adapter checks and inference completed: development BLEU {metrics['bleu']:.2f}, chrF {metrics['chrf']:.2f}. Inspect individual outputs before drawing conclusions.")
        ground.write_text(content)
        subprocess.run([str(ROOT/'venv/bin/python'), str(ROOT/'scripts/index_large_artifacts_v17.py')],
                       cwd=ROOT, check=False)
        note = f'Uni-Sign comparison {status}. Open {HERE.name}/' + ('REPORT.md' if status=='completed' else 'FAILURE.md')
        subprocess.run(['osascript','-e', 'on run argv\ndisplay notification (item 1 of argv) with title "SLT evaluation"\nend run', note],
                       capture_output=True, timeout=15, check=False)
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.launch:
        # Exclusive launch record prevents accidental duplicate costly runs.
        with (HERE/'launch.json').open('x') as record, (HERE/'process.log').open('a') as log:
            child = subprocess.Popen(['caffeinate', '-i', sys.executable, '-u', __file__], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True)
            json.dump(dict(pid=child.pid, launched_at=datetime.now(timezone.utc).isoformat()), record)
        ground = ROOT/'PROJECT_GROUND_TRUTH.md'
        ground.write_text(ground.read_text().replace('Uni-Sign baseline status: prepared for detached launch.',
                                                    'Uni-Sign baseline status: running; wait for its exit notification.'))
        print('Launched', child.pid, 'with an exit notification; no polling.')
    else:
        sys.exit(worker())
