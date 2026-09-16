"""Bounded pinned-model diagnostics. No training, test data, or status polling.

Run --launch to detach. Completion/failure writes reports and a macOS notification.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))


def save(name, value):
    path = HERE / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare():
    """Freeze selected existing examples, preserving their annotation limitations."""
    import numpy as np
    import cv2
    from scripts.replay_revisable_transcription_v17 import recording_times, validate_recording
    source = ROOT / 'artifacts/reports/stage2_research_review_20260915'
    evidence = json.loads((source / 'evidence.json').read_text())
    history = json.loads((ROOT / evidence['session_history']).read_text())
    cards = list(evidence['cards'])
    # An unsegmented long diagnostic exercises real context rollover. No WER.
    timestamps = np.asarray(history['video_source_timestamps_seconds'])
    first, last = np.searchsorted(timestamps, [80., 100.]).tolist()
    long_video = HERE / 'webcam_long_context.mp4'
    if not long_video.exists():
        subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', history['video'], '-an',
            '-vf', f'trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS',
            '-c:v', 'libx264', '-crf', '21', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(long_video)], check=True)
    cards.append(dict(id='webcam_long_context', media=str(long_video),
        media_sha256=sha(long_video), frame_source_times=timestamps[first:last].tolist(),
        reference=None, annotation_status='Unsegmented 80–100s diagnostic; no intended transcript or original reset replay.'))
    rows = []
    for card in cards:
        video = source / card['media']
        assert sha(video) == card['media_sha256']
        row = dict(item_id=card['id'], video=str(video), video_sha256=sha(video),
                   role='validation' if 'saved_evaluation' in card and 'final' in card['saved_evaluation'] else 'diagnostic',
                   reference=card.get('saved_evaluation', {}).get('reference'),
                   annotation_status=card['annotation_status'], annotations=card.get('intervals', []))
        if 'frame_source_times' in card:
            target = HERE / (card['id'] + '_timestamps.json')
            target.write_text(json.dumps(card['frame_source_times']) + '\n')
            row['frame_timestamps'] = str(target)
        else:
            row['source_time_offset'] = card.get('offset', 0.)
        assert row['role'] == 'validation' or row['reference'] is None
        validate_recording(row)
        capture = cv2.VideoCapture(str(video))
        try:
            assert capture.isOpened(), str(video)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            recording_times(card.get('frame_source_times'), frame_count,
                            capture.get(cv2.CAP_PROP_FPS))
            assert capture.read()[0], str(video)
        finally:
            capture.release()
        rows.append(row)
    value = dict(recordings=rows, model=evidence['live_models']['general_selector'],
                 capture_settings={k: history['config'][k] for k in
                                   ('processing_fps', 'detection_image_side', 'maximum_image_side')},
                 phases=[0., .25, .5, .75], protected_test_accessed=False,
                 limitation='Re-extracted saved/re-encoded video, not original camera features. No expert-labelled hold/repeat set exists yet.')
    target = HERE / 'manifest.json'
    if target.exists() and json.loads(target.read_text()) != value:
        raise ValueError('Frozen inputs changed; use a new report directory')
    save('manifest.json', value)
    return value


def labels(tokens, names):
    return [names[t - 1] for t in tokens if 0 < t <= 100]


def run(frozen):
    import numpy as np
    import torch
    from active.v17.train_stage_2_other_ctc_v17 import _edit_operations, directory_sha256
    from scripts.diagnose_stage1_window_v17 import recording_observations
    from scripts.replay_revisable_transcription_v17 import phase_windows, validate_recording
    from scripts.live_reel_stage1_v17 import parser, stitch_revisable_ctc_logits
    from scripts.live_stage2_ctc_v17 import LiveStage2CTC, collapse_ctc_path, supported_ctc_path, roll_ctc_prefix

    torch.set_num_threads(2)
    args = parser().parse_args(['--no-display', '--no-speech', '--naturalizer', 'literal',
                               '--stage2-arbiter', '--revisable-transcript'])
    for key, value in frozen['capture_settings'].items():
        setattr(args, key, value)
    assert args.stage2_other_preservation is None and args.stage2_live_checkpoint is None
    assert str(args.stage2_selector) == frozen['model']['path']
    assert sha(args.stage2_selector) == frozen['model']['sha256']
    model = LiveStage2CTC(args)
    provenance = model.provenance()
    for value in provenance.values():
        path = Path(value['path'])
        if path.is_dir():
            value['sha256'] = directory_sha256(path)
    code = ['scripts/live_stage2_ctc_v17.py', 'scripts/live_reel_stage1_v17.py',
            'scripts/diagnose_stage1_window_v17.py', 'scripts/replay_revisable_transcription_v17.py']
    save('provenance.json', dict(models=provenance, code_sha256={p:sha(ROOT/p) for p in code},
        manifest_sha256=sha(HERE/'manifest.json'), worker_sha256=sha(__file__),
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        protected_test_accessed=False))
    results = []
    started = time.monotonic()
    for row in frozen['recordings']:
        validate_recording(row)
        observations, recording = recording_observations(row, args)
        if not observations:
            raise ValueError('No observations for ' + row['item_id'])
        offset = row.get('source_time_offset', 0.)
        if offset:
            for item in observations:
                item.seconds += offset
        times = [item.seconds for item in observations]
        for phase in frozen['phases']:
            target = HERE / row['item_id'] / f'phase_{phase:.2f}'
            target.mkdir(parents=True, exist_ok=True)
            args.ctc_trace_dir = target
            prior, all_features, accepted_times, windows = [], [], [], []
            old_locked, old_hyp, old_positions = [], [], []
            new_locked, new_hyp, new_positions = [], [], []
            previous_token, raw_path, step_offset = 0, [], 0
            for indices in phase_windows(times, args.stage2_window_seconds,
                                          times[0] + phase * args.stage2_window_seconds):
                selected = [observations[i] for i in indices]
                if len(selected) < 4:
                    windows.append(dict(accepted=False, reason='fewer_than_four_observations',
                        start_seconds=selected[0].seconds, end_seconds=selected[-1].seconds))
                    continue
                if len(prior) == 8:
                    previous_token = raw_path[7]
                    raw_path = raw_path[8:]
                    step_offset += 8
                    old_locked, old_hyp, old_positions = roll_ctc_prefix(old_locked, old_hyp, old_positions)
                    new_locked, new_hyp, new_positions = roll_ctc_prefix(new_locked, new_hyp, new_positions)
                    prior.pop(0)
                feature, result = model.classify_window(selected, prior, previous_token)
                item = dict(result, context_start_step=step_offset)
                if feature is not None:
                    prior.append(feature)
                    all_features.append(feature)
                    accepted_times.append([selected[0].seconds, selected[-1].seconds])
                    raw_path = list(result['ctc_argmax'])
                    with np.load(result['emissions_path'], allow_pickle=False) as archive:
                        logits = archive['logits']
                    tokens, positions = collapse_ctc_path(logits, len(logits))
                    tokens, positions = supported_ctc_path(tokens, positions)
                    old_hyp, old_positions = labels(tokens, model.labels), list(positions)
                    new_hyp, new_positions = result['hypothesis'], result['token_positions']
                    item.update(legacy_hypothesis=old_locked+old_hyp,
                                corrected_hypothesis=new_locked+new_hyp,
                                context_window_source_bounds=accepted_times[-8:],
                                context_source_bounds_are_not_exact_ctc_alignments=True)
                windows.append(item)
            chunks = []
            for start in range(0, len(all_features), 6):
                features = all_features[start:start+8]
                logits, _ = model.decode_frozen_logits(features)
                chunks.append((start, logits))
                if start+len(features) == len(all_features):
                    break
            stitched = stitch_revisable_ctc_logits(chunks, len(all_features)*8)
            tokens, positions = collapse_ctc_path(stitched, len(stitched)) if len(stitched) else ((), ())
            whole = labels(tokens, model.labels)
            np.savez_compressed(target/'offline_emissions.npz', logits=stitched,
                                accepted_window_source_bounds=np.asarray(accepted_times))
            result = dict(item_id=row['item_id'], phase=phase, role=row['role'],
                reference=row['reference'], recording=recording, windows=windows,
                legacy_final=old_locked+old_hyp, corrected_final=new_locked+new_hyp,
                offline_final=whole, offline_mode='single_context' if len(all_features)<=8 else 'overlap_stitched',
                accepted_windows=len(all_features), emitted_ctc_steps=len(stitched),
                replay_has_runtime_pacing=False, original_live_features_reproduced=False)
            if row['reference'] is not None:
                result['edits'] = {name:dict(Counter(o['operation'] for o in _edit_operations(row['reference'],result[name])))
                                   for name in ('legacy_final','corrected_final','offline_final')}
            results.append(result)
            save('results.json', dict(results=results, protected_test_accessed=False))
            print(row['item_id'], phase, result['corrected_final'], flush=True)
    assert len(results) == len(frozen['recordings'])*len(frozen['phases'])
    changed = [r for r in results if r['legacy_final'] != r['corrected_final']]
    offline_diff = [r for r in results if r['corrected_final'] != r['offline_final']]
    summary = dict(recordings=len(frozen['recordings']), phase_runs=len(results),
        rollover_changed_final_runs=len(changed), live_offline_different_runs=len(offline_diff),
        elapsed_seconds=time.monotonic()-started, training_performed=False,
        protected_test_accessed=False, independent_hold_repeat_accuracy=None,
        limitation=frozen['limitation'])
    save('summary.json', summary)
    lines = ['# Held-sign replay report', '',
        f'Completed {len(results)} phase runs over {len(frozen["recordings"])} pinned recordings.', '',
        f'Boundary-state correction changed {len(changed)} final outputs; corrected live and offline outputs differ in {len(offline_diff)} runs.', '',
        'The same model logits feed both legacy and corrected collapse. Differences between those two isolate bookkeeping; offline comparison can additionally change visual context.', '',
        '**Limits:** Saved videos are re-extracted. Original camera features cannot be reconstructed exactly. Webcam/partial O5S5 references are not scored as ground truth. Phases are sensitivity probes, not independent samples. No training or model promotion occurred.', '',
        '| Recording | Phase | Legacy live | Corrected live | Offline |',
        '|---|---:|---|---|---|']
    for r in results:
        values = [' '.join(r[k]) or '∅' for k in ('legacy_final','corrected_final','offline_final')]
        lines.append(f'| {r["item_id"]} | {r["phase"]} | '+ ' | '.join(values)+' |')
    lines += ['', '## Interpretation and next gate', '',
        'A repeated sign within one short context cannot be fixed by rollover bookkeeping. Inspect its saved logits and window origins before selecting a training intervention. A changed output under a phase shift demonstrates sensitivity, not which phase is linguistically correct.', '',
        'The rollover regression is covered by focused tests including blank/OTHER-separated repeats and the actual Reel event loop. This does not establish long-hold accuracy on real unseen signers.', '',
        'Missing: expert-labelled normal/held/twice performances across independent signers, rest and OTHER coverage. Existing webcam clips remain development diagnostics. Do not train on these evaluations or invent their intended labels.', '',
        'Training is deferred pending a justified objective and separate training examples. Uni-Sign remains a separately scoped gloss-free comparison; no external model or dataset was downloaded by this replay.', '',
        'Artifacts: manifest.json, provenance.json, results.json, summary.json, and per-phase compressed emission archives. Each accepted-window archive contains logits, its frozen visual features, exact observed source timestamps, and the carried boundary token. CTC steps are model emission positions, not exact sign boundary timestamps.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    subprocess.run([str(ROOT/'venv/bin/python'), 'scripts/index_large_artifacts_v17.py'], cwd=ROOT, check=True)
    return summary


def completed_job(frozen):
    started = datetime.now(timezone.utc).isoformat()
    def interrupted(number, _frame):
        raise RuntimeError(f'Replay interrupted by signal {number}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        summary = run(frozen)
        status = dict(status='completed', started_utc=started, summary=summary,
                      report=str(HERE/'REPORT.md'))
    except BaseException as error:
        failure = traceback.format_exc()
        (HERE/'FAILURE.md').write_text('# Replay failed\n\n```text\n'+failure+'```\n')
        status = dict(status='failed', started_utc=started, error=repr(error),
                      report=str(HERE/'FAILURE.md'))
        print(failure, file=sys.stderr, flush=True)
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save('completion.json', status)
    # One exit notification, no monitoring loop or assistant polling.
    notice = f'Stage 2 replay {status["status"]}. Report: {status["report"]}'
    try:
        process = subprocess.run(['osascript', '-e',
            'on run argv\ndisplay notification (item 1 of argv) with title "SLT diagnostics"\nend run',
            notice], capture_output=True, text=True, timeout=15)
        status['notification_exit_code'] = process.returncode
        status['notification_error'] = process.stderr.strip()
    except Exception as error:
        status['notification_error'] = repr(error)
    save('completion.json', status)
    return 0 if status['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        frozen = prepare()
        print(json.dumps(dict(recordings=len(frozen['recordings']),
                              phases=len(frozen['phases']), hashes_and_timestamps='passed')))
        return 0
    if args.launch:
        if (HERE/'launch.json').exists():
            raise ValueError('Already launched; inspect its completion report before starting another job')
        frozen = prepare()
        with (HERE/'process.log').open('xb') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve())],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        launch = dict(pid=process.pid, launched_utc=datetime.now(timezone.utc).isoformat(),
                      report_directory=str(HERE), recordings=len(frozen['recordings']),
                      phase_runs=len(frozen['recordings'])*len(frozen['phases']),
                      job='diagnostic inference; no training', notification='one macOS notification on exit',
                      automatically_resumes_chat=False)
        save('launch.json', launch)
        print(json.dumps(launch))
        return 0
    frozen = json.loads((HERE/'manifest.json').read_text())
    return completed_job(frozen)


if __name__ == '__main__':
    raise SystemExit(main())
