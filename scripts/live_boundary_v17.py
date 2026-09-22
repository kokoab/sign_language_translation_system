"""Opt-in ASL temporal boundary runtime; shares its inference path with evaluation."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active.v17.approved_phrase_data_v17 import digest
from active.v17.stage1_window_v17 import raw_observation_features
from active.v17.temporal_boundary_v17 import FORMAT, FPS, BoundaryStream, TemporalBoundary


def validate_boundary_args(args):
    expected = dict(processing_fps=20., detection_image_side=640, maximum_image_side=1280,
                    minimum_point_confidence=.15, dense_model_auxiliary=False, commit_hits=1)
    for key, value in expected.items():
        if getattr(args, key) != value:
            raise ValueError(f'boundary runtime requires {key}={value}')


def classify_interval(reel, observations, event, args, context=.1):
    """Exactly one decision per event, preserving Reel identity/score rules."""
    from scripts.live_reel_stage1_v17 import VerifiedCommitLock
    selected = [o for o in observations if event['start_seconds'] - context <= o.seconds <= event['end_seconds'] + context]
    result = {**event, 'committed_gloss': None, 'context_seconds': context}
    if len(selected) < 4:
        return {**result, 'reason': 'fewer_than_four_observations'}
    started = time.perf_counter()
    proposal, verifier = reel.classify(selected), reel.verify(selected)
    agreement = proposal['model_score'] if proposal.get('candidate_gloss') == verifier.get('candidate_gloss') else 0.
    commit = bool(proposal.get('accepted') and verifier.get('accepted')) and VerifiedCommitLock(args.commit_hits, args.instant_commit_score).update(
        str(verifier.get('candidate_gloss')), max(verifier['model_score'], agreement),
        proposal=str(proposal.get('candidate_gloss')), proposal_score=proposal['model_score'], minimum_score=args.commit_score)
    return {**result, 'proposal': proposal, 'verifier': verifier,
            'committed_gloss': verifier.get('candidate_gloss') if commit else None,
            'recognition_seconds': time.perf_counter() - started}


class BoundaryRecognizer:
    def __init__(self, args, reel=None):
        from scripts.live_reel_stage1_v17 import ReelCascadeClassifier
        validate_boundary_args(args)
        path = Path(args.boundary_checkpoint)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint.get('format') != FORMAT:
            raise ValueError('not an ASL temporal boundary checkpoint')
        recipe = ROOT / checkpoint['recipe_path']
        if digest(recipe) != checkpoint['recipe_sha256']:
            raise ValueError('boundary recipe hash mismatch')
        contract = json.loads(recipe.read_text())
        code = 'active/v17/temporal_boundary_v17.py'
        if contract['code_sha256'][code] != digest(ROOT / code):
            raise ValueError('boundary model/preprocessing code hash mismatch')
        self.args = copy.copy(args)
        self.args.no_motion_trim = True
        self.model = TemporalBoundary(**checkpoint['model_config'])
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        # Small per-frame TCN stays on CPU; MPS is used by the dedicated trainer.
        self.model.eval()
        self.stream = BoundaryStream(self.model, checkpoint['hand_geometry'])
        self.reel = reel or ReelCascadeClassifier(self.args)
        self.metadata = dict(boundary_checkpoint=str(path), sha256=digest(path),
                             recipe_sha256=checkpoint['recipe_sha256'],
                             future_context_frames=self.model.lookahead,
                             boundary_device='cpu', promoted=False, reel=self.reel.provenance())
        self.observations = deque()

    def provenance(self):
        return self.metadata

    def reset(self):
        self.stream.reset()
        self.observations.clear()

    def observe(self, observation):
        if self.observations and observation.seconds - self.observations[-1].seconds > .26:
            self.reset()
        self.observations.append(observation)
        while self.observations and self.observations[0].seconds < observation.seconds - 6:
            self.observations.popleft()
        raw, times = raw_observation_features([observation])
        return self.stream.update(raw[0], times[0])

    def classify(self, observations, event):
        return classify_interval(self.reel, observations, event, self.args)


def run(args, shell=None, model=None):
    from scripts.app_shell_v17 import NullShell
    from scripts.live_isolated_v17 import SessionRecorder, orient_frame, make_speaker
    from scripts.live_stage2_ctc_v17 import LatestCamera, observe_stage2_frame
    from active.v17.extract_v17 import AppleVisionDetector
    model = model or BoundaryRecognizer(args)
    shell = shell or NullShell()
    capture = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError('cannot open video/camera')
    recorder = SessionRecorder(args, model, str(args.video or args.camera))
    recorder.data['prototype_scope'] = 'ASL temporal boundary candidate; literal stable events; not promoted'
    recorder.data['boundary_trace'] = []
    detector = AppleVisionDetector(args.minimum_point_confidence)
    started = time.perf_counter()
    camera = None if args.video else LatestCamera(capture, started)
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 1 else 30.
    executor = ThreadPoolExecutor(max_workers=1)
    speaker = None if args.no_speech else make_speaker(args)
    future = None
    pending, actions, glosses = deque(), deque(), []
    generation = job_generation = frame_index = processed = sequence = 0
    deadline = 0.
    wrists = {'left': None, 'right': None}
    paused_before = False
    seconds = 0.
    status = 'Watching hands and fingers'
    finished_count = 0
    if not args.no_display:
        shell.attach('Sign Language Translation', [1280, 720], started, lambda action, stamp: actions.append(action))

    def collect(wait=False):
        nonlocal future, status
        if future is None or (not wait and not future.done()):
            return
        result = future.result(); future = None
        result['ignored_after_reset'] = job_generation != generation
        if result['ignored_after_reset']:
            result['ignored_committed_gloss'] = result['committed_gloss']
            result['committed_gloss'] = None
        result['completed_elapsed_seconds'] = time.perf_counter() - started
        result['completed_observed_seconds'] = seconds
        if not result['ignored_after_reset'] and result['committed_gloss']:
            glosses.append(result['committed_gloss'])
        recorder.add(result)
        status = ' '.join(glosses[-8:]) or 'Watching hands and fingers'

    def submit():
        nonlocal future, job_generation
        if future is None and pending:
            event, obs, event_generation = pending.popleft()
            job_generation = event_generation
            future = executor.submit(model.classify, obs, event)

    try:
        while True:
            if camera:
                packet = camera.after(sequence)
                if packet is None:
                    if camera.failed: break
                    time.sleep(.001); continue
                sequence, seconds, raw = packet
            else:
                ok, raw = capture.read()
                if not ok: break
                seconds = frame_index / fps; frame_index += 1
                if getattr(args, 'realtime_video', False):
                    time.sleep(max(0., started + seconds - time.perf_counter()))
            paused = not args.no_display and shell.paused
            if paused != paused_before:
                actions.append('page_change'); paused_before = paused
            while actions:
                action = actions.popleft()
                if action == 'finish':
                    # Finish drains already completed sign events, never invents an
                    # end for the currently unfinished interval.
                    while future is not None or pending:
                        collect(wait=True); submit()
                    words = glosses[finished_count:]
                    if words:
                        utterance = dict(glosses=list(words), sentence=' '.join(words).lower(),
                                         displayed_and_spoken=True, renderer='literal', utterance_id=f'boundary-{finished_count}')
                        recorder.add_utterance(utterance)
                        if speaker: speaker.enqueue(utterance['sentence'], 'finished_sentence', utterance['utterance_id'])
                        finished_count = len(glosses)
                generation += 1; model.reset(); pending.clear()
                wrists = {'left': None, 'right': None}
                if action == 'reset':
                    glosses.clear(); finished_count = 0
                    if speaker: speaker.clear()
                recorder.add_event(dict(type='boundary_reset', action=action, seconds=seconds))
            collect(); submit()
            if speaker: speaker.update()
            canonical = orient_frame(raw, args.rotation, args.input_mirrored)
            if not paused:
                recorder.write_frame(canonical, seconds)
                if seconds + 1e-6 >= deadline:
                    observation = observe_stage2_frame(canonical, seconds, processed, detector, wrists, args)
                    processed += 1
                    deadline = max(deadline + 1 / FPS, seconds)
                    tick = time.perf_counter()
                    result = model.observe(observation)
                    if result is not None:
                        recorder.data['boundary_trace'].append({**result, 'compute_seconds': time.perf_counter() - tick})
                        for event in result['events']:
                            pending.append(({**event, 'available_seconds': result['available_seconds']}, list(model.observations), generation))
                        # Bound verification backlog explicitly; never silently call a
                        # skipped candidate a successful rejection.
                        if len(pending) > 8:
                            dropped = pending.popleft()[0]
                            recorder.add_event(dict(type='boundary_verifier_backlog_drop', **dropped))
                    submit()
            if not args.no_display:
                shown = cv2.flip(canonical, 1) if not args.no_mirror_display else canonical.copy()
                for i, line in enumerate(['ASL BOUNDARY CANDIDATE', status, 'R reset / F finish / Q quit']):
                    cv2.putText(shown, line, (20, 95 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                key = shell.present(shown, glosses)
                if key in (ord('q'), 27): break
                if key == ord('r'): actions.append('reset')
                if key in (ord('f'), ord(' ')): actions.append('finish')
        recorder.finish_video()
        while future is not None or pending:
            collect(wait=True); submit()
        recorder.data['unscored_tail_seconds'] = model.model.lookahead / FPS
        recorder.data['hypothesis'] = glosses
        recorder.data['capture_stats'] = dict(landmark_observations=processed, elapsed_seconds=time.perf_counter() - started)
    finally:
        recorder.finish_video()
        if camera: camera.close()
        else: capture.release()
        executor.shutdown(wait=True, cancel_futures=True)
        if speaker: speaker.clear()
        recorder.close()
    return dict(hypothesis=glosses, history=str(recorder.history_path))


if __name__ == '__main__':
    from scripts.app_shell_v17 import parser
    arguments = parser().parse_args()
    if not arguments.boundary_checkpoint:
        raise SystemExit('--boundary-checkpoint is required')
    print(json.dumps(run(arguments)))
