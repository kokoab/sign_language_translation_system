"""Live Reel on the streaming segmental pipeline (Apple Vision boundary student + span recognizer).

Default live mode of the app shell (`scripts/app_shell_v17.py`; the older Reel remains behind
`--classic-reel`). Direct run on a file: `venv/bin/python scripts/live_segmental_v17.py --video
clip.mp4 --no-display --no-speech`.

Screen: Apple Vision skeleton, the in-progress guess as amber "WORD?" with top-3 chips, committed
words in white on the rail, and the English sentence after Finish. Voice: each committed word is
spoken as it is shown; Finish (button, F, Space, or both open palms held up for one second)
flushes the last sign, renders the English
sentence with the configured Stage 3 naturalizer and speaks it. R / Reset clears, Q quits.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active.v17.segmental_runtime_v17 import CONFIG, FPS, build_runtime
from active.v17.approved_phrase_data_v17 import digest


def spelled_text(token):
    """fs-JOHN -> John for speech and English; letters are kept exactly as recognized."""
    word = token[3:] if token.startswith('fs-') else token
    return word[:1] + word[1:].lower()


def render_sentence(naturalizer, glosses, scores):
    """Stage 3 with spelled words passed as slots (FS0, FS1, ...) and restored afterwards.

    The renderer never sees the letters. If it does not produce every slot exactly once, the
    literal gloss sentence (with the spelled words) is used instead, so a name is never altered.
    """
    import re
    slots, inputs = {}, []
    for g in glosses:
        if g.startswith('fs-'):
            name = f'FS{len(slots)}'
            slots[name] = spelled_text(g)
            inputs.append(name)
        else:
            inputs.append(g)
    value = dict(naturalizer.rephrase(inputs, scores))
    value['spelled_slots'] = slots
    if not slots:
        return value
    sentence = str(value.get('sentence', ''))
    if all(len(re.findall(rf'\b{k}\b', sentence, re.I)) == 1 for k in slots):
        for k, word in slots.items():
            sentence = re.sub(rf'\b{k}\b', word, sentence, flags=re.I)
        value['sentence'] = sentence
        value['spelled_slot_mode'] = 'renderer'
    else:
        literal = ' '.join(slots.get(x, x.lower()) for x in inputs)
        value['renderer_sentence_with_slots'] = sentence
        value['sentence'] = literal[:1].upper() + literal[1:] + '.'
        value['spelled_slot_mode'] = 'literal_fallback'
    return value


def render_utterance(naturalizer, words):
    """Composition models punctuate the full utterance; older models retain pause splits."""
    from active.v17.segmental_runtime_v17 import split_clauses
    if not words:
        return dict(sentence='', clauses=[], clause_renderings=[], rendering_mode='empty')
    clauses = [words] if getattr(naturalizer, 'full_utterance', False) else split_clauses(words)
    parts = [render_sentence(naturalizer, [w['gloss'] for w in c], [w['score'] for w in c]) for c in clauses]
    sentence = ' '.join(str(p.get('sentence', '')).strip() for p in parts).strip()
    return dict(sentence=sentence, clauses=[[w['gloss'] for w in c] for c in clauses], clause_renderings=parts,
                rendering_mode='clauses' if len(parts) > 1 else parts[0].get('rendering_mode'))


class _Lock:
    """What ReelHud reads from a commit lock: no hit dots, nothing suppressed."""
    required_hits = 1
    hits = 0
    suppressed = None


class SegmentalRecognizer:
    """App-shell component: runtime, speaker and naturalizer, built off the camera path."""

    def __init__(self, args):
        from scripts.live_isolated_v17 import make_naturalizer, make_speaker
        self.args = args
        self.runtime = build_runtime(getattr(args, 'segmental_config', None) or CONFIG,
                                     device='mps' if args.device in ('auto', 'mps') else args.device,
                                     image_encoder=args.image_encoder,
                                     fingerspelling=not getattr(args, 'no_fingerspelling', False))
        self.speaker = None if args.no_speech else make_speaker(args)
        stage3 = self.runtime.config.get('stage3_checkpoint')
        if stage3:
            # The segmental Reel renders spelled words through slot tokens (FS0, FS1, ...).
            args.stage3_checkpoint = ROOT / stage3
        stage3_coreml = self.runtime.config.get('stage3_coreml')
        if (args.naturalizer == 'tiny' and self.runtime.recognizer.device == 'coreml' and stage3_coreml
                and not getattr(args, 'stage3_torch', False)):
            # Core ML pipeline: the same T5 graphs the phone runs (no transformers / PyTorch model).
            from active.v17.stage3_coreml_v17 import CoreMLStage3Naturalizer
            self.naturalizer = CoreMLStage3Naturalizer(ROOT / stage3_coreml)
        else:
            self.naturalizer = make_naturalizer(args)
        self.naturalizer_warmup = self.naturalizer.warm()
        config = self.runtime.config
        self.metadata = dict(mode='segmental_stream_v17', promoted_as_app_default=True, config=str(CONFIG),
                             config_sha256=digest(CONFIG),
                             boundary=config['boundary'], boundary_sha256=digest(ROOT / config['boundary']),
                             recognizer=config['recognizer'], recognizer_sha256=digest(ROOT / config['recognizer']),
                             lookahead_frames=self.runtime.lookahead, decoder=config['decoder'], stream=config['stream'],
                             device=self.runtime.recognizer.device, naturalizer=args.naturalizer,
                             stage3_backend=type(self.naturalizer).__name__,
                             stage3_checkpoint=str(args.stage3_checkpoint), letters=self.runtime.letters,
                             speech=self.speaker is not None)

    def provenance(self):
        return self.metadata


def run(args, shell=None, model=None):
    from scripts.app_shell_v17 import NullShell
    from scripts.live_isolated_v17 import SessionRecorder, orient_frame
    from scripts.live_stage2_ctc_v17 import LatestCamera, observe_stage2_frame
    from scripts.live_reel_stage1_v17 import FinishGesture, WINDOW_NAME, draw_reel_detection, persistent_auxiliary_detection
    import copy
    from scripts.reel_hud_v17 import ReelHud
    from active.v17.extract_v17 import AppleVisionDetector
    from active.v17.segmental_runtime_v17 import SpellingBuffer
    model = model or SegmentalRecognizer(args)
    runtime, speaker, naturalizer = model.runtime, model.speaker, model.naturalizer
    # The moving-letter (J/Z) merge is paused until validated; the config switch keeps it off.
    merge = runtime.config['stream'].get('moving_merge', False)
    speller = SpellingBuffer(rescore=getattr(runtime, 'score_letters', None) if merge else None)
    runtime.reset()
    shell = shell or NullShell()
    capture = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError('cannot open video/camera')
    recorder = SessionRecorder(args, model, str(args.video or args.camera))
    recorder.data['prototype_scope'] = 'segmental streaming Reel (Apple Vision only)'
    recorder.data['segmental_words'] = []
    recorder.add_event({'type': 'naturalizer_warmup', **(model.naturalizer_warmup or {})})
    detector = AppleVisionDetector(args.minimum_point_confidence)
    started = time.perf_counter()
    camera = None if args.video else LatestCamera(capture, started)
    fps_source = capture.get(cv2.CAP_PROP_FPS)
    fps_source = fps_source if np.isfinite(fps_source) and fps_source > 1 else 30.
    language = ThreadPoolExecutor(max_workers=1)
    hud = ReelHud(provisional=True)
    lock = _Lock()
    finish_gesture = FinishGesture(args.finish_gesture_hold_seconds, dropout_grace_seconds=0.)
    actions = deque()
    glosses, scores, finishing = [], [], []
    timed = []                   # committed words with timing, for clause splitting at Finish
    sentence = ''
    naturalizing = None          # (future, context)
    latest = None                # last Apple Vision observation (skeleton + quality stats)
    display_latest = None        # same, with the last body/face kept visible between detections
    visible_body = visible_face = None
    latest_result = None         # what the HUD shows: preview or the word just committed
    last_preview_key, last_preview_seconds = None, -np.inf
    frame_index = processed = sequence = camera_sequence = dropped = 0
    deadline = seconds = 0.
    wrists = {'left': None, 'right': None}
    paused_before = False
    compute, display_times = [], deque(maxlen=30)
    display_size = [1280, 720]
    last_button = {'reset': -np.inf, 'finish': -np.inf}
    if not args.no_display:
        def on_control(action, stamp):
            if stamp - last_button.get(action, -np.inf) >= .25:
                last_button[action] = stamp
                actions.append(action)
        shell.attach(WINDOW_NAME, display_size, started, on_control)

    def speak(text, kind, reference):
        if speaker is None:
            return
        speaker.enqueue(str(text), kind, reference)
        recorder.add_event({'type': 'speech_queued', 'text': text, 'kind': kind, 'reference': reference})

    def accept(words, spent):
        shown = []
        for w in words:
            shown += speller.push(w)
        emit(shown, spent)

    def emit(words, spent):
        nonlocal latest_result
        for w in words:
            w['compute_seconds'] = spent
            w['shown_elapsed_seconds'] = time.perf_counter() - started
            recorder.data['segmental_words'].append(w)
            glosses.append(w['gloss'])
            scores.append(w['score'])
            timed.append(w)
            speak(spelled_text(w['gloss']) if w['gloss'].startswith('fs-') else w['gloss'], 'gloss',
                  len(recorder.data['segmental_words']) - 1)
            latest_result = dict(gloss=w['gloss'], candidate_gloss=w['gloss'], committed_gloss=w['gloss'],
                                 accepted=True, model_score=w['score'], full_verifier={'commit_score': w['score']},
                                 top3=(latest_result or {}).get('top3') or [], latency_ms={'total': 1000 * spent})

    def show_preview(spent):
        """Amber in-progress guess; cleared once its word is committed or when nothing is open."""
        nonlocal latest_result
        p = runtime.preview
        if speller.letters:        # letters being spelled show as an amber, still-open word
            latest_result = dict(gloss='fs-' + speller.pending() + '…', candidate_gloss=None, accepted=True,
                                 committed_gloss=None, model_score=1.0, top3=[], latency_ms={'total': 1000 * spent})
            return
        if latest_result and latest_result.get('committed_gloss') and not latest_result.get('_aged'):
            latest_result['_aged'] = True        # keep a fresh commit on screen for one frame
            return
        if p is None:
            latest_result = dict(accepted=False, gloss='UNKNOWN', committed_gloss=None, top3=[],
                                 model_score=0., latency_ms={'total': 1000 * spent})
            return
        label = 'fs-' + p['gloss'][3:] if p['gloss'].startswith('FS_') else p['gloss']
        latest_result = dict(gloss=label if p['emit'] else 'UNKNOWN', candidate_gloss=label,
                             accepted=bool(p['emit']), committed_gloss=None, model_score=p['score'],
                             top3=p['top3'], latency_ms={'total': 1000 * spent})

    def finish():
        nonlocal naturalizing, sentence, finishing
        if naturalizing is not None:
            return
        accept(runtime.finish(), 0.)
        emit(speller.flush(), 0.)
        runtime.reset()
        words, word_scores, word_times = list(glosses), list(scores), list(timed)
        glosses.clear(); scores.clear(); timed.clear()
        if not words:
            sentence = 'No recognized signs to finish.'
            recorder.add_event({'type': 'finish_empty', 'seconds': seconds})
            return
        finishing = words
        context = dict(utterance_id=f'segmental-{len(recorder.data["utterances"]) + 1:04d}', glosses=words,
                       gloss_scores=word_scores, finish_started_at=time.perf_counter())
        naturalizing = (language.submit(render_utterance, naturalizer, word_times), context)

    def service_language():
        nonlocal naturalizing, sentence, finishing
        if naturalizing is None or not naturalizing[0].done():
            return
        future, context = naturalizing
        naturalizing = None
        value = future.result()
        started_at = context.pop('finish_started_at')
        utterance = {**context, **value, 'displayed_and_spoken': True,
                     'finish_total_ms': 1000 * (time.perf_counter() - started_at)}
        recorder.add_utterance(utterance)
        sentence = str(value.get('sentence', ' '.join(context['glosses']).lower()))
        finishing = []
        speak(sentence, 'finished_sentence', utterance['utterance_id'])

    def reset_all(source):
        nonlocal sentence, finishing, naturalizing, latest_result, wrists, visible_body, visible_face
        nonlocal finish_gesture
        finish_gesture = FinishGesture(args.finish_gesture_hold_seconds, dropout_grace_seconds=0.)
        runtime.reset()
        speller.flush()
        visible_body = visible_face = None
        glosses.clear(); scores.clear(); timed.clear(); finishing = []
        sentence = ''
        naturalizing = None
        latest_result = None
        wrists = {'left': None, 'right': None}
        if speaker:
            speaker.clear()
        recorder.add_event({'type': 'segmental_reset', 'source': source, 'seconds': seconds})

    try:
        while True:
            if camera:
                packet = camera.after(sequence)
                if packet is None:
                    if camera.failed:
                        break
                    time.sleep(.001)
                    continue
                sequence, seconds, raw = packet
                dropped += max(0, sequence - camera_sequence - 1)
                camera_sequence = sequence
            else:
                ok, raw = capture.read()
                if not ok:
                    break
                seconds = frame_index / fps_source
                frame_index += 1
                if getattr(args, 'realtime_video', False):
                    time.sleep(max(0., started + seconds - time.perf_counter()))
            paused = not args.no_display and shell.paused
            if paused != paused_before:
                runtime.reset()           # another page came and went: start the stream fresh
                finish_gesture = FinishGesture(args.finish_gesture_hold_seconds, dropout_grace_seconds=0.)
                wrists = {'left': None, 'right': None}
                paused_before = paused
            while actions:
                action = actions.popleft()
                if action == 'finish':
                    finish()
                elif action == 'reset':
                    reset_all('button')
            service_language()
            if speaker:
                started_speech = speaker.update()
                if started_speech is not None:
                    recorder.add_event({'type': 'speech_started', **started_speech})
            if paused:
                key = shell.present(raw, glosses)
                if key in (ord('q'), 27):
                    break
                continue
            canonical = orient_frame(raw, args.rotation, args.input_mirrored)
            recorder.write_frame(canonical, seconds)
            if seconds + 1e-6 >= deadline:
                tick = time.perf_counter()
                latest = observe_stage2_frame(canonical, seconds, processed, detector, wrists, args)
                processed += 1
                # Body/face run every few frames by contract; keep the last ones on screen only.
                visible, visible_body, visible_face = persistent_auxiliary_detection(
                    latest.detection, visible_body, visible_face)
                display_latest = copy.copy(latest)
                display_latest.detection = visible
                deadline = max(deadline + 1 / FPS, seconds)
                allow_gesture = not args.no_finish_gesture and getattr(shell, 'allow_finish', True)
                triggered = finish_gesture.update(
                    latest.assigned if allow_gesture else {'left': None, 'right': None}, seconds,
                    latest.detection.body_xy, latest.detection.body_confidence)
                # The control pose must never enter the word/letter decoder. Keep its
                # pre-gesture tail available for Finish, or resume if the hold is cancelled.
                words = [] if finish_gesture.active else runtime.observe(latest)
                spent = time.perf_counter() - tick
                compute.append(spent)
                if words:
                    if sentence and not finishing:
                        sentence = ''       # a new utterance has started
                    accept(words, spent)
                else:
                    show_preview(spent)
                preview_key = (latest_result or {}).get('candidate_gloss') or (latest_result or {}).get('gloss')
                if preview_key != last_preview_key and seconds - last_preview_seconds >= .25:
                    recorder.add_event(dict(type='segmental_preview', seconds=seconds,
                                            gloss=preview_key, pending_spelling=speller.pending(),
                                            score=(latest_result or {}).get('model_score', 0.)))
                    last_preview_key, last_preview_seconds = preview_key, seconds
                emit(speller.tick(seconds, active_hands=any(h is not None for h in latest.assigned.values())), spent)
                if finish_gesture.active:
                    latest_result = None
                if triggered:
                    recorder.add_event({'type': 'finish_requested', 'source': 'two_hand_gesture', 'seconds': seconds})
                    finish()
            if not args.no_display:
                now = time.perf_counter()
                display_times.append(now)
                fps = (len(display_times) - 1) / (display_times[-1] - display_times[0]) if len(display_times) > 1 else 0.
                shown = draw_reel_detection(canonical, display_latest, mirror=not args.no_mirror_display)
                display_size[:] = [shown.shape[1], shown.shape[0]]
                active = runtime.preview is not None or (
                    latest is not None and any(latest.assigned.get(side) is not None for side in ('left', 'right')))
                shown = hud.draw(
                    shown, latest, latest_result, lock, False, active, fps,
                    glosses, finishing,
                    (f'Hold both open hands to finish  {round(100 * finish_gesture.progress)}%'
                     if finish_gesture.active and not finish_gesture.latched else sentence),
                    naturalizing is not None,
                    None if speaker is None else speaker.current_text,
                    None, {'dropped': dropped, 'observations': processed},
                    top_inset=shell.top_inset, minimal=getattr(shell, 'minimal_hud', False))
                key = shell.present(shown, glosses)
                if key in (ord('q'), 27):
                    break
                if key == ord('r'):
                    actions.append('reset')
                if key in (ord('f'), ord(' ')):
                    actions.append('finish')
        recorder.finish_video()
        finish()
        if naturalizing is not None:
            naturalizing[0].result()
            service_language()
        recorder.data['hypothesis'] = [w['gloss'] for w in recorder.data['segmental_words']]
        c = np.asarray(compute or [0.])
        recorder.data['capture_stats'] = dict(landmark_observations=processed, dropped_camera_frames=dropped,
                                              elapsed_seconds=time.perf_counter() - started,
                                              frame_compute_ms_median=1000 * float(np.median(c)),
                                              frame_compute_ms_p90=1000 * float(np.percentile(c, 90)))
    finally:
        recorder.finish_video()
        if camera:
            camera.close()
        else:
            capture.release()
        language.shutdown(wait=True, cancel_futures=True)
        if speaker:
            speaker.clear()
        recorder.close()
    return dict(hypothesis=recorder.data.get('hypothesis', []), sentence=sentence, history=str(recorder.history_path))


if __name__ == '__main__':
    from scripts.app_shell_v17 import parser
    arguments = parser().parse_args()
    print(json.dumps(run(arguments)))
