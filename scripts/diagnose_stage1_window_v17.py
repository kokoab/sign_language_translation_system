"""Matched existing-model diagnostics; invoked by replay_revisable_transcription_v17."""
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from active.v17.extract_v17 import AppleVisionDetector, orient_frame
from active.v17.train_stage_2_other_ctc_v17 import directory_sha256, sha256
from scripts.live_reel_continuous_v17 import parser as live_parser
from scripts.live_reel_stage1_v17 import ReelCascadeClassifier
from scripts.live_stage2_ctc_v17 import LiveStage2CTC, observe_stage2_frame, collapse_ctc_path, supported_ctc_path
from scripts.replay_revisable_transcription_v17 import ROOT, recording_times, phase_windows


class CapturedPrediction:
    """Observe real model outputs without changing the existing inference path."""
    def __init__(self, model):
        self.model = model
        self.last = None

    def predict(self, provider):
        result = self.model.predict(provider)
        self.last = np.asarray(next(iter(result.values()))).reshape(-1).copy()
        return result

    def __getattr__(self, name):
        return getattr(self.model, name)


def class_evidence(logits, labels, target):
    if logits is None:
        return None
    values = logits[:len(labels)]
    probability = np.exp(values - values.max())
    probability /= probability.sum()
    index = labels.index(target)
    order = np.argsort(-probability)
    return dict(target=target, rank=int(np.flatnonzero(order == index)[0]) + 1,
                probability=float(probability[index]), logit=float(values[index]),
                top5=[dict(label=labels[i], probability=float(probability[i])) for i in order[:5]])


def recording_observations(row, args):
    capture = cv2.VideoCapture(str(ROOT / row['video']))
    if not capture.isOpened():
        raise ValueError('cannot open ' + row['video'])
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamps = json.loads((ROOT / row['frame_timestamps']).read_text()) if row.get('frame_timestamps') else None
    times = recording_times(timestamps, count, capture.get(cv2.CAP_PROP_FPS), row.get('exclude_final_frames', 0))
    detector = AppleVisionDetector(args.minimum_point_confidence)
    wrists = dict(left=None, right=None)
    processed, deadline = 0, 0.
    output = []
    try:
        for seconds in times:
            ok, frame = capture.read()
            if not ok:
                raise ValueError('unexpected video decode failure before excluded tail')
            if seconds + 1e-6 < deadline:
                continue
            item = observe_stage2_frame(orient_frame(frame, args.rotation, args.input_mirrored), seconds, processed, detector, wrists, args)
            processed += 1
            if row.get('discard_pixels'):
                item.frame = np.empty((*item.frame.shape[:2], 0), dtype=item.frame.dtype)
            deadline = max(deadline + 1 / args.processing_fps, seconds)
            for side, hand in item.assigned.items():
                if hand is not None and hand.confidence[0] > 0:
                    wrists[side] = hand.xy[0].copy()
            if row.get('diagnostic_start_seconds', 0) <= seconds <= row.get('diagnostic_end_seconds', float('inf')):
                output.append(item)
    finally:
        capture.release()
    return output, dict(decoded_frames=len(times), processed_frames=processed,
                        retained_frames=len(output), original_timestamp_count=None if timestamps is None else len(timestamps),
                        excluded_final_frames=row.get('exclude_final_frames', 0))


def diagnose(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    args = live_parser().parse_args(['--no-display', '--no-speech', '--naturalizer', 'literal'])
    paths = dict(stage1_start=ROOT/'artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth',
                 stage1_proposal=args.orientation_coreml, stage1_verifier=args.stage1_coreml,
                 verifier_checkpoint=args.unified_checkpoint, stage2=args.stage2_other_preservation,
                 selector=args.stage2_selector, primary=args.stage2_primary, specialist=args.stage2_specialist,
                 encoder=args.stage2_encoder, image_encoder=args.image_encoder)
    frozen = dict(checkpoints={key: dict(path=str(p), sha256=directory_sha256(p) if p.is_dir() else sha256(p)) for key, p in paths.items()},
                  recordings=[dict(**row, video_sha256=sha256(ROOT/row['video']),
                                   timestamps_sha256=sha256(ROOT/row['frame_timestamps']) if row.get('frame_timestamps') else None) for row in rows],
                  settings={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  shifts=[0., .25, .5, .75], protected_test_accessed=False)
    frozen = json.loads(json.dumps(frozen))
    freeze_path = output/'diagnostic_freeze.json'
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != frozen:
        raise ValueError('diagnostic inputs/settings changed; use a new output directory')
    freeze_path.write_text(json.dumps(frozen, indent=2)+'\n')
    proposal, ctc = ReelCascadeClassifier(args), LiveStage2CTC(args)
    proposal.orientation = CapturedPrediction(proposal.orientation)
    proposal.full.stage1 = CapturedPrediction(proposal.full.stage1)
    results = []
    for row in rows:
        observations, recording = recording_observations(row, args)
        target = row.get('diagnostic_gloss', 'HUNGRY')
        for phase in frozen['shifts']:
            prior, windows = [], []
            for indices in phase_windows([o.seconds for o in observations], args.stage2_window_seconds, phase * args.stage2_window_seconds):
                selected = [observations[i] for i in indices]
                item = dict(start=selected[0].seconds, end=selected[-1].seconds, frames=len(selected),
                            hand_frame_coverage=sum(any(h is not None for h in o.assigned.values()) for o in selected)/len(selected))
                proposal.orientation.last = proposal.full.stage1.last = None
                item['proposal'] = proposal.classify(selected)
                item['proposal_evidence'] = class_evidence(proposal.orientation.last, proposal.labels, target)
                if len(selected) < 4:
                    item['verifier'] = dict(accepted=False, rejection_reasons=['fewer_than_four_frames'])
                else:
                    item['verifier'] = proposal.verify(selected)
                item['verifier_evidence'] = class_evidence(proposal.full.stage1.last, proposal.labels, target)
                feature, result = ctc.classify_window(selected, prior[-7:])
                item['stage2'] = result
                if feature is not None:
                    prior.append(feature)
                    repaired, _ = ctc.decode_frozen_logits(prior[-8:])
                    preservation = ctc.other_preservation
                    ctc.other_preservation = None
                    try:
                        original, _ = ctc.decode_frozen_logits(prior[-8:])
                    finally:
                        ctc.other_preservation = preservation
                    for name, logits in [('before_other_suppression', original), ('after_other_suppression', repaired)]:
                        tokens, positions = collapse_ctc_path(logits, len(logits))
                        supported, _ = supported_ctc_path(tokens, positions)
                        item[name] = dict(tokens=list(tokens), labels=[ctc.labels[t-1] if t <= 100 else 'OTHER' for t in tokens],
                                          supported_labels=[ctc.labels[t-1] for t in supported], argmax_path=logits.argmax(-1).tolist())
                windows.append(item)
            results.append(dict(item_id=row['item_id'], phase=phase, recording=recording, windows=windows, limitation=row.get('limitation')))
            (output/'diagnostics.json').write_text(json.dumps(results, indent=2)+'\n')
            print(row['item_id'], 'phase', phase, 'windows', len(windows), flush=True)
