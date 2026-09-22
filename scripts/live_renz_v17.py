"""Experimental buffered Renz segmentation with the unchanged Reel classifier."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sign_regions(probabilities, start, owned_start, owned_end):
    """Class zero runs, upstream half-gap extension, unique midpoint ownership."""
    intervals, left = [], None
    for index, boundary in enumerate(list(np.asarray(probabilities) > .5) + [True]):
        if not boundary and left is None:
            left = index
        elif boundary and left is not None:
            intervals.append([left, index])
            left = None
    for index in range(len(intervals) - 1):
        half = (intervals[index + 1][0] - intervals[index][1]) // 2
        intervals[index][1] += half
        intervals[index + 1][0] -= half
    regions = [(start + a / 25, start + b / 25) for a, b in intervals]
    return [(a, b) for a, b in regions if owned_start < (a + b) / 2 <= owned_end]


def rgb_input(observations):
    times = np.array([o.seconds for o in observations])
    clock = np.arange(times[0], times[-1] + .00001, 1 / 25)
    indices = np.searchsorted(times, clock).clip(0, len(times) - 1)
    previous = np.maximum(indices - 1, 0)
    indices = np.where(abs(times[previous] - clock) < abs(times[indices] - clock), previous, indices)
    images = []
    for index in indices:
        image = observations[index].frame
        height, width = image.shape[:2]
        scale = 256 / max(height, width)
        resized = cv2.resize(image, (round(width * scale), round(height * scale)))
        square = np.zeros((256, 256, 3), np.uint8)
        y, x = (256 - resized.shape[0]) // 2, (256 - resized.shape[1]) // 2
        square[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        images.append(square[16:240, 16:240, ::-1].copy())
    return np.asarray(images)


class BufferedRenz:
    def __init__(self, args):
        from scripts.live_reel_stage1_v17 import ReelCascadeClassifier
        from active.v17.approved_phrase_data_v17 import digest
        vendor = ROOT / 'artifacts/vendor/renz_sign_segmentation/demo/models'
        weights = ROOT / 'artifacts/models/renz_pretrained_v17'
        self.args = args
        self.device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
        torch.set_num_threads(2)
        self.i3d = load_module('renz_live_i3d', vendor / 'i3d.py').InceptionI3d(num_classes=981, num_in_frames=16, include_embds=True)
        state = torch.load(weights / 'i3d_kinetics_bslcp.pth.tar', map_location='cpu', weights_only=True)['state_dict']
        self.i3d.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()})
        self.i3d.to(self.device).eval()
        if self.device == 'mps':
            # MPS lacks 3D pooling; retain published arithmetic on CPU, convolutions on MPS.
            for layer in self.i3d.modules():
                if isinstance(layer, (torch.nn.MaxPool3d, torch.nn.AvgPool3d)):
                    original = layer.forward
                    layer.forward = lambda x, operation=original: operation(x.cpu()).to(x.device)
        self.segmenter = load_module('renz_live_mstcn', vendor / 'mstcn.py').MultiStageModel(4, 10, 64, 1024, 2)
        self.segmenter.load_state_dict(torch.load(weights / 'mstcn_bslcp_i3d_bslcp.model', map_location='cpu', weights_only=True))
        self.segmenter.eval()  # Tiny CPU temporal network; avoid needless accelerator transfers.
        self.reel = ReelCascadeClassifier(args)
        self.models = {'renz': {p.name: digest(p) for p in weights.iterdir()}, 'reel': self.reel.provenance(), 'device': self.device, 'pooling_device': 'cpu'}

    def provenance(self):
        return self.models

    def boundaries(self, images):
        padded = np.pad(images, ((8, 7), (0, 0), (0, 0), (0, 0)), mode='edge')
        features = []
        with torch.inference_mode():
            for start in range(0, len(images), 4):
                batch = np.stack([padded[j:j + 16] for j in range(start, min(start + 4, len(images)))])
                value = torch.from_numpy(batch).permute(0, 4, 1, 2, 3).float().to(self.device) / 255 - .5
                features.append(self.i3d(value)['embds'].reshape(len(batch), 1024).cpu())
            features = torch.cat(features)
            probabilities = []
            for start in range(0, len(features), 100):
                value = features[start:start + 100].T[None].contiguous()
                probabilities.extend(self.segmenter(value, torch.ones_like(value))[-1].softmax(1)[0, 1].tolist())
        if not np.isfinite(probabilities).all():
            raise ValueError('nonfinite Renz boundaries')
        return probabilities

    def predict(self, observations, owned_start, owned_end):
        from scripts.live_reel_stage1_v17 import VerifiedCommitLock
        started = time.perf_counter()
        probabilities = self.boundaries(rgb_input(observations))
        predictions = []
        for start, end in sign_regions(probabilities, observations[0].seconds, owned_start, owned_end):
            selected = [o for o in observations if start <= o.seconds < end]
            if len(selected) < 4:
                predictions.append(dict(start_seconds=start, end_seconds=end, committed_gloss=None, reason='short_region'))
                continue
            proposal, verifier = self.reel.classify(selected), self.reel.verify(selected)
            agreement = proposal['model_score'] if proposal.get('candidate_gloss') == verifier.get('candidate_gloss') else 0.
            commit = bool(proposal.get('accepted') and verifier.get('accepted')) and VerifiedCommitLock(self.args.commit_hits, self.args.instant_commit_score).update(str(verifier.get('candidate_gloss')), max(verifier['model_score'], agreement), proposal=str(proposal.get('candidate_gloss')), proposal_score=proposal['model_score'], minimum_score=self.args.commit_score)
            predictions.append(dict(start_seconds=start, end_seconds=end, committed_gloss=verifier.get('candidate_gloss') if commit else None, proposal=proposal, verifier=verifier))
        return dict(predictions=predictions, boundary_probability=probabilities, start_seconds=observations[0].seconds, end_seconds=observations[-1].seconds, owned_start=owned_start, owned_end=owned_end, inference_seconds=time.perf_counter() - started)


def run(args, shell, model):
    from scripts.live_isolated_v17 import SessionRecorder, orient_frame
    from scripts.live_stage2_ctc_v17 import LatestCamera, observe_stage2_frame
    from active.v17.extract_v17 import AppleVisionDetector
    source = str(args.video) if args.video else args.camera
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f'could not open {source}')
    recorder = SessionRecorder(args, model, str(source))
    recorder.data['prototype_scope'] = 'experimental buffered Renz; 4s windows/2s hop/1s right context; literal output'
    detector = AppleVisionDetector(args.minimum_point_confidence)
    started = time.perf_counter()
    camera = None if args.video else LatestCamera(capture, started)
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 1 else 30.
    executor = ThreadPoolExecutor(max_workers=1)
    observations, actions, glosses = deque(), deque(), []
    future = None
    generation = job_generation = processed = sequence = frame_index = 0
    next_end = None
    owned_end = -1.
    next_process = 0.
    wrists = {'left': None, 'right': None}
    status = 'Slow diagnostic: collecting 4 seconds'; delay = 0.; paused_before = True; flushing = False
    skipped = 0; reviewed = 0.
    shell.attach('Sign Language Translation', [1280, 720], started, lambda action, seconds: actions.append(action))
    try:
        while True:
            if camera:
                packet = camera.after(sequence)
                if packet is None:
                    if camera.failed: break
                    time.sleep(.001)
                    continue
                sequence, seconds, raw = packet
            else:
                ok, raw = capture.read()
                if not ok: break
                seconds = frame_index / fps; frame_index += 1
                time.sleep(max(0., min(.05, started + seconds - time.perf_counter())))
            if shell.paused != paused_before:
                actions.append('reset_buffer'); paused_before = shell.paused
            while actions:
                action = actions.popleft()
                if action == 'finish': flushing = True
                else:
                    generation += 1; observations.clear(); next_end = None; owned_end = seconds
                    wrists = {'left': None, 'right': None}; flushing = False
                    if action == 'reset': glosses.clear()
                    recorder.add_event({'type': 'renz_reset', 'seconds': seconds, 'action': action})
            if future is not None and future.done():
                result = future.result(); future = None
                if job_generation == generation:
                    reviewed = result['owned_end']
                    delay = seconds - reviewed
                    recorder.add_event({'type': 'renz_buffer', **result, 'display_delay_seconds': delay})
                    for prediction in result['predictions']:
                        recorder.add(prediction)
                        if prediction['committed_gloss']: glosses.append(prediction['committed_gloss'])
                    status = f"Last inference {result['inference_seconds']:.1f}s; delay {delay:.1f}s"
            canonical = orient_frame(raw, args.rotation, args.input_mirrored)
            if not shell.paused:
                recorder.write_frame(canonical, seconds)
                if seconds + 1e-6 >= next_process:
                    observations.append(observe_stage2_frame(canonical, seconds, processed, detector, wrists, args)); processed += 1
                    next_process = max(next_process + 1 / args.processing_fps, seconds)
                    if next_end is None: next_end = seconds + 4
                while observations and observations[0].seconds < seconds - 12: observations.popleft()
                if future is None and next_end is not None and (seconds >= next_end or flushing):
                    if observations and next_end - 4 < observations[0].seconds - .15:
                        skipped += 1
                        recorder.add_event({'type': 'renz_backlog_skipped', 'from_seconds': owned_end, 'to_seconds': observations[0].seconds})
                        next_end = observations[0].seconds + 4
                        status = 'Processing behind camera; skipped old buffer (recorded)'
                    end = seconds if flushing else next_end
                    clip = [o for o in observations if end - 4 <= o.seconds <= end]
                    if len(clip) >= 4:
                        new_owned_end = end if flushing else end - 1
                        future = executor.submit(model.predict, clip, owned_end, new_owned_end)
                        job_generation = generation; owned_end = new_owned_end; next_end = end + 2; flushing = False
            shown = cv2.flip(canonical, 1) if not args.no_mirror_display else canonical.copy()
            for index, line in enumerate(['RENZ EXPERIMENTAL | buffered 4s | R reset / F flush / Q quit', status + (' | processing...' if future else ''), f'Camera {seconds:.1f}s | reviewed through {reviewed:.1f}s | skipped buffers {skipped}', ' '.join(glosses[-12:])]):
                cv2.putText(shown, line, (20, 95 + index * 35), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
            key = shell.present(shown, glosses)
            if key in (ord('q'), 27): break
            if key == ord('r'): actions.append('reset')
            if key in (ord('f'), ord(' ')): actions.append('finish')
    finally:
        recorder.finish_video()
        if camera: camera.close()
        else: capture.release()
        if future is not None:
            recorder.add_event({'type': 'renz_pending_discarded_on_exit'})
        executor.shutdown(wait=True, cancel_futures=True)
        recorder.close()
