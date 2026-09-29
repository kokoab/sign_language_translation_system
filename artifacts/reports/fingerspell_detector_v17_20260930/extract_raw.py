"""Apple Vision raw landmarks [T,61,5] + times (20 Hz, live observation contract) for detector clips."""
import json, sys, random
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_temporal_boundary_v17 import arguments
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.stage1_window_v17 import raw_observation_features
OUT = Path(__file__).parent / 'raw'
args = arguments('data', ROOT / 'artifacts/generated/tmp_sessions')


def items():
    out = []
    for p in sorted((ROOT / 'data/local/fingerspell_trigger_v17/citizen').glob('*/*.mp4')):
        out.append(dict(kind='fingerspell_sign', split='citizen_' + p.parent.name, path=p))
    for p in sorted((ROOT / 'data/local/fingerspell_trigger_v17/asllrp').glob('*.mp4')):
        out.append(dict(kind='fingerspell_sign', split='asllrp', path=p))
    rng = random.Random(0)
    for d in sorted((ROOT / 'data/local/citizen100_v17/raw/train').iterdir()):
        clips = sorted(p for p in d.glob('*.mp4') if not p.name.startswith('._'))
        rng.shuffle(clips)
        out += [dict(kind='sign', split='citizen_train', gloss=d.name, path=p) for p in clips[:6]]
    return out


k, n = map(int, sys.argv[1].split('/'))
OUT.mkdir(exist_ok=True)
for i, it in enumerate(items()[k::n]):
    target = OUT / (it['split'] + '__' + it['path'].stem + '.npz')
    if target.exists():
        continue
    det = AppleVisionDetector(args.minimum_point_confidence)
    cap = cv2.VideoCapture(str(it['path'])); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    wr, j, deadline, obs = {'left': None, 'right': None}, 0, 0., []
    while True:
        ok, frame = cap.read()
        if not ok: break
        t = j / fps; j += 1
        if t + 1e-6 < deadline: continue
        deadline = max(deadline + .05, t)
        obs.append(observe_stage2_frame(frame, t, len(obs), det, wr, args))
    cap.release()
    if len(obs) < 4:
        continue
    raw, times = raw_observation_features(obs)
    np.savez_compressed(target, raw=raw.astype(np.float32), times=times, meta=json.dumps(
        dict(kind=it['kind'], split=it['split'], gloss=it.get('gloss', 'FINGERSPELL'), video=str(it['path'].relative_to(ROOT)))))
    if i % 50 == 0:
        print(i, it['split'], flush=True)
print('done', flush=True)
