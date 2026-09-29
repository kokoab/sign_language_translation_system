"""Parity fixture for the Swift FINGERSPELL trigger: raw frames -> window rows, probabilities, firings."""
import base64, glob, gzip, json, pickle, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from active.v17.fingerspell_trigger_v17 import FingerspellTrigger, TriggerModel, window_features
from active.v17.stage1_window_v17 import raw_observation_features
model = TriggerModel(); trig = FingerspellTrigger(model)
clips = []
raw_dir = ROOT / 'artifacts/reports/fingerspell_detector_v17_20260930/raw'
for p in sorted(raw_dir.glob('citizen_val__*.npz')) + sorted(raw_dir.glob('citizen_train__*.npz'))[::12]:
    d = np.load(p); clips.append((p.stem, d['raw'], d['times']))
for p in sorted(glob.glob(str(ROOT / 'artifacts/reports/letter_arbitration_v17_20260929/session_cache/*.pkl.gz')))[:2]:
    obs, _ = pickle.load(gzip.open(p)); raw, times = raw_observation_features(obs)
    clips.append((Path(p).name[:22], raw[:1200], times[:1200]))
cases = []
for name, raw, times in clips:
    raw = np.asarray(raw, np.float32)
    if len(raw) < 20:
        continue
    rows = window_features(raw)
    trig.reset()
    fires = [list(f) for r, t in zip(raw, times) if (f := trig.push(r, t))]
    cases.append(dict(name=name, raw=base64.b64encode(raw.tobytes()).decode(), times=[float(t) for t in times],
                      rows=rows.astype(float).tolist(), probs=[model.probability(r) for r in rows], fires=fires))
(HERE / 'trigger_fixture.json').write_text(json.dumps(cases))
print(len(cases), 'cases', sum(len(c['rows']) for c in cases), 'rows', sum(len(c['fires']) for c in cases), 'fires')
