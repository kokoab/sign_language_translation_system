"""Verify saved real-video streams and CPU/MPS agreement of published encoder."""
from smoke import *

torch.set_num_threads(2)
streams = dict(np.load(OUT / 'streams.npz'))
report = json.loads((OUT / 'smoke.json').read_text())
length = report['frames']
for key, value in streams.items():
    assert value.shape == (length, 14 if key == 'body_posture' else 384)
    assert np.isfinite(value).all()
model = SHubertModel(SHubertConfig())
checkpoint = torch.load(WEIGHTS / 'checkpoint_836_400000.pt', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()
source = {key: torch.from_numpy(value) for key, value in streams.items()}
for key in list(source):
    source['label_' + key] = torch.zeros((length, 1))
with torch.inference_mode():
    expected = model.extract_features([source], padding_mask=None, kmeans_labels=None, mask=False)['x'].numpy()
actual = np.load(OUT / 'features.npy')
np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
result = {'cpu_mps_max_abs_difference': float(np.abs(actual-expected).max()), 'passed': True,
          'scope': 'SHuBERT encoder on saved real-video streams; frontend/DINO CPU parity not tested'}
(OUT / 'check.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
