"""Checkpoint compatibility only; synthetic input is not an accuracy test."""
import hashlib
import importlib.util
import json
from pathlib import Path
import torch

root = Path(__file__).resolve().parent
for row in json.loads((root / 'provenance.json').read_text()):
    weight = row['weight']
    assert hashlib.sha256(Path(weight['local']).read_bytes()).hexdigest() == weight['sha256']
spec = importlib.util.spec_from_file_location('zhao_segment', root / 'source/SegmentASLTransformer/model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
state = torch.load(root / 'weights/transformer_model_f1_0.8689.pth', map_location='cpu', weights_only=True)
model = module.TransformerSegmenter(189, 20, 32, 128, 4, 2, num_classes=3).eval()
model.load_state_dict(state, strict=True)
assert torch.backends.mps.is_available()
torch.manual_seed(7)
x = torch.randn(1, 32, 189)
idx = torch.zeros(1, 32, dtype=torch.long)
prob = torch.zeros(1, 32)
with torch.no_grad():
    cpu = model(x, idx, prob, idx)[0]
    model.to('mps')
    mps = model(x.to('mps'), idx.to('mps'), prob.to('mps'), idx.to('mps'))[0].cpu()
assert torch.isfinite(mps).all()
delta = float((cpu - mps).abs().max())
assert delta < 1e-4, delta
hand = torch.load(root / 'weights/asl_model.pt', map_location='cpu', weights_only=True)
assert len(hand) == 1 and list(hand[0]['fc.weight'].shape) == [88, 256]
result = dict(device='mps', strict_segment_tensors=len(state), output_shape=list(mps.shape), cpu_mps_max_abs_difference=delta, handshape_output_classes=88, synthetic_only=True, accuracy_tested=False, passed=True)
(root / 'check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
