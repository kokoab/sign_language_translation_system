"""Strict synthetic weight/forward check; no pose conversion or accuracy claim."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
SOURCE = HERE / 'source/sign_language_segmentation/model'

def model_from_upstream(config):
    namespace = dict(torch=torch, nn=nn, F=F, SimpleNamespace=SimpleNamespace, BIO=range(4))
    # Execute only reviewed architecture definitions. Lightning training/metrics
    # dependencies are omitted; tensor operations and parameter names are unchanged.
    for name in ('pose_encoder.py', 'model.py'):
        tree = ast.parse((SOURCE / name).read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) or
                     (name == 'pose_encoder.py' and isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign)))]
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == 'PoseTaggingModel':
                node.bases = [ast.parse('nn.Module', mode='eval').body]
                node.body = [n for n in node.body if not isinstance(n, ast.FunctionDef) or n.name in ('__init__', 'encode', 'forward')]
                init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
                init.body = [ast.parse('self.hparams = SimpleNamespace(num_frames=num_frames)').body[0]
                             if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and
                             isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'save_hyperparameters'
                             else n for n in init.body]
        tree.body.insert(0, ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0))
        exec(compile(ast.fix_missing_locations(tree), str(SOURCE / name), 'exec'), namespace)
    return namespace['PoseTaggingModel'](**config)


def main():
    provenance = json.loads((HERE / 'provenance.json').read_text())
    for item in provenance['files']:
        assert hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() == item['sha256']
    weights = ROOT / 'artifacts/models/pose_boundary_dgs_2026'
    model = model_from_upstream(json.loads((weights / 'config.json').read_text())).float().eval()
    state = load_file(str(weights / 'model.safetensors'))
    model.load_state_dict(state, strict=True)
    torch.manual_seed(17621)
    x = torch.randn(1, 64, 50, 6)
    times = torch.arange(64).float()[None] / 25
    with torch.inference_mode():
        cpu = model(x, timestamps=times)
        altered = x.clone(); altered[:,32:] += 2
        future = model(altered, timestamps=times)
        delta = float((cpu['sign'][:,:16] - future['sign'][:,:16]).abs().max())
        assert all(t.shape == (1,64,4) and torch.isfinite(t).all() for t in cpu.values())
        gpu = model.to('mps')(x.to('mps'), timestamps=times.to('mps'))
        difference = max(float((cpu[k]-gpu[k].cpu()).abs().max()) for k in cpu)
        assert difference < .005
    # Fresh normal tensors: the preceding device move ran under inference_mode.
    model = model_from_upstream(json.loads((weights / 'config.json').read_text())).float()
    model.load_state_dict(state, strict=True)
    model.to('mps').train()
    loss = -model(x.to('mps'), timestamps=times.to('mps'))['sign'][...,2].mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    result = dict(synthetic_mps_gradient_finite=True, optimizer_steps=0, strict_tensors=len(state), trainable_parameters=sum(p.numel() for p in model.parameters()),
                  input_shape=list(x.shape), output_shape=list(cpu['sign'].shape),
                  cpu_mps_max_difference=difference, future_input_changes_early_output=delta,
                  scope='Synthetic compatibility only. Full-sequence model; no real-video ASL accuracy or live latency established.',
                  harness='Upstream architecture AST; nn.Module replaces Lightning bookkeeping only.')
    (HERE/'compatibility.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__ == '__main__': main()
