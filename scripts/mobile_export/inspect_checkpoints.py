"""Inspect v16 checkpoints — confirm architecture keys, d_model, vocab_size, etc.

Read-only. Does not modify any codebase files.
"""
import os, sys, json, torch
from pathlib import Path

REPO = Path('/Users/frnzlo/Documents/machine_learning/SLT')
sys.path.insert(0, str(REPO))

CKPT_STAGE1 = REPO / 'src_v16' / 'output_v16_d384' / 'best_model.pth'
CKPT_STAGE2 = Path('/Users/frnzlo/Downloads/models 2/output_stage2_v16/stage2_best_model.pth')
T5_DIR      = REPO / 'weights' / 'slt_final_t5_model'


def inspect(path: Path, label: str):
    print(f'\n=== {label} ===')
    print(f'path: {path}')
    if not path.exists():
        print('  MISSING')
        return
    print(f'  size: {path.stat().st_size/1e6:.1f} MB')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        print(f'  type: {type(ckpt)} (not a dict)')
        return
    meta_keys = [k for k in ckpt.keys() if k != 'model_state_dict' and k != 'optimizer_state_dict' and k != 'scheduler_state_dict' and k != 'ema_shadow']
    for k in meta_keys:
        v = ckpt[k]
        if isinstance(v, (int, float, str, bool)):
            print(f'  {k} = {v}')
        elif isinstance(v, dict):
            print(f'  {k}: dict with {len(v)} entries')
        elif v is None:
            print(f'  {k} = None')
        else:
            print(f'  {k}: {type(v).__name__}')
    sd = ckpt.get('model_state_dict', {})
    print(f'  state_dict: {len(sd)} keys')
    # Show top-level module names
    mods = sorted({k.split('.')[0] for k in sd.keys()})
    print(f'  top-level modules: {mods}')
    # Show a few key shapes
    target_patterns = [
        'encoder.input_proj.0.weight',
        'encoder.pos_enc',
        'encoder.blocks.0.attn.in_proj_weight',
        'frame_attn.0.weight',
        'classifier.0.weight',
        'classifier.6.weight',
        'tcn.branch3.0.weight',
        'seq_blocks.0.attn.in_proj_weight',
        'ctc_head.weight',
        'inter_ctc_proj.weight',
        'seq_pos_enc',
    ]
    for pat in target_patterns:
        for k, v in sd.items():
            kc = k.replace('_orig_mod.', '')
            if kc == pat:
                print(f'    {kc}: {tuple(v.shape)}')
                break


if __name__ == '__main__':
    inspect(CKPT_STAGE1, 'Stage 1 — output_v16_d384/best_model.pth')
    inspect(CKPT_STAGE2, 'Stage 2 — output_stage2_v16/stage2_best_model.pth')
    print('\n=== Stage 3 T5 ===')
    print(f'path: {T5_DIR}')
    if T5_DIR.exists():
        for f in sorted(T5_DIR.iterdir()):
            print(f'  {f.name}: {f.stat().st_size/1e6:.1f} MB')
        cfg = T5_DIR / 'config.json'
        if cfg.exists():
            c = json.loads(cfg.read_text())
            print(f'  model_type: {c.get("model_type")}')
            print(f'  architectures: {c.get("architectures")}')
            print(f'  d_model: {c.get("d_model")}')
            print(f'  num_layers: {c.get("num_layers")}')
            print(f'  num_heads: {c.get("num_heads")}')
            print(f'  vocab_size: {c.get("vocab_size")}')
