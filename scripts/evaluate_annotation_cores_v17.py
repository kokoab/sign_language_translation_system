"""Fixed saved-model diagnostic on already-used development sign cores; no training."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.finalize_asllrp_other_ctc_manifest_v17 import sha256
from active.v17.train_unified_streaming_ctc_v17 import RawSequence, rolling_windows, encode
from active.v17.train_streaming_tcn_ctc_v17 import restore_source_frames, collapse_ctc
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.model_unified_streaming_ctc_v17 import load_unified_streaming_head


def core_metrics(rows):
    output = {}
    for kind in ('known', 'oov'):
        selected = [r for r in rows if (r['expected'] == 101) == (kind == 'oov')]
        output[kind] = dict(
            samples=len(selected), exact=sum(r['predicted'] == [r['expected']] for r in selected),
            other_emitted=sum(101 in r['predicted'] for r in selected),
            blank_only=sum(not r['predicted'] for r in selected),
            false_known=sum(any(0 < t < 101 for t in r['predicted']) for r in selected) if kind == 'oov' else None,
            other_without_expected=sum(101 in r['predicted'] and r['expected'] not in r['predicted']
                                       for r in selected) if kind == 'known' else None)
    return output


@torch.inference_mode()
def run():
    torch.set_num_threads(2)
    report = ROOT / 'artifacts/reports/annotation_identity_audit_v17_20260921'
    manifest = report / 'development_cores.json'
    cores = json.loads(manifest.read_text())
    models = ROOT / 'artifacts/models/flores_other_mps_retry_v17_20260921'
    paths = [models / f'{arm}_{seed}/best_model.pth' for seed in (17321, 17322)
             for arm in ('without_flores', 'with_flores')]
    first = torch.load(paths[0], map_location='cpu', weights_only=False)
    base_path = ROOT / first['base_checkpoint']
    base_hash = sha256(base_path)
    base = torch.load(base_path, map_location='cpu', weights_only=False)
    stage1 = SLTStage1V17(Stage1V17Config(**base['model_config']))
    stage1.load_state_dict(base['model_state_dict'], strict=True)
    labels = {r['canonical_label']: r['class_index'] for r in
              json.loads((ROOT / 'active/v17/citizen100_manifest.json').read_text())['classes']}
    assert base['label_to_index'] == labels
    raw, cache = [], {}
    for row in cores:
        if row['archive'] not in cache:
            with np.load(ROOT / row['archive'], allow_pickle=False) as d:
                cache[row['archive']] = restore_source_frames(d['landmarks'], d['window_source_ranges'])
        frames = cache[row['archive']][row['start_frame']:row['end_frame_exclusive']]
        raw.append(RawSequence(rolling_windows(frames, 4, 8), (row['expected_ctc_index'],),
                               row['status'], row['identity'] + ':' + row['annotation_id']))
    samples = encode(stage1, raw, torch.device('cpu'), 32, 'window')
    results = {}
    for path in paths:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['base_checkpoint_sha256'] == base_hash and payload['label_to_index'] == labels
        model = load_unified_streaming_head(payload, device='cpu').eval()
        predictions = []
        for sample in samples:
            logits = model(torch.from_numpy(sample.evidence.astype(np.float32))[None])[0]
            predicted = list(collapse_ctc(logits.argmax(-1).numpy()))
            predictions.append(dict(identity=sample.identity, expected=sample.targets[0], predicted=predicted))
        results[path.parent.name] = dict(checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=sha256(path),
                                         metrics=core_metrics(predictions), predictions=predictions)
        print(json.dumps({'model': path.parent.name, 'metrics': results[path.parent.name]['metrics']}), flush=True)
    artifact = dict(protocol='Presegmented core CTC; stride4/window8; fixed four prior arms; reused development signer.',
                    manifest_sha256=sha256(manifest), base_checkpoint_sha256=base_hash,
                    archive_sha256={p: sha256(ROOT / p) for p in cache},
                    independent_unseen_evaluation=False, training_started=False, test_accessed=False,
                    results=results)
    (report / 'core_evaluation.json').write_text(json.dumps(artifact, indent=2) + '\n')


if __name__ == '__main__':
    run()
