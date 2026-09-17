"""Read-only model/config audit: CPU/meta only, no weights or training downloads."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from transformers import BartConfig, BartForConditionalGeneration, MT5Config, MT5ForConditionalGeneration, T5Config, T5ForConditionalGeneration, T5Tokenizer

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def remote_json(url):
    with urlopen(url, timeout=40) as response:
        return json.load(response)


def count(config, cls):
    with torch.device('meta'):
        model = cls(config)
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    torch.set_num_threads(1)
    local = ROOT/'data/local/unisign_asl_baseline_20260916/mt5-base'
    config = MT5Config.from_pretrained(str(local), local_files_only=True)
    text_parameters = count(config, MT5ForConditionalGeneration)
    provenance = json.loads((ROOT/'artifacts/reports/stage1_direct_translation_20260917/provenance.json').read_text())
    non_text = provenance['parameters'] - text_parameters
    projection = (256 + 1) * config.d_model
    encoder_parameters = non_text - projection
    rows = [dict(model='current Uni-Sign mT5 hybrid', text_parameters=text_parameters,
                 total_parameters=provenance['parameters'], vocab_size=config.vocab_size,
                 d_model=config.d_model, encoder_layers=12, decoder_layers=12)]
    for size in [64000, 32128]:
        smaller = MT5Config.from_dict(config.to_dict())
        smaller.vocab_size = size
        n = count(smaller, MT5ForConditionalGeneration)
        assert text_parameters-n == 2*(config.vocab_size-size)*config.d_model
        rows.append(dict(model=f'current hybrid with {size} retained tokens (hypothetical)',
                         text_parameters=n, total_parameters=n+non_text,
                         vocab_size=size, d_model=768, encoder_layers=12, decoder_layers=12))
    for repo in ['google/t5-v1_1-base', 'google/flan-t5-base', 'google/flan-t5-small', 'facebook/bart-base']:
        revision = remote_json('https://huggingface.co/api/models/'+repo)['sha']
        url = f'https://huggingface.co/{repo}/resolve/{revision}/config.json'
        d = remote_json(url)
        (HERE/(repo.split('/')[-1]+'_config.json')).write_text(json.dumps(d, indent=2)+'\n')
        bart = repo=='facebook/bart-base'
        c = (BartConfig if bart else T5Config).from_dict(d)
        n = count(c, BartForConditionalGeneration if bart else T5ForConditionalGeneration)
        rows.append(dict(model=repo, revision=revision, config_url=url,
                         text_parameters=n, total_parameters=n+encoder_parameters+(256+1)*c.d_model,
                         vocab_size=c.vocab_size, d_model=c.d_model,
                         encoder_layers=c.encoder_layers if bart else c.num_layers,
                         decoder_layers=c.decoder_layers if bart else c.num_decoder_layers))
    for row in rows:
        row['fp16_weight_MB'] = row['total_parameters']*2/1e6
        row['parameter_reduction_percent'] = 100*(1-row['total_parameters']/provenance['parameters'])
    manifest_path = ROOT/'artifacts/reports/stage1_direct_translation_20260917/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    tokenizer = T5Tokenizer.from_pretrained(str(local), local_files_only=True, legacy=False)
    ids = [tokenizer(row['reference'])['input_ids'] for row in manifest['train']]
    lengths = [len(x) for x in ids]
    result = dict(training_run=False, gpu_used=False, downloaded_model_weights=False,
        method='Parameter objects on meta device, official pinned configs, local training references only.',
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        encoder_parameters=encoder_parameters, current_text_parameters=text_parameters,
        current_embedding_and_output_parameters=2*config.vocab_size*config.d_model,
        train_target_audit=dict(count=len(ids), mean_tokens=float(np.mean(lengths)),
            median_tokens=float(np.median(lengths)), max_tokens=max(lengths),
            p95_tokens=float(np.percentile(lengths,95)), unique_token_ids=len({i for s in ids for i in s}),
            fixed_70_token_padding_fraction=1-sum(lengths)/(70*len(ids))),
        candidates=rows,
        limitations=['Counts and theoretical weight bytes, not speed or accuracy measurements.',
                     'Trimmed vocabulary membership has not been selected or validated.',
                     'Existing pretrained model weights were not converted or evaluated.'])
    assert len(ids)==994 and max(lengths)<=70
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
