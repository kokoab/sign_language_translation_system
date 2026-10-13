"""Check evidence consistency and source immutability; pin local model files."""
import json
from pathlib import Path
from prepare import sha, HERE, ROOT
from replay import selfcheck

def main():
    identity=json.loads((HERE/'identity.json').read_text())
    assert sha(HERE/'cases.json')==identity['cases_sha256']
    for name,expected in identity['sources'].items():assert sha(ROOT/name)==expected,name
    cases=json.loads((HERE/'cases.json').read_text())
    expected={c['id'] for c in cases}
    for name in ['tiny','flan','flan_fewshot','qwen','qwen_direct']:
        p=HERE/f'{name}_outputs.json'
        if not p.exists():continue
        outputs=json.loads(p.read_text());assert {r['id'] for r in outputs}==expected
        rows=json.loads((HERE/f'{name}_replay.json').read_text())
        assert len(rows)==4*sum(c['event_complete'] for c in cases)
        for r in rows:
            c=next(c for c in cases if c['id']==r['id'])
            assert c['event_complete']
            assert all(e['n']<=len(c['glosses']) for e in r['events'])
            assert all(e['ready']>=e['at'] for e in r['events'])
            assert all(e['at']>=c['arrivals'][e['n']-1]-c['arrivals'][0] for e in r['events'])
    selfcheck()
    fingerprints={}
    flan=Path.home()/'.cache/huggingface/hub/models--google--flan-t5-base'
    revision=(flan/'refs/main').read_text().strip()
    folders=[flan/'snapshots'/revision,HERE/'models/qwen05']
    for folder in folders:
        if folder.exists():
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix in ('.json','.safetensors','.bin','.model','.txt'):
                    fingerprints[str(p)]=dict(bytes=p.stat().st_size,sha256=sha(p))
    qwen=HERE/'models/qwen05/model.safetensors'
    if qwen.exists():
        assert fingerprints[str(qwen)]['sha256']=='fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
    evidence={p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.json','.jsonl','.csv','.md') and p.name!='integrity.json'}
    out=dict(source_hashes_verified=len(identity['sources']),flan_revision=revision,
             qwen_revision='7ae557604adf67be50417f59c2c2f167def9a775',
             local_models=fingerprints,evidence=evidence,atlas_writes=False,
             training_seconds=0,validation='PASS')
    (HERE/'integrity.json').write_text(json.dumps(out,indent=2)+'\n')
    print('PASS: source hashes, causal replay invariants, local model and evidence fingerprints')

if __name__=='__main__':main()
