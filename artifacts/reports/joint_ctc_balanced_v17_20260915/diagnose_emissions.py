"""Locate training emissions relative to verified intervals; no decoder changes."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('balanced_diagnosis',HERE/'run_balanced.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
m=n.m;torch,np=m.torch,m.np

def main():
    torch.set_num_threads(2)
    data=m.load_data();checkpoint_path=sorted(n.MODELS.glob('epoch_*.pth'))[-1]
    payload=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    model=m.old.fresh().cpu().eval();model.load_state_dict(payload['state_dict'])
    manifest=json.loads(m.old.MANIFEST.read_text())
    source_rows={(r['source'],r['source_item_id']):r for r in manifest['rows'] if r['role']=='train'}
    groups={}
    with torch.inference_mode():
        for source in ('asllrp_contiguous','asllrp_other_ctc'):
            rows=[r for r in data['sequences'] if r['role']=='train' and r['source']==source]
            rows=[rows[i] for i in np.linspace(0,len(rows)-1,min(32,len(rows)),dtype=int)]
            counts=Counter()
            for begin in range(0,len(rows),4):
                batch=rows[begin:begin+4];logits,lengths=model.sequences([r['value'] for r in batch])
                for r,scores,length in zip(batch,logits,lengths):
                    value=r['value'];times=np.concatenate([t[k] for t,k in zip(value.times,value.keep)])
                    events=source_rows[(source,r['identity'])]['intervals']
                    regions=np.zeros(length,dtype=int);coverage=np.zeros(length,dtype=int)
                    for e in events:
                        mask=(times>=e['start_seconds'])&(times<=e['end_seconds'])
                        regions[mask]=2 if e['label']=='__OTHER__' else 1;coverage[mask]+=1
                    regions[coverage>1]=3
                    path=scores[:length].argmax(-1).tolist();previous=None
                    for region,p in zip(regions,path):
                        if 1<=p<=100 and p!=previous:counts[['gap','known_interval','other_interval','overlap'][region]]+=1
                        previous=p
                    counts['reference_known']+=sum(t<101 for t in r['targets'])
                    counts['sequences']+=1
            groups[source]=dict(counts)
    result=dict(epoch=payload['epoch'],checkpoint_sha256=n.sha(checkpoint_path),groups=groups,
        training_only=True,limitation='Emission token timestamps may differ from annotation alignment; region counts are diagnostic, not event-level accuracy.')
    m.save('training_emission_locations.json',result);m.old.emit(result)

if __name__=='__main__':main()
