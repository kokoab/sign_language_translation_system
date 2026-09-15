"""Training-only separation of blank calibration from anchor label discrimination."""
import importlib.util
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('anchor_diagnosis_run',HERE/'run_anchor.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch,np=m.torch,m.np

def main():
    torch.set_num_threads(2)
    data=m.load_data();path=sorted(m.MODELS.glob('epoch_*.pth'))[-1]
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=m.old.fresh().cpu().eval();model.load_state_dict(checkpoint['state_dict'])
    groups={}
    with torch.inference_mode():
        for source in ('asllrp_contiguous','asllrp_other_ctc'):
            rows=[r for r in data['sequences'] if r['role']=='train' and r['source']==source and any(t<101 for _,t in r['anchors'])]
            rows=[rows[i] for i in np.linspace(0,len(rows)-1,min(32,len(rows)),dtype=int)]
            counts=Counter();blank_probs=[];target_probs=[];nll=[]
            for begin in range(0,len(rows),4):
                batch=rows[begin:begin+4];logits,_=model.sequences([r['value'] for r in batch])
                for r,values in zip(batch,logits):
                    for index,target in r['anchors']:
                        if target==101:continue
                        scores=values[index];p=scores.softmax(-1)
                        counts['known_anchors']+=1;counts['blank_argmax']+=int(scores.argmax()==0)
                        counts['correct_argmax']+=int(scores.argmax()==target)
                        counts['correct_excluding_blank']+=int(scores[1:].argmax()+1==target)
                        blank_probs.append(float(p[0]));target_probs.append(float(p[target]));nll.append(float(-p[target].log()))
            groups[source]=dict(counts,mean_blank_probability=float(np.mean(blank_probs)),mean_target_probability=float(np.mean(target_probs)),mean_target_nll=float(np.mean(nll)),sequences=len(rows))
    result=dict(epoch=checkpoint['epoch'],checkpoint_sha256=m.old.old._sha256(path),groups=groups,training_only=True,decoder_changed=False)
    m.save('training_anchor_diagnosis.json',result);m.old.emit(result)

if __name__=='__main__':main()
