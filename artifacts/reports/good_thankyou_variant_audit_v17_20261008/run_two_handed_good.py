"""Experiment: GOOD trained only on visually verified two-handed clips (chin -> non-dominant palm).

Variant C local replay from 96.83 (full roll 0, floor 361, patience 80, 80 epochs, seed 1701,
34/33/33 replay, local lip mask) with one change: one-handed/unclear GOOD training clips are
excluded from Citizen (rejections copy), SemLex and local (filtered manifests); 30 two-handed GOOD
remain (Citizen 2, SemLex 6, local 22). THANKYOU and all validation data unchanged. Evaluation
splits GOOD validation clips by visual verdict. Validation only; no test access.
"""
import json,os,subprocess,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/Volumes/secret/SLT/SLT');HERE=Path(__file__).resolve().parent;OUT=HERE/'two_handed_good';INPUTS=HERE/'two_handed_good_inputs'
CHAIN=ROOT/'artifacts/reports/canonical_recognition_comparison_v17_20261007/chain_9683_floor361_mildroll'
PY=str(ROOT/'venv/bin/python');ENV=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
def status(**k):(OUT/'status.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),**k},indent=2)+'\n')
def call(name,cmd,timeout):
    status(state='running',stage=name)
    with (OUT/(name+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
EVAL='''
import sys,json,csv,torch,collections
from pathlib import Path
sys.path.insert(0,'.')
from active.v17.train_unfrozen_phrase_adapt_v17 import isolated_raw,tensors,logits_for
from active.v17.model_v17 import SLTStage1V17,Stage1V17Config
review={}
for r in csv.DictReader(open(sys.argv[3])):
    if r['source'].endswith('_val'):review[(r['source'].split('_')[0],'GOOD/'+Path(r['archive']).name.split('.')[0])]=r['visual']
raw=isolated_raw(Path('artifacts/generated/unfrozen_phrase_adapt_v17'));dev=torch.device('mps');result={}
for name,path in json.loads(sys.argv[1]).items():
    c=torch.load(path,map_location='cpu',weights_only=False)
    net=SLTStage1V17(Stage1V17Config(**c['model_config']));net.load_state_dict(c['model_state_dict'])
    m=type('L',(torch.nn.Module,),{'forward':lambda s,l,*a:s.net(l)})();m.net=net;m.to(dev).eval()
    lab=c['label_to_index'];inv={v:k for k,v in lab.items()};row=dict(epoch=c['epoch'])
    for split in ('citizen_val','semlex_val','local_val'):
        v=tensors(raw[split]);pred=logits_for(m,v,dev).argmax(1).tolist();t=v[4].tolist();ids=[str(x) for x in raw[split]['item_ids']]
        dom=split.split('_')[0];groups=collections.defaultdict(collections.Counter)
        for i,p,y in zip(ids,pred,t):
            g=inv[y]
            if g=='GOOD':g='GOOD_'+review.get((dom,i),'unreviewed')
            elif g!='THANKYOU':continue
            groups[g][inv[p] if inv[p] in ('GOOD','THANKYOU') else 'other']+=1
        row[split]=dict(correct=sum(p==y for p,y in zip(pred,t)),n=len(t),groups={k:dict(v) for k,v in groups.items()})
    result[name]=row;m.cpu()
Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result,indent=1))
'''
def run():
    OUT.mkdir(exist_ok=False)
    cmd=list(json.loads((CHAIN/'recipe.json').read_text())['local_replay_command'])
    for flag,value in (('--supplement-manifest',INPUTS/'semlex_train_two_handed_good.json'),
                       ('--local-manifest',INPUTS/'local_train_two_handed_good.json'),('--output',OUT/'local_replay')):
        cmd[cmd.index(flag)+1]=str(value)
    cmd+=['--rejections',str(INPUTS/'citizen_rejections_two_handed_good.csv')]
    (OUT/'recipe.json').write_text(json.dumps({'command':cmd,'baseline':str(CHAIN/'local_replay'),'test_accessed':False},indent=2)+'\n')
    try:
        call('local_replay_train',cmd,4*3600)
        call('semlex_promotion_gate',[PY,'scripts/evaluate_semlex_citizen100_train_audit.py',str(OUT/'local_replay/best_promotion_gate_model.pth'),
             '--feature-root','data/local/semlex_citizen100_val_audit/landmarks_v17','--provenance',
             'data/local/semlex_citizen100_val_audit/download_provenance.json','--expected-split','val',
             '--output-dir',str(OUT/'semlex_promotion_gate')],1800)
        targets={'96.83_isolated':str(ROOT/'artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/best_model.pth'),
                 'variant_c_baseline':str(CHAIN/'local_replay/best_promotion_gate_model.pth'),
                 'no_local_good_thankyou':str(HERE/'no_local_good_thankyou/local_replay/best_promotion_gate_model.pth'),
                 'two_handed_good':str(OUT/'local_replay/best_promotion_gate_model.pth')}
        call('evaluate_groups',[PY,'-c',EVAL,json.dumps(targets),str(OUT/'group_eval.json'),str(HERE/'visual_review_GOOD.csv')],3600)
        status(state='complete',promotion='none; experiment')
    except Exception:
        status(state='failed',error=traceback.format_exc());raise
    finally:
        subprocess.run(['osascript','-e','display notification "Two-handed GOOD experiment finished or stopped." with title "ATLAS experiment"'],check=False)
if __name__=='__main__':run()
