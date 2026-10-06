"""Bounded serial fine-tuning through the existing approved Stage-1 trainer."""
import json,os,subprocess,sys,hashlib,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/Volumes/secret/SLT/SLT');OUT=Path(__file__).resolve().parent/'finetune'
SOURCE=ROOT/'artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/best_model.pth'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def status(**kwargs):
    (OUT/'status.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),**kwargs},indent=2)+'\n')
def run():
    OUT.mkdir(exist_ok=False)
    assert sha(SOURCE)=='5c40b13336b4692d5f7e1e70a9ba430aa2b35ef4e12946952e15fe1f9e54924b'
    py=str(ROOT/'venv/bin/python')
    base=[py,'-m','active.v17.train_stage_1_v17','--fine-tune-from',str(SOURCE),
        '--temporal-encoder','partwise_global','--part-depth','1','--device','mps',
        '--epochs','20','--patience','20','--batch-size','64','--workers','0','--seed','1701',
        '--lr','0.00005','--warmup-epochs','2','--weight-decay','0.03','--label-smoothing','0.1',
        '--ema-decay','0.999','--sampling','class_source_balanced',
        '--source-probabilities','citizen=.5,semlex=.5','--maximum-roll-degrees','180','--mild-roll-degrees','12',
        '--supplement-root','data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17',
        '--supplement-manifest','data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json',
        '--approve-supplement']
    commands={name:base+['--full-roll-probability',prob,'--output',str(OUT/name)]
              for name,prob in [('mild_control','0'),('full_roll','0.35')]}
    (OUT/'recipe.json').write_text(json.dumps({'commands':commands,'source_sha256':sha(SOURCE),
        'code_sha256':{p:sha(ROOT/p) for p in ['active/v17/train_stage_1_v17.py','active/v17/model_v17.py']},
        'test_accessed':False},indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    try:
        for name,cmd in commands.items():
            status(state='running',stage=name)
            with (OUT/(name+'.log')).open('w') as log:
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        for name,path in [('original',SOURCE)]+[(n,OUT/n/'best_model.pth') for n in commands]:
            status(state='evaluating',stage=name)
            cmd=[py,'-m','active.v17.evaluate_orientation_robustness_v17',str(path),
                 '--device','mps','--output-dir',str(OUT/('orientation_'+name))]
            with (OUT/('orientation_'+name+'.log')).open('w') as log:
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        status(state='complete',promotion='none; results require review')
    except Exception:
        status(state='failed',error=traceback.format_exc());raise
    finally:
        subprocess.run(['osascript','-e','display notification "Squeezeformer fine-tuning finished or stopped. Check finetune/status.json." with title "ATLAS experiment"'],check=False)
if __name__=='__main__':run()
