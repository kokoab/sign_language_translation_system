"""Rerun the paired recipe, retaining trained-epoch weights for orientation diagnostics.

The first paired run (finetune/) selected epoch 0 in both arms, so the trained weights
were never evaluated. This run uses the identical recipe plus --save-diagnostic-checkpoints.
best_model.pth selection is unchanged; diagnostic weights are never promoted.
"""
import json,os,subprocess,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/Volumes/secret/SLT/SLT');HERE=Path(__file__).resolve().parent;OUT=HERE/'finetune_diagnostic'
SOURCE=ROOT/'artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/best_model.pth'
REFERENCE=ROOT/'artifacts/generated/kaggle_stage1_orientation_robust_pull_v2/stage1_v17_orientation_robust_v1/best_model.pth'
HASHES={SOURCE:'5c40b13336b4692d5f7e1e70a9ba430aa2b35ef4e12946952e15fe1f9e54924b',
        REFERENCE:'a7490409b3dfd76ba1ff432d2392b5e27df33f12e1b088cd36609fe03c082366'}
def sha(p):
    import hashlib;h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def status(**kwargs):
    (OUT/'status.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),**kwargs},indent=2)+'\n')
def run():
    OUT.mkdir(exist_ok=False)
    for path,digest in HASHES.items():assert sha(path)==digest,path
    recipe=json.loads((HERE/'finetune/recipe.json').read_text())
    py=str(ROOT/'venv/bin/python')
    commands={}
    for name,cmd in recipe['commands'].items():
        cmd=list(cmd);cmd[cmd.index('--output')+1]=str(OUT/name)
        commands[name]=cmd+['--save-diagnostic-checkpoints']
    (OUT/'recipe.json').write_text(json.dumps({'commands':commands,'source_sha256':HASHES[SOURCE],
        'reference_sha256':HASHES[REFERENCE],'first_run_recipe':'finetune/recipe.json',
        'code_sha256':{p:sha(ROOT/p) for p in ['active/v17/train_stage_1_v17.py','active/v17/model_v17.py']},
        'test_accessed':False},indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    try:
        for name,cmd in commands.items():
            status(state='running',stage=name)
            with (OUT/(name+'.log')).open('w') as log:
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        targets=[('reference_a7490409',REFERENCE)]+[(f'{n}_{kind}',OUT/n/f'{kind}_model.pth')
                 for n in commands for kind in ('final','best_trained')]
        for name,path in targets:
            status(state='evaluating',stage=name)
            cmd=[py,'-m','active.v17.evaluate_orientation_robustness_v17',str(path),
                 '--device','mps','--output-dir',str(OUT/('orientation_'+name))]
            with (OUT/('orientation_'+name+'.log')).open('w') as log:
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        status(state='complete',promotion='none; diagnostic weights only')
    except Exception:
        status(state='failed',error=traceback.format_exc());raise
    finally:
        subprocess.run(['osascript','-e','display notification "Diagnostic fine-tune rerun finished or stopped. Check finetune_diagnostic/status.json." with title "ATLAS experiment"'],check=False)
if __name__=='__main__':run()
