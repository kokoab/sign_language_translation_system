"""Step 2 of CHAIN_PLAN.md: raw-pixel roll with Vision auto-orientation, 96.83 and a7490409."""
import hashlib,json,os,subprocess,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/Volumes/secret/SLT/SLT');OUT=Path(__file__).resolve().parent/'raw_orientation'
TARGETS={'partwise_9683':(ROOT/'artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/best_model.pth',
                          '5c40b13336b4692d5f7e1e70a9ba430aa2b35ef4e12946952e15fe1f9e54924b'),
         'reference_a7490409':(ROOT/'artifacts/generated/kaggle_stage1_orientation_robust_pull_v2/stage1_v17_orientation_robust_v1/best_model.pth',
                               'a7490409b3dfd76ba1ff432d2392b5e27df33f12e1b088cd36609fe03c082366')}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def status(**kwargs):
    (OUT/'status.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),**kwargs},indent=2)+'\n')
def run():
    OUT.mkdir(exist_ok=False)
    for path,digest in TARGETS.values():assert sha(path)==digest,path
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    try:
        for name,(path,_) in TARGETS.items():
            status(state='running',stage=name)
            cmd=[str(ROOT/'venv/bin/python'),'-m','active.v17.evaluate_raw_orientation_v17',str(path),
                 '--split','val','--clips-per-class','1','--device','mps','--vision-auto-orient',
                 '--output',str(OUT/name/'metrics.json')]
            with (OUT/(name+'.log')).open('w') as log:
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=5400)
        status(state='complete')
    except Exception:
        status(state='failed',error=traceback.format_exc());raise
    finally:
        subprocess.run(['osascript','-e','display notification "Raw orientation evaluation finished or stopped." with title "ATLAS experiment"'],check=False)
if __name__=='__main__':run()
