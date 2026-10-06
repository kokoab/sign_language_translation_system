"""Steps 3a/3b of CHAIN_PLAN.md: local replay from 96.83, gate evaluation, fusion."""
import argparse,hashlib,json,os,subprocess,traceback
from datetime import datetime,timezone
from pathlib import Path
# Defaults reproduce the first run; Variant B (CHAIN_PLAN.md) passes --name/--citizen-floor/--patience.
_cli=argparse.ArgumentParser();_cli.add_argument('--name',default='chain_9683')
_cli.add_argument('--citizen-floor',type=int,default=365);_cli.add_argument('--patience',default='20')
_cli.add_argument('--skip-fusion-reference',action='store_true')
_cli.add_argument('--full-roll-probability',default='0.35');CLI=_cli.parse_args()
ROOT=Path('/Volumes/secret/SLT/SLT');OUT=Path(__file__).resolve().parent/CLI.name
PY=str(ROOT/'venv/bin/python')
SOURCE=ROOT/'artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/best_model.pth'
SOURCE_SHA='5c40b13336b4692d5f7e1e70a9ba430aa2b35ef4e12946952e15fe1f9e54924b'
CITIZEN_FLOOR,SEMLEX_FLOOR=CLI.citizen_floor,853
LOCAL=['--local-root','data/local/local_deep_clean_v17/landmarks/train',
       '--local-manifest','data/local/local_deep_clean_v17/train_final_manifest.json',
       '--local-tiers','owner_approved_v16_deep_clean','--approve-local-supplement',
       '--local-validation-root','data/local/local_deep_clean_v17/landmarks/val',
       '--local-validation-manifest','data/local/local_deep_clean_v17/val_final_manifest.json',
       '--mask-local-mouth-nodes']
TRAIN=[PY,'-m','active.v17.train_stage_1_v17','--fine-tune-from',str(SOURCE),
       '--temporal-encoder','partwise_global','--part-depth','1','--device','mps',
       '--epochs','80','--patience',CLI.patience,'--batch-size','64','--workers','0','--seed','1701',
       '--lr','0.00005','--warmup-epochs','4','--weight-decay','0.03','--label-smoothing','0.1',
       '--ema-decay','0.999','--sampling','class_source_balanced',
       '--source-probabilities','citizen=.34,semlex=.33,local=.33',
       '--full-roll-probability',CLI.full_roll_probability,'--maximum-roll-degrees','180','--mild-roll-degrees','12',
       '--supplement-root','data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17',
       '--supplement-manifest','data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json',
       '--approve-supplement',*LOCAL,'--citizen-top1-floor-correct',str(CITIZEN_FLOOR),
       '--output',str(OUT/'local_replay')]
ENV=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def status(**kwargs):
    (OUT/'status.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),**kwargs},indent=2)+'\n')
def call(name,cmd,timeout):
    status(state='running',stage=name)
    with (OUT/(name+'.log')).open('w') as log:
        subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
def correct(metrics):return int(round(metrics['top1']*metrics['samples']/100.0))
def gates(name,path,parent_local):
    call('semlex_'+name,[PY,'scripts/evaluate_semlex_citizen100_train_audit.py',str(path),
        '--feature-root','data/local/semlex_citizen100_val_audit/landmarks_v17',
        '--provenance','data/local/semlex_citizen100_val_audit/download_provenance.json',
        '--expected-split','val','--output-dir',str(OUT/('semlex_'+name))],1800)
    call('orientation_'+name,[PY,'-m','active.v17.evaluate_orientation_robustness_v17',str(path),
        '--device','mps','--output-dir',str(OUT/('orientation_'+name))],1800)
    ck=__import__('torch').load(path,map_location='cpu',weights_only=False)
    semlex=json.loads((OUT/('semlex_'+name)/'summary.json').read_text())
    orient=json.loads((OUT/('orientation_'+name)/'metrics.json').read_text())
    row=dict(name=name,path=str(path),sha256=sha(path),epoch=ck['epoch'],
        citizen_correct=correct(ck['validation_metrics']),
        semlex_correct=int(round(semlex['clip_top1']*semlex['clips'])),
        local_top1=ck['local_validation_metrics']['top1'],parent_local_top1=parent_local,
        worst_angle_correct=min(a['top1_correct'] for a in orient['per_angle'] if a['angle_degrees']!=0),
        mean_rotated_top1=orient['mean_nonzero_top1'])
    row['gates']=dict(citizen=row['citizen_correct']>=CITIZEN_FLOOR,semlex=row['semlex_correct']>=SEMLEX_FLOOR,
        local=row['local_top1']>parent_local,orientation=row['worst_angle_correct']>=2)
    row['eligible']=all(row['gates'].values())
    return row
def fusion(name,landmark,floor):
    cmd=[PY,'-m','active.v17.train_unified_multimodal_student_v17','--rebuild-cache',
         '--cache-dir',str(OUT/(name+'_cache')),'--output',str(OUT/name),'--citizen-floor-correct',str(floor)]
    if landmark is not None:cmd+=['--landmark-checkpoint',str(landmark)]
    call(name,cmd,7200)
def run():
    OUT.mkdir(exist_ok=False)
    assert sha(SOURCE)==SOURCE_SHA
    (OUT/'recipe.json').write_text(json.dumps({'plan':'CHAIN_PLAN.md','local_replay_command':TRAIN,
        'citizen_floor':CITIZEN_FLOOR,'patience':CLI.patience,'full_roll_probability':CLI.full_roll_probability,'semlex_floor':SEMLEX_FLOOR,'source_sha256':SOURCE_SHA,
        'code_sha256':{p:sha(ROOT/p) for p in ['active/v17/train_stage_1_v17.py','active/v17/model_v17.py',
            'active/v17/train_unified_multimodal_student_v17.py']},'test_accessed':False},indent=2)+'\n')
    try:
        call('local_replay_train',TRAIN,4*3600)
        result=json.loads((OUT/'local_replay'/'result.json').read_text())
        parent_local=result['training_data_provenance']['initialization']['initial_local_validation_metrics']['top1']
        rows=[gates(n,OUT/'local_replay'/f,parent_local) for n,f in
              (('promotion_gate','best_promotion_gate_model.pth'),('citizen_best','best_model.pth'))]
        eligible=[r for r in rows if r['eligible']]
        selected=eligible[0] if eligible else rows[0]
        selection=dict(candidates=rows,selected=selected['name'],selected_path=selected['path'],
                       selected_sha256=selected['sha256'],gate_eligible=bool(eligible))
        (OUT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        if not CLI.skip_fusion_reference:fusion('fusion_reference_august',None,361)
        fusion('fusion',Path(selected['path']),selected['citizen_correct'])
        call('raw_orientation_selected',[PY,'-m','active.v17.evaluate_raw_orientation_v17',selected['path'],
             '--split','val','--clips-per-class','1','--device','mps','--vision-auto-orient',
             '--output',str(OUT/'raw_orientation_selected'/'metrics.json')],5400)
        status(state='complete',promotion='none; results require review')
    except Exception:
        status(state='failed',error=traceback.format_exc());raise
    finally:
        subprocess.run(['osascript','-e','display notification "96.83 chain finished or stopped. Check "+CLI.name+"/status.json." with title "ATLAS experiment"'],check=False)
if __name__=='__main__':run()
