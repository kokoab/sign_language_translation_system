"""Phrase-segment adaptation and interval recognizer for both chains on the approved recipe split.

Recipe: active/v17/phrase_segment_recipe_manifest_20261007.json (train = approved train ∩ lab
train, 179; validation = approved validation − lab test, 139; lab tune = recognizer selection;
lab test untouched). Commands reproduce the August stages exactly (phrase stage verified to
reproduce reel_v2 epoch 26 363/872/2810/172/723; span recognizer reproduction run separately),
changing only base checkpoint, isolated cache and the phrase/span split.
Then, for the 96.83 chain only: letter head (head-A recipe) and Core ML FP32/FP16 exports.
"""
import hashlib,json,os,subprocess,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/Volumes/secret/SLT/SLT');HERE=Path(__file__).resolve().parent;OUT=HERE/'downstream_recipe'
PY=str(ROOT/'venv/bin/python')
RECIPE=ROOT/'active/v17/phrase_segment_recipe_manifest_20261007.json'
VIEW=ROOT/'data/local/phrase_segment_recipe_v17_20261007'
CHAINS={'chain_9683':dict(base=HERE/'chain_9683_floor361_mildroll/fusion/best_model.pth',
                          cache=HERE/'chain_9683_floor361_mildroll/fusion_cache'),
        'august':dict(base=ROOT/'artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth',
                      cache=ROOT/'artifacts/generated/unified_multimodal_student_v17')}
SPAN_CFG='{"lookahead":8,"span_mode":"both","alpha":0.7,"w_r":2,"w_a":0.25,"w_b":0.5,"c":-1,"log_theta":-1.6094}'
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
def floors(domains):
    # 2026-09-25 rule: no domain more than 1 point below the parent; a domain already below 90 keeps the parent value.
    return {d:(v-1.0 if v-1.0>=90.0 else v) for d,v in ((d,domains[d]['top1']) for d in ('citizen','semlex','local'))}
def verify_view():
    recipe=json.loads(RECIPE.read_text())
    expected={(e['split'],kind,Path(e[kind]).name):e[kind+'_sha256'] for e in recipe['entries'] for kind in ('rgb','hand')}
    found={}
    for kind,folder in (('rgb','multimodal'),('hand','hand')):
        for split in ('train','validation'):
            for link in (VIEW/folder/split/'local_phrases').iterdir():found[(split,kind,link.name)]=link
    assert set(found)==set(expected),'recipe view membership differs from manifest'
    for key,link in found.items():assert sha(link)==expected[key],key
    return recipe
def run():
    OUT.mkdir(exist_ok=False)
    verified=subprocess.run([PY,'-m','active.v17.approved_phrase_data_v17'],cwd=ROOT,capture_output=True,text=True,check=True)
    canonical=json.loads(verified.stdout[verified.stdout.index('{'):])
    recipe=verify_view()
    assert recipe['canonical_sha256']==canonical['sha256'] and recipe['recipe_training_ready']
    (OUT/'recipe.json').write_text(json.dumps({'recipe_manifest':str(RECIPE.relative_to(ROOT)),'recipe_sha256':sha(RECIPE),
        'canonical_sha256':canonical['sha256'],'chains':{k:{'base':str(v['base']),'base_sha256':sha(v['base']),
        'cache_sha256':{p.name:sha(p) for p in sorted(v['cache'].glob('*.npz'))}} for k,v in CHAINS.items()},
        'span_cfg':SPAN_CFG,'code_sha256':{p:sha(ROOT/p) for p in ['active/v17/train_unified_phrase_adapt_v17.py',
        'scripts/train_span_recognizer_v17.py','scripts/train_letter_head_v17.py','active/v17/export_span_recognizer_batched_coreml_v17.py']},
        'test_accessed':False},indent=2)+'\n')
    def stages(name,chain):
        phrase=OUT/name/'phrase_adapt'
        call(f'{name}_phrase_adapt',[PY,'-m','active.v17.train_unified_phrase_adapt_v17','--base',str(chain['base']),
             '--isolated-cache',str(chain['cache']),'--phrase-rgb',str(VIEW/'multimodal'),'--phrase-hand',str(VIEW/'hand'),
             '--selection-key','reel_v2','--pair-loss-weight','0','--pair-sampling-multiplier','1',
             '--output-dir',str(phrase)],3600)
        selected=json.loads((phrase/'result.json').read_text())['selected']['domains']
        call(f'{name}_span_recognizer',[PY,'scripts/train_span_recognizer_v17.py','--output-dir',str(OUT/name/'span_recognizer'),
             '--base',str(phrase/'best_model.pth'),'--epochs','8','--seed','27927','--sets','local_train',
             '--exclude-prefix-spans','--recipe-manifest',str(RECIPE),'--floors',json.dumps(floors(selected)),
             '--cfg',SPAN_CFG],4*3600)
    try:
        stages('chain_9683',CHAINS['chain_9683'])
        recognizer=OUT/'chain_9683/span_recognizer/best_model.pth'
        letters=OUT/'chain_9683/letter_head_a/model.pth';letters.parent.mkdir(parents=True,exist_ok=True)
        call('chain_9683_letter_head_a',[PY,'scripts/train_letter_head_v17.py','--recognizer',str(recognizer),
             '--output',str(letters),'--no-corrections'],4*3600)
        (OUT/'coreml').mkdir(exist_ok=True)
        for precision in ('float32','float16'):
            tag='FP32' if precision=='float32' else 'FP16'
            call(f'export_{tag}',[PY,'active/v17/export_span_recognizer_batched_coreml_v17.py',str(recognizer),
                 str(OUT/'coreml'/f'SpanRecognizerV17Chain9683LettersB8{tag}.mlpackage'),'--precision',precision,
                 '--letter-head',str(letters),'--fixed-batch','8'],3600)
        stages('august',CHAINS['august'])
        status(state='complete',promotion='none; results require review')
    except Exception:
        status(state='failed',error=traceback.format_exc());raise
    finally:
        subprocess.run(['osascript','-e','display notification "Downstream recipe run finished or stopped." with title "ATLAS experiment"'],check=False)
if __name__=='__main__':run()
