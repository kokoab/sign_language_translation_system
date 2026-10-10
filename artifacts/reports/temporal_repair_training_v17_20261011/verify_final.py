"""Fail closed on incomplete experiment outputs or changed production/input sources."""
from pathlib import Path
import hashlib, json
import torch
from features import O, BASE, sha
from train import validate_recipe

recipe=json.loads((O/'training_recipe.json').read_text());assert validate_recipe(recipe)
assert json.loads((O/'pipeline_completion.json').read_text())['returncode']==0
result=json.loads((O/'training_results.json').read_text())
assert len(result['runs'])==4 and all(len(r['history'])==7 for r in result['runs'])
assert result['selected']['row']['eligible']
base=torch.load(BASE,map_location='cpu',weights_only=False)
for arm in ['selected','diagnostic']:
    checkpoint=torch.load(O/f'{arm}.pth',map_location='cpu',weights_only=False)
    for field in ['landmark_model_state_dict','hand_model_state_dict']:
        assert all(torch.equal(v,checkpoint[field][k]) for k,v in base[field].items())
    export=json.loads((O/f'{arm}_export.json').read_text())
    assert export['word_top1_agreement']==export['letter_top1_agreement']==export['samples']==378
    assert export['letter_reference_exact_max_abs']==0
    assert sha(O/f'{arm}.pth')==export['checkpoint_sha256']
for arm in ['incumbent','selected','diagnostic']:
    for runtime in ['control','continuous_contact']:
        rows=json.loads((O/f'{arm}_{runtime}_replay.json').read_text())['records']
        assert len(rows)==len({r['id'] for r in rows})==72
    validation=json.loads((O/f'{arm}_validation.json').read_text())['records']
    assert len(validation)==570
    assert len({r['reference'][0] for r in validation if r['dataset']=='citizen_val'})==100
    genuine=json.loads((O/f'{arm}_genuine_controls.json').read_text())['records']
    assert len(genuine)==59
for value in json.loads((O/'actual_fp16_letter_preservation.json').read_text()).values():
    assert value['n']==value['top1_agreement']==378 and value['logit_max_abs']==0
production=json.loads((O/'production_preservation.json').read_text())
for path,digest in production['pins'].items():assert sha(path)==digest,path
for p in [O.parent/'temporal_repair_contract_v17_20261011/contract.json',O.parents[2]/'active/v17/approved_phrase_manifest_20260921_v2.json']:
    assert not json.loads(p.read_text())['training_ready']
native={str(p):sha(p) for folder in ['eval_control','eval_continuous_contact'] for p in (O/folder).glob('*.swift')}
for name in ['evaluation_cases.json','genuine_control_cases.json','export_stable.py','run_evaluation.py','run_candidate_controls.py','analyze_evaluation.py','evaluate_development_cores.py','evaluate_cached_heads.py','check_coreml_letters.py']:
    native[str(O/name)]=sha(O/name)
(O/'evaluation_source_pins.json').write_text(json.dumps(native,indent=2)+'\n')
verification={'passed':True,'four_training_arms':True,'native_phase_replays':432,'cached_streaming_replays':1710,'genuine_control_replays':177,'citizen_classes_per_arm':100,'encoder_weights_unchanged':True,'actual_fp16_letter_outputs_exact':True,'production_sources_unchanged':True,'canonical_training_flag_unchanged':True,'new_phone_test':False,'installed':False}
(O/'verification.json').write_text(json.dumps(verification,indent=2)+'\n');print(json.dumps(verification,indent=2))
