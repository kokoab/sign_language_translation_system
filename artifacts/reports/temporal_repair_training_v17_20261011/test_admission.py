import json,copy
from train import validate_recipe
from features import O
c=json.loads((O/'training_recipe.json').read_text());assert validate_recipe(c);checks={}
def reject(name,fn):
 d=copy.deepcopy(c);fn(d)
 try:validate_recipe(d)
 except ValueError as e:checks[name]=str(e)
 else:raise AssertionError(name)
reject('not_ready',lambda d:d.__setitem__('recipe_training_ready',False))
reject('different_training_rule',lambda d:d['training'].__setitem__('temporal_share',.8))
reject('missing_core_cache',lambda d:d['pins'].pop(str(O/'features/temporal_train.npz')))
reject('tampered_cache',lambda d:d['pins'].__setitem__(str(O/'features/temporal_train.npz'),'0'*64))
reject('incomplete_event_audit',lambda d:d['input_summary'].__setitem__('events_attempted',143))
reject('promotion_bypass',lambda d:d.__setitem__('promotion_ready',True))
(O/'admission_tests.json').write_text(json.dumps({'valid_recipe_passed':True,'invalid_cases_rejected':checks},indent=2)+'\n');print('Admission verification and six negative cases passed.')
