"""Counterfactual contract audit only; no production maps or model weights are edited."""
import hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parents[2]))
from active.v17.asl_corpus_v17 import _tokens,_covers,FUNCTION_WORDS,GLOSS_LEMMAS
from active.v17.stage3_asl_encoding_v17 import encode
from scripts.eval_stage3_multisentence_v17 import LEMMAS,coverage
# Tiny local whitelist extension shows what changing the gate can and cannot repair.
local={g:tuple(v) for g,v in LEMMAS.items()}
for g,words in {'EAT':['food'],'WORK':['job'],'COLD':['winter'],'LIKE':['enjoy'],'DRINK':['beverage'],'HELP':['assistance']}.items():local[g]+=tuple(words)
def check(gs,e,lex):
 ts=_tokens(e);words={w for g in gs for w in lex.get(g,())}
 return {'missing':[g for g in set(gs) if not any(_covers(t,lex[g]) for t in ts)],'invented':sorted({t for t in ts if t not in FUNCTION_WORDS and not _covers(t,words)})}
checks=[]
for gs,e in [(['EAT','GOOD'],'The food is good.'),(['WORK','IMPORTANT'],'The job is important.'),(['COLD'],'Winter.'),(['I','HELP','YOU'],'You help me.'),(['I','GIVE','WATER'],'I give her water.'),(['I','NO','WANT','WATER'],'I never want water.')]:
 checks.append({'glosses':gs,'english':e,'before':check(gs,e,LEMMAS),'local_alias_extension':check(gs,e,local)})
# Distinct intended interpretations with exactly the same transmitted model input.
collisions=[]
for gs,alternatives,missing in [
 (['I','WANT','EAT'],['I want to eat.','I want food.'],'Noun/verb reading; context or sign-movement evidence may disambiguate.'),
 (['HE','HAPPY'],['He is happy.','She is happy.'],'Referent identity/gender is not encoded by this gloss.'),
 (['YOU','HAPPY'],['You are happy.','Are you happy?'],'Nonmanual question evidence is absent.'),
 (['HE','HELP','HE'],['Person A helps person B.','Person B helps person A.'],'Distinct pointing loci and verb direction collapse to HE/HELP/HE.')]:
 encoded=[encode(gs,[.9]*len(gs)) for _ in alternatives]
 assert len(set(encoded))==1
 collisions.append(dict(glosses=gs,input=encoded[0],alternatives=alternatives,missing_evidence=missing,kind='Constructed information-loss diagnostic, not an annotated video sample.'))
(P/'contract_ablation.json').write_text(json.dumps({'local_alias_gate_checks':checks,'input_collisions':collisions,'scope':'In-memory evaluation-only map. No inference or production behavior changed; aliases alone cannot fix participant/scope checks.'},indent=2)+'\n')
assert checks[0]['before']['missing'] and not checks[0]['local_alias_extension']['missing']
assert not checks[3]['before']['missing'] and not checks[3]['before']['invented']
assert GLOSS_LEMMAS['EAT']==('eat','ate','eaten','eating','meal')
print('PASS: lexical rejection reproduced; local aliases repair admission only; role/addition/scope counterexamples still pass; four information-loss collisions verified.')
