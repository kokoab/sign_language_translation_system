"""Read-only model experiment; generated gloss probes are not validated ASL videos."""
import csv,hashlib,json,re,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from active.v17.asl_corpus_v17 import GLOSS_LEMMAS, FUNCTION_WORDS, _covers, _tokens
HERE=Path(__file__).resolve().parent
MOBILE=ROOT.parent/'mobile_app/slt_mobile_app/ios/Runner/LiveReel'
PACKAGES=ROOT/'artifacts/coreml/stage3_multisentence_tiny_v17_20260929'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj): (HERE/name).write_text(json.dumps(obj,indent=2)+'\n')
labels=json.loads((MOBILE/'Models/live_reel_labels.json').read_text())
manifest=json.loads((ROOT/'active/v17/citizen100_manifest.json').read_text())
assert [r['canonical_label'] for r in manifest['classes']]==labels
# Map every class to a small interpretable text probe. These are diagnostic gloss inputs,
# not assertions that linear glosses preserve all spatial/nonmanual ASL information.
contexts={}
def addgroup(words,pattern):
 for w in words.split():contexts[w]=pattern.replace('{}',w)
addgroup('I YOU WE THEY HE','{} HAPPY')
addgroup('MY YOUR OUR','{} CHILD HAPPY')
contexts.update(WHAT='YOU WANT WHAT',WHERE='YOUR CHILD WHERE',WHEN='YOU GO WHEN',WHO='YOUR FRIEND WHO',WHY='YOU SAD WHY',HOW='YOU HOW',HELLO='HELLO MY FRIEND',GOODBYE='GOODBYE MY FRIEND',PLEASE='PLEASE HELP I',THANKYOU='THANKYOU MY FRIEND',SORRY='I SORRY',YES='YES I UNDERSTAND',NO='I NO UNDERSTAND',MAYBE='MAYBE I GO TOMORROW')
addgroup('HELP LIKE LOVE KNOW UNDERSTAND SEE HEAR','I {} YOU')
addgroup('WANT NEED','I {} WATER')
contexts.update(THINK='I THINK YOU GOOD',FEEL='I FEEL GOOD',HAVE='I HAVE WATER',GIVE='YOU GIVE I WATER',TAKE='I TAKE WATER',COME='YOU COME HOME',GO='I GO SCHOOL',STOP='PLEASE STOP',WAIT='PLEASE WAIT',TRY='I TRY LEARN',USE='I USE SIGN LANGUAGE',MAKE='I MAKE EAT',FIND='I FIND MY CHILD',LISTEN='I LISTEN YOU',TALK='I TALK YOU',TELL='YOU TELL I',ASK='I ASK YOU',ANSWER='PLEASE ANSWER I',LEARN='I LEARN SIGN LANGUAGE',WORK='I WORK HOME',READ='I LIKE READ',WRITE='I LIKE WRITE',EAT='I WANT EAT',DRINK='I WANT DRINK',WATER='I WANT WATER',SLEEP='I NEED SLEEP')
addgroup('GOOD BAD HAPPY SAD ANGRY EXCITED TIRED SICK HUNGRY HOT COLD READY','I {}')
addgroup('BIG SMALL IMPORTANT','MY SCHOOL {}')
contexts.update(MORE='I WANT MORE WATER',LESS='I WANT LESS WATER',SAME='MY SCHOOL YOUR SCHOOL SAME',DIFFERENT='MY SCHOOL YOUR SCHOOL DIFFERENT',EASY='SIGN LANGUAGE EASY')
addgroup('NOW TOMORROW YESTERDAY MORNING NIGHT','{} I WORK')
contexts.update(TIME='TIME WHAT',DAY='GOOD DAY',WEEK='I WORK WEEK',YEAR='YEAR WHAT')
addgroup('HOME SCHOOL HOSPITAL','I GO {}')
addgroup('DOCTOR FAMILY FRIEND MOTHER FATHER CHILD MAN WOMAN','I SEE {}')
contexts.update(NAME='MY NAME fs-Alex',SIGN='I LEARN SIGN',LANGUAGE='I LEARN SIGN LANGUAGE')
assert set(contexts)==set(labels),(set(labels)-set(contexts),set(contexts)-set(labels))
cases=[]
def case(family,g,focus=None,**extra):
 if isinstance(g,str):g=g.split()
 assert all(x in labels or x.startswith('fs-') for x in g),g
 row=dict(id=f'{family}_{sum(r["family"]==family for r in cases):03d}',family=family,glosses=g,scores=[.9]*len(g),focus=focus,incremental=True,**extra)
 cases.append(row);return row
for w in labels:
 case('isolated',w,w,incremental_probe=False)['incremental']=False
 case('context',contexts[w],w)
 # Identical content in multi-sentence stream tests contextual interference and locks.
 case('wrapped','HELLO '+contexts[w]+' THANKYOU',w)
subjects={'I':'I','YOU':'you','WE':'we','THEY':'they','HE':'he'}
objects={'I':'me','YOU':'you','WE':'us','THEY':'them','HE':'him'}
for a in subjects:
 for b in subjects:
  if a==b:continue
  for v in ['HELP','SEE','LIKE','LOVE']:
   case('roles',f'{a} {v} {b}',v,subject=a,object=b,reference=f'{subjects[a]} {v.lower()} {objects[b]}',interpretation='Declared subject-verb-object diagnostic; directional/spatial evidence absent.')
for owner in ['MY','YOUR','OUR']:
 for noun in ['CHILD','FRIEND','MOTHER','FATHER','FAMILY','DOCTOR']:
  for placement in ['subject','object']:
   g=f'{owner} {noun} HAPPY' if placement=='subject' else f'I HELP {owner} {noun}'
   case('possession',g,noun,owner=owner,placement=placement)
for time in ['YESTERDAY','NOW','TOMORROW']:
 for verb in ['EAT','DRINK','GO','COME','WORK','SEE','GIVE']:
  tail={'GO':'HOME','COME':'HOME','SEE':'YOU','GIVE':'YOU WATER'}.get(verb,'')
  case('time',f'{time} I {verb} {tail}',verb,time=time)
for v in ['WANT','NEED','LIKE','LOVE','KNOW','UNDERSTAND','HAVE','SEE','HEAR','EAT','DRINK','SLEEP','WORK','GO','HELP']:
 tail={'WANT':'WATER','NEED':'WATER','LIKE':'YOU','LOVE':'YOU','KNOW':'YOU','HAVE':'WATER','SEE':'YOU','HEAR':'YOU','GO':'HOME','HELP':'YOU'}.get(v,'')
 case('negation',f'I NO {v} {tail}',v)
# Candidate lexical readings: unreviewed sign-form compatibility is explicitly marked.
for focus,g,reading in [
 ('EAT','EAT GOOD','food'),('EAT','EAT HOT','food'),('EAT','EAT COLD','food'),('EAT','MY EAT WHERE','food'),('EAT','I WANT EAT','eat OR food'),('EAT','I NEED EAT','eat OR food'),('EAT','I HAVE EAT','food OR eaten'),('EAT','I LIKE EAT','food OR eating'),
 ('DRINK','DRINK COLD','drink as noun'),('DRINK','MY DRINK WHERE','drink as noun'),('DRINK','I WANT DRINK','drink as noun OR verb'),('DRINK','I LIKE DRINK','drink as noun OR verb'),
 ('WORK','MY WORK IMPORTANT','work/job'),('WORK','YOUR WORK WHERE','work/job'),('WORK','I HAVE WORK','work/job'),
 ('SIGN','SIGN GOOD','sign/signing'),('SIGN','I LIKE SIGN','signing'),('SIGN','YOUR SIGN DIFFERENT','sign/signing'),
 ('HELP','I NEED HELP','help as noun'),('HELP','THANKYOU YOUR HELP','help as noun'),('LOVE','YOUR LOVE IMPORTANT','love as noun'),
 ('ANSWER','I NEED ANSWER','answer as noun'),('ANSWER','YOUR ANSWER DIFFERENT','answer as noun'),('TIME','I HAVE TIME','time'),
 ('COLD','I COLD','temperature; illness/head-cold is NOT licensed'),('COLD','COLD I NO WORK','winter OR temperature candidate'),('HOT','WATER HOT','hot'),
 ('LIKE','I LIKE YOU','liking; do not infer similar'),('SAME','I YOU SAME','same/alike'),('GOOD','I GOOD','good/well'),
 ('HE','HE HAPPY','he/she/they without gender evidence'),('THEY','THEY HAPPY','plural they'),('WE','YOU HELP WE','us'),('I','YOU HELP I','me'),
 ('YOUR','WATER YOUR','yours'),('MY','WATER MY','mine'),('OUR','WATER OUR','ours'),
 ('DAY','DAY I WORK','day; today is not guaranteed'),('NIGHT','NIGHT I WORK','night; tonight is not guaranteed'),
 ('NO','I NO HAVE WATER','not; never is stronger'),('MORE','I WANT MORE','more'),('LESS','I WANT LESS','less'),
 ('LEARN','I LEARN','learn; student not automatically licensed'),('DOCTOR','MY DOCTOR GOOD','doctor'),
 ('TALK','I TALK YOU','talk/speak'),('SICK','I SICK','sick/ill'),('CHILD','MY CHILD HAPPY','child/kid'),
 ('MOTHER','MY MOTHER HAPPY','mother/mom'),('FATHER','MY FATHER HAPPY','father/dad'),('SMALL','MY CHILD SMALL','small/little')]:
 case('meaning',g,focus,reading=reading)
for g in ['I WANT EAT','YOU HELP I','HE HAPPY','MY CHILD HAPPY','YOUR CHILD HAPPY','OUR CHILD HAPPY','EAT GOOD','DRINK COLD','I NO WANT WATER','YESTERDAY I GO HOME','I HAVE BIG','HELLO I WANT EAT THANKYOU','I HELP YOU YOU HELP I','I WANT EAT EAT GOOD','I NO WANT WATER I WANT EAT']:
 for confidence in [.5,.2]:
  row=case('confidence',g,confidence=confidence);row['scores']=[confidence]*len(row['glosses'])
for g in ['I HELP YOU YOU HELP I','I WANT EAT EAT GOOD','HE HAPPY HE NO TIRED','MY CHILD SICK I NEED HELP','I WANT WATER YOU WANT EAT','YESTERDAY I WORK TOMORROW I SLEEP','I NO WANT WATER I WANT EAT','HELLO MY NAME fs-Alex I WANT EAT THANKYOU']:
 case('multisentence',g)
for g in ['I WANT','MY','YOUR','I HAVE BIG','I GIVE','GIVE I','HELP','HE','EAT','COLD','YOU HELP YOU','NO NO','I I','EAT EAT','HELP HELP']:
 case('ambiguous',g,reading='Underspecified/fragment/repetition; no single reference imposed.')
dump('cases.json',cases)
# Exact source/package identity; the bundle resources are already compiled, so compare
# copied source package weights and token/manifest files with the phone resource sources.
paths=[MOBILE/f for f in ['LiveReelCore.swift','LiveReelModels.swift','LiveReelDecoder.swift','LiveReelStage3.swift','LiveReelEngine.swift']]
paths += [MOBILE/'Models/stage3_tokens.json',MOBILE.parent/'stage3_mobile_naturalizer_manifest_v17.json',ROOT/'active/v17/asl_corpus_v17.py',ROOT/'active/v17/citizen100_manifest.json']
for name in ['Stage3T5EncoderV17.mlpackage','Stage3T5DecoderStepV17.mlpackage']:
 for f in sorted((PACKAGES/name).rglob('*')):
  if f.is_file():
   rel=f.relative_to(PACKAGES);mobile=MOBILE/'Models'/rel
   assert mobile.exists() and sha(f)==sha(mobile),(f,mobile)
   paths.append(f)
for name in ['stage3_tokens.json','stage3_mobile_naturalizer_manifest_v17.json']:
 source=MOBILE/'Models'/name if name=='stage3_tokens.json' else MOBILE.parent/name
 assert sha(PACKAGES/name)==sha(source),name
corpus=ROOT/'data/local/stage3_multisentence_corpus_v17/corpus.jsonl'
recipe=json.loads((ROOT/'artifacts/reports/stage3_multisentence_bakeoff_v17_20260929/tiny/recipe.json').read_text())
assert sha(corpus)==recipe['corpus_sha256']
paths.append(corpus)
dump('identity.json',dict(checkpoint=json.loads((PACKAGES/'stage3_tokens.json').read_text())['checkpoint'],sources={str(p):sha(p) for p in paths},cases=len(cases),families=dict(Counter(r['family'] for r in cases)),scope='Text diagnostic; not ASL video accuracy. No protected visual test, training, API calls or installation.'))
# Stream training corpus, counting label/English realization coverage and exact probe overlap.
stats={w:dict(train_rows=0,word_counts=Counter()) for w in labels};train_keys=Counter();train_examples=defaultdict(list)
for line in corpus.open():
 row=json.loads(line)
 if row['split']!='train':continue
 gs=row['glosses'];train_keys[tuple(gs)]+=1
 for w in set(gs)&set(labels):
  stats[w]['train_rows']+=1
  stats[w]['word_counts'].update(set(_tokens(row['english'])))
 for target in ['food','meal','she','her','mine','yours','ours','job','winter']:
  if target in _tokens(row['english']) and len(train_examples[target])<3:train_examples[target].append({k:row[k] for k in ['glosses','english']})
for r in cases:r['exact_training_matches']=train_keys[tuple(r['glosses'])]
dump('cases.json',cases)
rows=[]
for r in manifest['classes']:
 w=r['canonical_label'];lemmas=list(GLOSS_LEMMAS.get(w,()))
 counts=stats[w]['word_counts'];rows.append(dict(label=w,index=r['class_index'],raw_gloss=r['citizen_raw_gloss'],asl_lex_code=r['citizen_asl_lex_code'],category=r['category'],lemmas=lemmas,train_rows=stats[w]['train_rows'],lemma_hits={a:counts[a] for a in lemmas},selected_word_hits={a:counts[a] for a in ['food','meal','me','my','mine','she','her','his','their','our','job','winter','today','tonight','never'] if counts[a]}))
dump('vocabulary_audit.json',rows);dump('training_examples.json',train_examples)
# Legacy coverage sensitivity/specificity controls; accepted is not semantic correctness.
from scripts.eval_stage3_multisentence_v17 import coverage
controls=[('valid_alias',['EAT','GOOD'],'The food is good.'),('valid_alias',['I','HELP','YOU'],'I help you.'),('role_reversal',['I','HELP','YOU'],'You help me.'),('invented_possession',['I','HAPPY'],'My happiness.'),('wrong_person',['I','HAPPY'],'You are happy.'),('invented_recipient',['I','GIVE','WATER'],'I give her water.'),('invented_recipient',['I','GIVE','WATER'],'I give him water.'),('unlicensed_gender',['HE','HAPPY'],'She is happy.'),('unlicensed_gender',['HE','HAPPY'],'He is happy.'),('wrong_owner',['MY','CHILD','HAPPY'],'Your child is happy.'),('dropped_negation',['I','NO','WANT','WATER'],'I want water.'),('stronger_negation',['I','NO','WANT','WATER'],'I never want water.'),('tense_reversal',['YESTERDAY','I','WORK'],'Tomorrow I work.'),('invented_today',['DAY','I','WORK'],'Today I work.'),('invented_tonight',['NIGHT','I','WORK'],'Tonight I work.')]
dump('coverage_controls.json',[dict(kind=k,glosses=g,english=e,coverage=coverage(g,e)) for k,g,e in controls])
print(json.dumps(dict(cases=len(cases),families=dict(Counter(r['family'] for r in cases)),labels=len(rows),training_overlap=sum(bool(r['exact_training_matches']) for r in cases)),indent=2))
