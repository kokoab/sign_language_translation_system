from pathlib import Path
import json, random, itertools, hashlib, sys
root=Path('/Volumes/secret/SLT/SLT');sys.path.insert(0,str(root))
from scripts.repair_stage3_composition_v17 import build_rows, make_row, key, digest, SOURCE, SEED, USER_PROBES
base=root/'artifacts/reports/stage3_composition_v17_20260929'
out=base/'request_continuation';out.mkdir(exist_ok=True)
source=[json.loads(l) for l in SOURCE.open()]
reserved={key(r) for r in source if r['split']!='train'}|USER_PROBES
rng=random.Random(SEED);new={}
def add(g,e,f):
 if g not in reserved:new.setdefault(g,make_row(g,e,f,rng))
recipients=[(f'{p} {n}',f'{p.lower()} {n.lower()}') for p,n in itertools.product(['MY','YOUR','OUR'],['CHILD','FRIEND','MOTHER','FATHER','FAMILY'])]
objects=[('WATER','water'),('MORE WATER','more water'),('TIME','time'),('MORE TIME','more time'),('HELP','help'),('MORE HELP','more help')]
people=[('I','I',False),('YOU','You',False),('WE','We',False),('THEY','They',False),('HE','He',True),('MY MOTHER','My mother',True),('MY FATHER','My father',True)]
greetings=[('HELLO','Hello'),('GOOD MORNING','Good morning'),('GOOD DAY','Good day'),('HELLO GOOD MORNING','Hello, good morning')]
for (r,re),(o,oe) in itertools.product(recipients,objects):
 for pre,pe in [('', ''),('PLEASE ','Please ')]:
  g=f'{pre}GIVE {r} {o}';e=f'{pe}give {re} {oe}.';e=e[0].upper()+e[1:];add(g,e,'recipient_request')
  for h,he in greetings:add(f'{h} {g}',f'{he}. {e}','greeting_recipient_request')
 for s,se,third in people:
  if s==r:continue
  add(f'{s} GIVE {r} {o}',f'{se} '+('gives' if third else 'give')+f' {re} {oe}.','recipient_statement')
  for t,te,verb in [('TOMORROW','tomorrow','will give'),('YESTERDAY','yesterday','gave')]:
   for g in (f'{s} GIVE {r} {o} {t}',f'{t} {s} GIVE {r} {o}'):
    add(g,f'{se} {verb} {re} {oe} {te}.','recipient_time')
for r,re in recipients:
 for c,ce in [('I NEED HELP','I need help'),('I HUNGRY','I am hungry'),('I SICK','I am sick'),('WE GO HOME TOMORROW','we are going home tomorrow')]:
  add(f'PLEASE TELL {r} {c}',f'Please tell {re} {ce}.','recipient_message')
addresses=[('FRIEND','friend'),('MY FRIEND','my friend'),('MOTHER','Mother'),('FATHER','Father'),('DOCTOR','Doctor'),('FS0','FS0')]
for (h,he),(a,ae),(s,se) in itertools.product(greetings,addresses,[('I HELP','I help.'),('I LISTEN','I listen.'),('I UNDERSTAND','I understand.'),('I WAIT','I wait.'),('I NEED HELP','I need help.'),('I NEED WATER','I need water.')]):
 add(f'{h} {a} {s}',f'{he}, {ae}. {se}','address_action')
# Replay four thousand existing TRAIN rows, balanced between original and new composition data.
prior=[json.loads(l) for l in (base/'train.jsonl').open()]
a=[r for r in prior if not r.get('provenance')];b=[r for r in prior if r.get('provenance')]
replay=rng.sample(a,2000)+rng.sample(b,2000)
rows=replay+list(new.values());assert not ({key(r) for r in rows}&reserved)
path=out/'train.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
recipe=dict(format='stage3_text_request_continuation_v1',seed=SEED,initialization='artifacts/models/stage3_composition_v17_20260929',train_sha256=digest(path),replay_rows=len(replay),added_rows=len(new),reserved_sequence_overlap=0,user_probe_training_overlap=[],schedule=dict(epochs=3,batch_size=32,learning_rate=.0001,device='mps',seed=SEED),selection='fixed final epoch; generated data training-only, not validation/model-selection truth',scope='text-only model update for recipient/object roles; no runtime phrase triggers')
(out/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n');print(json.dumps(recipe,indent=2))
