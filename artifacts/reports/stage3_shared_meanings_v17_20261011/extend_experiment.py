import json,sys,itertools,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.eval_stage3_multisentence_v17 import coverage
from active.v17.asl_corpus_v17 import GLOSS_LEMMAS,_tokens
P=Path(__file__).resolve().parent
rows=[]
def add(f,g,**extra):
 gs=g.split();rows.append(dict(id=f'{f}_{sum(r["family"]==f for r in rows):03}',family=f,glosses=gs,scores=[.9]*len(gs),incremental=True,**extra))
for a,b in itertools.permutations(['YESTERDAY','NOW','TOMORROW'],2):
 for v,w in itertools.product(['WORK','SLEEP','EAT','DRINK'],repeat=2):
  add('time_scope',f'{a} I {v} {b} I {w}',first_time=a,second_time=b,first_verb=v,second_verb=w)
for a,b in itertools.permutations(['I','YOU','WE','THEY','HE'],2):
 for order in ['recipient_first','object_first']:
  tail=f'{b} WATER' if order=='recipient_first' else f'WATER {b}'
  add('give_roles',f'{a} GIVE {tail}',subject=a,recipient=b,order=order)
for owner in ['MY','YOUR','OUR']:
 for noun in ['WATER','HOME','SCHOOL','CHILD','FRIEND','EAT']:
  add('predicate_possession',f'{noun} {owner}',owner=owner,noun=noun)
for subj in ['I','YOU','WE','THEY','HE']:
 for q,tail in [('WHAT','WANT'),('WHERE','GO'),('WHEN','WORK'),('WHY','SAD'),('HOW','FEEL'),('WHO','SEE')]:
  add('questions',f'{subj} {tail} {q}',subject=subj,question=q)
for g in ['MY MOTHER SICK HE NEED HELP','MY FATHER SICK HE NEED HELP','WOMAN HAPPY HE WORK SCHOOL','MAN HAPPY HE WORK SCHOOL','MY FRIEND HAPPY HE WORK SCHOOL','MY NAME fs-Alex HE HAPPY','HE HELP HE','HE GIVE HE WATER','THEY HELP THEY','WE HELP WE','I HELP I','YOU HELP YOU']:
 add('reference_ambiguity',g,interpretation='Referent co-reference, gender, role or pronoun-copy unresolved without spatial/nonmanual evidence; not a unique target.')
for a,b in itertools.permutations(['WANT WATER','LIKE EAT','NEED HELP'],2):
 for position in [0,1]:
  add('negation_scope',f'I {"NO " if position==0 else ""}{a} I {"NO " if position==1 else ""}{b}',negative_clause=position)
for v in ['EAT','DRINK','WORK','HELP','ANSWER','SIGN','LOVE','SLEEP','READ','WRITE','GO','LEARN']:
 for prefix in ['I HAVE','I NEED','I WANT']:
  add('noun_verb_frame',prefix+' '+v,focus=v,reading='Verb/noun/obligation ambiguity; do not impose unique interpretation without evidence.')
# Postposed subject/pronoun-copy experiments: deliberately not scored by English SVO.
for a,b in itertools.permutations(['I','YOU','WE','THEY','HE'],2):
 for v in ['HELP','SEE']:
  add('topic_order',f'{b} {a} {v}',subject=a,object=b,interpretation='Candidate object-topic reading; needs nonmanual/spatial evidence for unique roles.')
# Negative lexical-sense controls: do not conflate English homonyms with ASL aliases.
controls=[
 ('unsupported_sense','COLD','I have a cold.'),('supported_context_alias','COLD','Winter.'),
 ('unsupported_sense','LIKE','It is like that.'),('supported_context_alias','LIKE','Enjoy.'),
 ('unsupported_sense','SIGN','Sign the contract.'),('unsupported_sense','WORK','That works.'),
 ('unsupported_sense','HAVE','I have to go.'),('supported_context_alias','WORK','Job.'),
 ('supported_context_alias','EAT','Food.'),('supported_context_alias','DRINK','Beverage.'),
 ('supported_paraphrase','HELP','Assistance.'),('supported_paraphrase','ANSWER','Reply.'),
 ('supported_paraphrase','MAKE','Create.'),('supported_paraphrase','FIND','Discover.'),
 ('supported_paraphrase','UNDERSTAND','Comprehend.'),('supported_paraphrase','IMPORTANT','Significant.'),
 ('supported_paraphrase','READY','Prepared.'),('supported_paraphrase','SAME','Alike.'),
 ('supported_paraphrase','EASY','Simple.'),('supported_paraphrase','HOSPITAL','Hospital.'),
 ('unsupported_role','I','Mine.'),('unsupported_role','YOU','Yours.'),('unsupported_role','WE','Ours.'),
 ('unsupported_role','THEY','Theirs.'),('unsupported_role','HE','His.'),
 ('valid_alias','MY','Mine.'),('valid_alias','YOUR','Yours.'),('valid_alias','OUR','Ours.'),
 ('valid_alias','HE','She.'),('valid_alias','HE','Her.'),('valid_alias','THEY','Them.'),('valid_alias','WE','Us.'),
 ('valid_alias','I','Me.'),('valid_alias','THANKYOU','Thanks.'),('valid_alias','MOTHER','Mom.'),
 ('valid_alias','FATHER','Dad.'),('valid_alias','CHILD','Kid.'),('valid_alias','GOODBYE','Bye.'),
 ('valid_alias','SORRY','Apologize.'),('valid_alias','MAYBE','Perhaps.'),('valid_alias','TRY','Attempt.'),
 ('valid_alias','TALK','Speak.'),('valid_alias','SICK','Ill.'),('valid_alias','SMALL','Little.')]
(P/'extended_cases.json').write_text(json.dumps(rows,indent=2)+'\n')
(P/'extended_coverage_controls.json').write_text(json.dumps([dict(kind=k,glosses=g.split(),english=e,coverage=coverage(g.split(),e)) for k,g,e in controls],indent=2)+'\n')
print('Extended cases:',len(rows),'checker controls:',len(controls))
