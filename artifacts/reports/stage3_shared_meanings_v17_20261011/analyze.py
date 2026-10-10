"""Saved-output audit. Narrow mechanical measures are not translation accuracy."""
import collections,csv,hashlib,itertools,json,re,sys
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2];sys.path.insert(0,str(ROOT))
from scripts.eval_stage3_multisentence_v17 import coverage, LEMMAS
from active.v17.asl_corpus_v17 import GLOSS_LEMMAS,FUNCTION_WORDS,_tokens,_covers

def read(n):return json.loads((P/n).read_text())
def dump(n,v):(P/n).write_text(json.dumps(v,indent=2)+'\n')
def ok(c):return not(c['missing'] or c['invented'] or not c['slot_ok'])
rows=[]
for pre in ['', 'extended_', 'phone_']:
 cases=read(pre+'cases.json');results=[json.loads(l) for l in (P/(pre+'results.jsonl')).open()]
 assert len(cases)==len(results) and [c['id'] for c in cases]==[r['id'] for r in results]
 for c,r in zip(cases,results):
  row={**c,**r};row['coverage']=coverage(c['glosses'],r['full']);row['legacy_coverage_pass']=ok(row['coverage']);rows.append(row)
byid={r['id']:r for r in rows}
assert len(byid)==len(rows)==865
ident=read('identity.json')
for path,h in ident['sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
# Exact sequence exposure in the current corpus. Prefixes/subsequences and predecessor
# checkpoints may overlap even when exact whole sequences do not.
train=collections.Counter();globalwords=collections.Counter();exposure={g:collections.Counter() for g in GLOSS_LEMMAS}
for line in (ROOT/'data/local/stage3_multisentence_corpus_v17/corpus.jsonl').open():
 r=json.loads(line)
 if r['split']!='train':continue
 train[tuple(r['glosses'])]+=1
 toks=set(_tokens(r['english']));globalwords.update(toks)
 for g in set(r['glosses'])&set(exposure):exposure[g].update(toks)
for r in rows:r['exact_training_matches']=train[tuple(r['glosses'])]
# Counterfactual validator tests: controlled swaps preserve all content words.
subjects={'I':'I','YOU':'you','WE':'we','THEY':'they','HE':'he'}
objects={'I':'me','YOU':'you','WE':'us','THEY':'them','HE':'him'}
def sverb(g,v):return v+'s' if g=='HE' else v
controls=[]
for a,b in itertools.permutations(subjects,2):
 for v in ['help','see','like','love']:
  gs=[a,v.upper(),b]
  good=f'{subjects[a]} {sverb(a,v)} {objects[b]}.';bad=f'{subjects[b]} {sverb(b,v)} {objects[a]}.'
  controls.append(dict(kind='role_swap',glosses=gs,faithful=good,corrupted=bad,faithful_pass=ok(coverage(gs,good)),corrupted_pass=ok(coverage(gs,bad))))
for a,b in itertools.permutations(['MY','YOUR','OUR'],2):
 for n,m in [('CHILD','FATHER'),('MOTHER','FRIEND'),('DOCTOR','FAMILY')]:
  gs=[a,n,'HELP',b,m];good=f'{a.lower()} {n.lower()} helps {b.lower()} {m.lower()}.';bad=f'{b.lower()} {n.lower()} helps {a.lower()} {m.lower()}.'
  controls.append(dict(kind='owner_swap',glosses=gs,faithful=good,corrupted=bad,faithful_pass=ok(coverage(gs,good)),corrupted_pass=ok(coverage(gs,bad))))
for v,w in itertools.product(['work','sleep','eat','drink'],repeat=2):
 past={'work':'worked','sleep':'slept','eat':'ate','drink':'drank'}
 gs=['YESTERDAY','I',v.upper(),'TOMORROW','I',w.upper()]
 good=f'Yesterday I {past[v]}. Tomorrow I will {w}.';bad=f'Yesterday I will {v}. Tomorrow I {past[w]}.'
 controls.append(dict(kind='tense_swap',glosses=gs,faithful=good,corrupted=bad,faithful_pass=ok(coverage(gs,good)),corrupted_pass=ok(coverage(gs,bad))))
for a,b in itertools.permutations(['want water','like food','need help'],2):
 # Use eating to stay within the *current* vocabulary gate; FOOD is separately audited.
 a=a.replace('food','to eat');b=b.replace('food','to eat')
 gs=['I','NO']+[x.upper() for x in a.split() if x!='to']+['I']+[x.upper() for x in b.split() if x!='to']
 good=f'I do not {a}. I {b}.';bad=f'I {a}. I do not {b}.'
 controls.append(dict(kind='negation_scope_swap',glosses=gs,faithful=good,corrupted=bad,faithful_pass=ok(coverage(gs,good)),corrupted_pass=ok(coverage(gs,bad))))
dump('counterfactual_controls.json',controls)
# Role checks allow tense/modal/question variation; inspect exceptional outputs manually.
verbpatterns={'HELP':'help(?:s|ed)?','SEE':'(?:see|sees|saw|seen)','LIKE':'like(?:s|d)?','LOVE':'love(?:s|d)?'}
for r in rows:
 if r['family']=='roles':
  a,b=r['subject'],r['object'];v=r['focus']
  r['role_realization_pass']=bool(re.search(r'\b'+subjects[a]+r'\s+(?:(?:can|will|do|does|did)\s+)?'+verbpatterns[v]+r'\s+'+objects[b]+r'\b',r['full'],re.I))
 if r['family']=='give_roles':
  toks=_tokens(r['full']);a,b=subjects[r['subject']].lower(),objects[r['recipient']]
  r['role_realization_pass']=a in toks and b in toks and toks.index(a)<toks.index(b) and 'water'in toks
 if r['family']=='possession':r['owner_retained']=r['owner'].lower() in _tokens(r['full'])
 if r['family']=='predicate_possession':
  accepted={'MY':['my','mine'],'YOUR':['your','yours'],'OUR':['our','ours']}[r['owner']]
  r['owner_retained']=any(x in _tokens(r['full']) for x in accepted)
 if r['family']=='time_scope' and {r['first_time'],r['second_time']}=={'YESTERDAY','TOMORROW'}:
  r['manual_time_scope_pass']=r['id'] not in {f'time_scope_{i:03}' for i in [16,17,18,19,20,21,28,29]}
  r['review_note']='Both time markers, action identities, and past/future associations checked against the declared two-clause probe; this is not full-sentence ASL accuracy.'
# Per-class ledger retains exact source identity and gives scope-specific evidence.
audit=read('vocabulary_audit.json')
risk={}
def riskset(names,kind,note):
 for g in names.split():risk[g]=(kind,note)
riskset('I YOU WE THEY HE','reference','Resolve subject/object from sentence structure; spatial referent identity is absent. Do not turn pointing into possessive or infer gender from HE.')
riskset('MY YOUR OUR','possession','Preserve owner; allow my/mine, your/yours, our/ours as syntax requires; never equate with I/YOU/WE.')
riskset('WHAT WHERE WHEN WHO WHY HOW','question','Preserve question focus; nonmanual question marking is absent from Stage3 input.')
riskset('HELLO GOODBYE PLEASE THANKYOU SORRY YES','interaction','Allow conventional English phrasing without inventing an event or participant.')
riskset('NO','negation','Preserve scope; not/never/nothing are not unconditional equivalents.')
riskset('MAYBE WANT NEED HAVE','modality','Preserve possibility, desire, need and possession. HAVE before a verb can provoke unsupported obligation; sign-form evidence may be needed.')
riskset('NOW TOMORROW YESTERDAY MORNING NIGHT TIME DAY WEEK YEAR','time','Preserve time and clause attachment; do not automatically add today/tonight/this-year or reuse another clause tense.')
riskset('GIVE TAKE COME GO HELP TELL ASK ANSWER SEE HEAR LISTEN TALK','roles_and_direction','Check agent/recipient and sense. Direction/locations are not passed to Stage3. English homonyms are not automatically valid sign senses.')
riskset('EAT DRINK WORK SIGN LOVE','noun_verb_or_polysemy','Some noun/verb readings may share a sign; verify exact raw sign identity and context before admitting aliases. WORD string is not the full sign meaning.')
riskset('COLD HOT LIKE','homonym_trap','Do not infer English homonym equivalence: COLD temperature vs illness; LIKE affinity vs similarity; HOT temperature vs other English senses. COLD/winter is context-dependent.')
riskset('MORE LESS SAME DIFFERENT','comparison','Preserve comparison, quantity and referents; English phrase choice depends on context.')
riskset('CHILD MAN WOMAN FAMILY FRIEND MOTHER FATHER DOCTOR','entity','Allow reviewed lexical variants and number; do not invent ownership, gender beyond evidence, or co-reference.')
notes={
 'EAT':'FOOD is omitted from legacy lemma map; current train corpus has no food target. Noun-food probes fail while verb-eat works. Exact Citizen form EAT1; visual FOOD coverage untested.',
 'DRINK':'Both noun and verb English observed; beverage is a candidate paraphrase, not a license for different alcoholic-drink handshapes.',
 'WORK':'Work/job can overlap in context. WORK meaning labor must not absorb English work=operate/succeed.',
 'COLD':'Source: temperature sign may mean winter in context, not illness/head-cold. No blanket English-synonym expansion.',
 'LIKE':'Source: affinity/enjoyment; similar/same uses a different sign.',
 'SIGN':'Signing/sign are possible context readings; signature/contract is a separate sense/sign.',
 'HE':'Evaluation map extends she/her/hers, base corpus map only he/him/his. Current output favors he. Input lacks gender/referent identity.',
 'TALK':'Isolated TALK becomes Ask; contextual TALK is retained.',
 'I':'Raw Citizen gloss ME maps to I. Isolated I becomes Is it here?; context generally restores first person.'}
for r in audit:
 g=r['label'];kind,note=risk.get(g,('lexical_and_inflection','Preserve lexical content; allow ordinary English inflections and independently reviewed paraphrases. No new shared-sign equivalence established.'))
 r.update(risk_family=kind,policy_note=notes.get(g,note),context=byid[f'context_{r["index"]:03}']['full'],isolated=byid[f'isolated_{r["index"]:03}']['full'],wrapped=byid[f'wrapped_{r["index"]:03}']['full'])
 r['context_input']=' '.join(byid[f'context_{r["index"]:03}']['glosses'])
 r['singleton_focus_retained']=any(_covers(t,LEMMAS[g]) for t in _tokens(r['isolated']))
 r['context_coverage_pass']=byid[f'context_{r["index"]:03}']['legacy_coverage_pass']
dump('vocabulary_audit.json',audit)
with (P/'vocabulary_audit.csv').open('w',newline='') as f:
 fields=['index','label','raw_gloss','asl_lex_code','risk_family','lemmas','train_rows','isolated','context_input','context','wrapped','policy_note']
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(audit)
inc=[r for r in rows if isinstance(r.get('incremental'),str)]
summary=dict(total_cases=len(rows),unique_gloss_sequences=len({tuple(r['glosses']) for r in rows}),vocabulary_classes=len(audit),families=dict(collections.Counter(r['family'] for r in rows)),native_modes=dict(collections.Counter(r['mode'] for r in rows)),incremental_cases=len(inc),incremental_differences=[r['id'] for r in inc if r['full']!=r['incremental']],incremental_cases_with_locks=sum(r['locked']>0 for r in inc),training_overlap=sum(r['exact_training_matches']>0 for r in rows),singleton_focus_missing=[r['label'] for r in audit if not r['singleton_focus_retained']],role_realization={f:dict(total=sum(r['family']==f for r in rows),passed=sum(r.get('role_realization_pass',False) for r in rows if r['family']==f)) for f in ['roles','give_roles']},possession_owner_retained=sum(r.get('owner_retained',False) for r in rows if r['family']=='possession'),predicate_owner_missing=[r['id'] for r in rows if r['family']=='predicate_possession' and not r['owner_retained']],past_future_scope=dict(total=sum('manual_time_scope_pass'in r for r in rows),failed=[r['id'] for r in rows if r.get('manual_time_scope_pass') is False]),counterfactual_gate={f:dict(total=sum(r['kind']==f for r in controls),faithful_accepted=sum(r['kind']==f and r['faithful_pass'] for r in controls),corrupted_accepted=sum(r['kind']==f and r['corrupted_pass'] for r in controls)) for f in sorted({r['kind'] for r in controls})},global_training_word_rows={w:globalwords[w] for w in ['food','meal','job','winter','beverage','assistance','she','her','he','him']},training_rows=sum(train.values()),notes=['Counts are diagnostic, not general ASL accuracy. No video inference or independent fluent-signer annotation. Some probes are intentionally ambiguous.','Current-corpus exact overlaps are counted; absence of overlap does not establish independence from predecessor checkpoints.','Confidence .2 is an injected stress case below the .4 recognizer threshold, not evidence that such words normally commit.'])
dump('summary.json',summary);dump('reviewed_results.json',rows)
with (P/'results.csv').open('w',newline='') as f:
 fields=['id','family','glosses','scores','focus','reading','full','incremental','locked','mode','legacy_coverage_pass','exact_training_matches']
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2))
