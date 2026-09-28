"""Text-only Stage 3 composition repair. Generated examples are training-only.

No visual/phrase archives are loaded. Existing non-train sequences stay reserved.
Training uses a fixed, declared schedule, not synthetic validation or test selection.
Probes are development diagnostics; no generated held-out accuracy is claimed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'data/local/stage3_asl_corpus_v17/corpus_with_fs_slots.jsonl'
INIT = ROOT / 'artifacts/models/stage3_v17_asl_order_fs_v1'
REPORT = ROOT / 'artifacts/reports/stage3_composition_v17_20260929'
OUT = ROOT / 'artifacts/models/stage3_composition_v17_20260929'
SEED = 17929
USER_PROBES = {'HELLO MY FRIEND HOW YOU', 'HELLO GOOD MORNING HOW YOU FRIEND'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def key(row):
    return ' '.join(row['glosses'])


def make_row(gloss, english, family, rng):
    g = gloss.split()
    return dict(glosses=g, english=english, split='train', noise_indices=[],
                confidences=[round(rng.uniform(.60, .99), 3) for _ in g],
                structure='composition_' + family, provenance='generated_training_only')


def build_rows(source):
    rng = random.Random(SEED)
    reserved = {key(r) for r in source if r['split'] != 'train'} | USER_PROBES
    replay = [r for r in source if r['split'] == 'train']
    if any(key(r) in reserved for r in replay):
        raise ValueError('existing training rows overlap reserved sequences')
    added = {}

    def add(g, e, family):
        if g not in reserved and len(g.split()) <= 15:
            added.setdefault(g, make_row(g, e, family, rng))

    greetings = [('HELLO', 'Hello'), ('GOOD MORNING', 'Good morning'),
                 ('GOOD DAY', 'Good day'), ('HELLO GOOD MORNING', 'Hello, good morning'),
                 ('HELLO GOOD DAY', 'Hello, good day'), ('GOOD NIGHT', 'Good night')]
    addresses = [('FRIEND', 'friend'), ('MY FRIEND', 'my friend'),
                 ('MOTHER', 'Mother'), ('FATHER', 'Father'), ('DOCTOR', 'Doctor'),
                 ('FS0', 'FS0'), ('MY CHILD', 'my child')]
    questions = [('HOW YOU', 'How are you?'), ('HOW YOU FEEL', 'How do you feel?'),
                 ('HOW YOUR DAY', 'How is your day?'), ('WHAT YOUR NAME', 'What is your name?'),
                 ('YOUR NAME WHAT', 'What is your name?'), ('WHAT YOU WANT', 'What do you want?'),
                 ('WHAT YOU NEED', 'What do you need?'), ('WHERE YOU GO', 'Where are you going?'),
                 ('WHEN YOU GO HOME', 'When are you going home?'),
                 ('HOW YOUR FAMILY', 'How is your family?'),
                 ('WHY YOU SAD', 'Why are you sad?'), ('WHY YOU ANGRY', 'Why are you angry?'),
                 ('HOW YOUR WORK', 'How is your work?'), ('WHAT YOU THINK', 'What do you think?'),
                 ('WHERE YOUR MOTHER', 'Where is your mother?'),
                 ('WHERE YOUR FATHER', 'Where is your father?')]
    states = [(f'I {s}', f'I am {s.lower()}.') for s in
              ['HAPPY', 'HUNGRY', 'TIRED', 'SICK', 'COLD', 'ANGRY', 'SAD', 'READY', 'EXCITED']]
    requests = [('I NEED HELP', 'I need help.'), ('I NEED WATER', 'I need water.'),
                ('PLEASE HELP', 'Please help.'), ('PLEASE WAIT', 'Please wait.'),
                ('I WANT WATER', 'I want water.'), ('I NEED MORE TIME', 'I need more time.'),
                ('PLEASE LISTEN', 'Please listen.'), ('I LOVE YOU', 'I love you.'),
                ('THANKYOU', 'Thank you.'), ('SORRY', 'Sorry.')]
    for g, e in greetings + questions + states + requests:
        add(g, e if e.endswith(('.', '?')) else e + '.', 'atomic')
    for (g, e), (a, ae) in itertools.product(greetings, addresses):
        add(g + ' ' + a, e + ', ' + ae + '.', 'address')
        for q, qe in questions:
            add(f'{g} {a} {q}', f'{e}, {ae}. {qe}', 'greeting_address_question')
            add(f'{g} {q} {a}', f'{e}. {qe[:-1]}, {ae}?', 'greeting_question_address')
        for q, qe in states + requests:
            add(f'{g} {a} {q}', f'{e}, {ae}. {qe}', 'greeting_address_statement')
    for (q, qe), (a, ae) in itertools.product(questions, addresses):
        add(f'{q} {a}', f'{qe[:-1]}, {ae}?', 'question_address')
    for (g, e), (q, qe) in itertools.product(greetings, questions + states + requests):
        add(f'{g} {q}', f'{e}. {qe}', 'greeting_clause')

    # Explicit time attachment across subject, destination, time and order.
    people = [('I', 'I', 'am'), ('YOU', 'You', 'are'), ('WE', 'We', 'are'),
              ('THEY', 'They', 'are'), ('HE', 'He', 'is'), ('MY MOTHER', 'My mother', 'is'),
              ('MY FATHER', 'My father', 'is'), ('MY FRIEND', 'My friend', 'is'), ('FS0', 'FS0', 'is')]
    places = [('DOCTOR', 'to the doctor'), ('HOSPITAL', 'to the hospital'),
              ('SCHOOL', 'to school'), ('HOME', 'home'), ('WORK', 'to work')]
    times = [('TOMORROW', 'tomorrow', False), ('TOMORROW MORNING', 'tomorrow morning', False),
             ('TOMORROW NIGHT', 'tomorrow night', False), ('NOW', 'now', False),
             ('YESTERDAY', 'yesterday', True), ('YESTERDAY MORNING', 'yesterday morning', True),
             ('YESTERDAY NIGHT', 'last night', True)]
    motions = []
    for (s, se, be), (p, pe), (t, te, past) in itertools.product(people, places, times):
        e = f'{se} ' + ('went' if past else f'{be} going') + f' {pe} {te}.'
        for g in (f'{s} GO {p} {t}', f'{t} {s} GO {p}'):
            add(g, e, 'time_attachment'); motions.append((g, e))
    for (g, e), (m, me) in itertools.product(greetings, motions):
        add(f'{g} {m}', f'{e}. {me}', 'greeting_time')
    name = ('MY NAME FS0', 'My name is FS0.')
    for (g, e), (s, se) in itertools.product(greetings, states + requests + questions):
        add(f'{g} {name[0]} {s}', f'{e}. {name[1]} {se}', 'name_clause')
        add(f'{g} {s} {name[0]}', f'{e}. {se} {name[1]}', 'clause_name')
    for s, se in states + requests + questions:
        add(f'{s} {name[0]}', f'{se} {name[1]}', 'clause_name')
        add(f'{name[0]} {s}', f'{name[1]} {se}', 'name_clause')

    # Diverse combinations from established TRAIN examples, retaining noise supervision
    # in replay but composing only clean, slot-free examples.
    clean = [r for r in replay if not r.get('noise_indices') and len(r['glosses']) <= 6
             and not any(g.startswith('FS') for g in r['glosses'])]
    for _ in range(6500):
        a, b = rng.sample(clean, 2)
        add(key(a) + ' ' + key(b), a['english'].rstrip('.!?') + '. ' + b['english'], 'two_clauses')
    for _ in range(3500):
        r = rng.choice(clean); g, e = rng.choice(greetings)
        add(g + ' ' + key(r), e + '. ' + r['english'], 'greeting_replay')
    return replay, list(added.values()), reserved


def prepare():
    REPORT.mkdir(parents=True, exist_ok=True)
    source = [json.loads(l) for l in SOURCE.open() if l.strip()]
    replay, added, reserved = build_rows(source)
    rows = replay + added
    path = REPORT / 'train.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    manifest = dict(format='stage3_text_composition_training_v1', seed=SEED,
        initialization=str(INIT.relative_to(ROOT)), source_sha256=digest(SOURCE),
        train_sha256=digest(path), replay_rows=len(replay), added_rows=len(added),
        families=dict(Counter(r['structure'] for r in added)),
        reserved_sequence_overlap=len({key(r) for r in rows} & reserved),
        user_probe_training_overlap=sorted({key(r) for r in rows} & USER_PROBES),
        schedule=dict(epochs=5, batch_size=32, learning_rate=.0002, device='mps', seed=SEED),
        selection='fixed final epoch; no generated validation/test or checkpoint selection',
        scope='text-only Stage3; no visual data, Stage1/2 training or phrase-gate bypass',
        runtime='neural generation; reviewed exact-phrase overrides disabled')
    (REPORT / 'recipe.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k:manifest[k] for k in ('replay_rows','added_rows','reserved_sequence_overlap','user_probe_training_overlap')}))


def train():
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Adafactor
    from active.v17.stage3_asl_encoding_v17 import encode
    recipe = json.loads((REPORT / 'recipe.json').read_text())
    path = REPORT / 'train.jsonl'
    assert digest(path) == recipe['train_sha256']
    assert recipe['reserved_sequence_overlap'] == 0
    assert recipe['user_probe_training_overlap'] == []
    rows = [json.loads(l) for l in path.open()]
    assert all(r['split'] == 'train' for r in rows)
    torch.set_num_threads(4); torch.manual_seed(SEED); random.seed(SEED)
    assert torch.backends.mps.is_available(), 'MPS is required for this recipe'
    tok = AutoTokenizer.from_pretrained(INIT, local_files_only=True)
    source = tok([encode(r['glosses'], r['confidences']) for r in rows], padding=True, return_tensors='pt')
    target = tok([r['english'] for r in rows], padding=True, return_tensors='pt').input_ids
    assert source.input_ids.shape[1] <= 64 and target.shape[1] <= 64, 'refuse silent truncation'
    target[target == tok.pad_token_id] = -100
    encoded = [{k:v[i] for k,v in dict(input_ids=source.input_ids, attention_mask=source.attention_mask, labels=target).items()} for i in range(len(rows))]
    model = AutoModelForSeq2SeqLM.from_pretrained(INIT, local_files_only=True).to('mps')
    opt = Adafactor(model.parameters(), lr=recipe['schedule']['learning_rate'], relative_step=False, scale_parameter=False, warmup_init=False)
    loader = DataLoader(encoded, batch_size=recipe['schedule']['batch_size'], shuffle=True)
    start = time.monotonic(); history=[]
    for epoch in range(recipe['schedule']['epochs']):
        model.train(); losses=[]
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss=model(**{k:v.to('mps') for k,v in batch.items()}).loss
            if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        entry=dict(epoch=epoch+1, mean_train_loss=sum(losses)/len(losses), elapsed_seconds=time.monotonic()-start)
        history.append(entry); print(json.dumps(entry), flush=True)
    OUT.mkdir(parents=True, exist_ok=True); model.to('cpu').save_pretrained(OUT); tok.save_pretrained(OUT)
    contract=json.loads((INIT/'stage3_input_contract.json').read_text())
    contract.update(reviewed_templates_enabled=False, utterance_segmentation='model', recipe_sha256=digest(REPORT/'recipe.json'), training_corpus_sha256=digest(path))
    (OUT/'stage3_input_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    (REPORT/'training_result.json').write_text(json.dumps(dict(recipe=recipe, history=history, seconds=time.monotonic()-start, checkpoint=str(OUT), weights_sha256=digest(OUT/'model.safetensors')),indent=2)+'\n')


def probe():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from active.v17.stage3_asl_encoding_v17 import encode
    from active.v17.train_stage3_asl_v17 import generate
    torch.set_num_threads(4)
    cases = [dict(glosses=s.split(), origin='user_requested', confidences=[.9]*len(s.split())) for s in sorted(USER_PROBES)]
    history=json.loads((ROOT/'artifacts/reports/phone_all_history_review_20260929/translation_pairs.json').read_text())
    for r in history:
        slots=[]; g=[]
        for token in r['glosses']:
            if token.startswith('fs-'): g.append('FS'+str(len(slots))); slots.append(token)
            else:g.append(token)
        cases.append(dict(glosses=g,confidences=[w['score'] for w in r['finish_words']],origin='phone_history',phone_output=r['sentence'],session=r['session'],number=r['number'],slots=slots))
    for s in ['HELLO GOOD MORNING HOW YOUR FAMILY','GOOD DAY MY FRIEND HOW YOU','HELLO DOCTOR I NEED HELP','HELLO MY FRIEND I NEED WATER','GOOD MORNING HOW YOU FEEL FRIEND','I GO DOCTOR TOMORROW MORNING','I SICK MY NAME FS0','MY NAME FS0 I HUNGRY','I NO WANT WATER','YOU NO NEED HELP','WE LOVE OUR FAMILY','PLEASE GIVE MY CHILD WATER','HOW YOU FRIEND','MY FRIEND SICK','HELLO FS0 HOW YOU','HELLO GOOD MORNING MY NAME FS0 HOW YOU']:
        cases.append(dict(glosses=s.split(), confidences=[.9]*len(s.split()),origin='generated_diagnostic_only'))
    trainkeys={key(json.loads(l)) for l in (REPORT/'train.jsonl').open()}
    for name,path in [('baseline',INIT),('candidate',OUT)]:
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        model=AutoModelForSeq2SeqLM.from_pretrained(path,local_files_only=True).eval()
        outputs=generate(model,tok,[encode(r['glosses'],r['confidences']) for r in cases],torch.device('cpu'),16)
        for r,o in zip(cases,outputs):r[name]=o;r['in_training']=key(r) in trainkeys
        del model
    (REPORT/'probes.json').write_text(json.dumps(cases,indent=2)+'\n')
    for r in cases:print(json.dumps({k:r[k] for k in ('glosses','baseline','candidate','in_training')}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','probe'])
    p.add_argument('--report', type=Path, default=REPORT)
    p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--init', type=Path, default=INIT)
    args=p.parse_args();REPORT=args.report;OUT=args.output;INIT=args.init
    {'prepare':prepare,'train':train,'probe':probe}[args.action]()
