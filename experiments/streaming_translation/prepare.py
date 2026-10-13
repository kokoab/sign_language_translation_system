"""Freeze historical evidence and explicit diagnostics without changing sources."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = ROOT / 'artifacts/reports/phone_all_history_review_20260929'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    pairs_path = SOURCES / 'translation_pairs.json'
    pairs = json.loads(pairs_path.read_text())
    sources = {str(pairs_path.relative_to(ROOT)): sha(pairs_path)}
    rows = []
    for i, pair in enumerate(pairs):
        path = SOURCES / 'sessions' / (pair['session'] + '.json')
        sources[str(path.relative_to(ROOT))] = sha(path)
        events = json.loads(path.read_text())['events']
        end = pair['event_index']
        assert events[end]['event'] == 'sentence'
        begin = end - 2
        while begin >= 0 and events[begin]['event'] not in ('sentence', 'reset', 'start'):
            begin -= 1
        words = [e['word'] for e in events[begin+1:end-1] if e['event'] == 'word']
        complete = [w['gloss'] for w in words] == pair['glosses']
        # Use arrival timestamps from original word events, never revised Finish spans.
        rows.append(dict(id=f'phone_{i:03}', family='phone', session=pair['session'],
                         glosses=pair['glosses'], scores=[w['score'] for w in pair['finish_words']],
                         arrivals=[w['commit_seconds'] for w in words] if complete else None,
                         event_complete=complete, finish_timestamp=None,
                         prior_review=pair['review'], historical_english=pair['sentence']))
    # Declared gloss interpretations; not expert-annotated ASL recordings.
    diagnostics = [
        ('greeting', 'HELLO GOOD MORNING HOW YOU', ['Hello, good morning. How are you?', 'Hello, good morning, how are you?']),
        ('experience', 'HELLO YESTERDAY MORNING I FEEL GOOD HOW YOU', ['Hello, I felt good yesterday morning. How are you?']),
        ('friend', 'HELLO MY FRIEND HOW YOU', ['Hello, my friend. How are you?', 'Hello, my friend, how are you?']),
        ('role_i', 'I HELP YOU', ['I help you.', 'I am helping you.']),
        ('role_you', 'YOU HELP I', ['You help me.', 'You are helping me.']),
        ('give_i', 'I GIVE YOU WATER', ['I give you water.', 'I am giving you water.']),
        ('give_you', 'YOU GIVE I WATER', ['You give me water.', 'You are giving me water.']),
        ('owner_my', 'MY CHILD HAPPY', ['My child is happy.']),
        ('owner_your', 'YOUR CHILD HAPPY', ['Your child is happy.']),
        ('past', 'YESTERDAY I GO DOCTOR', ['I went to the doctor yesterday.', 'Yesterday I went to the doctor.']),
        ('future', 'TOMORROW I GO DOCTOR', ['I will go to the doctor tomorrow.', 'Tomorrow I will go to the doctor.', 'I am going to the doctor tomorrow.']),
        ('post_time', 'I GO DOCTOR TOMORROW MORNING', ['I will go to the doctor tomorrow morning.', 'I am going to the doctor tomorrow morning.', 'I go to the doctor tomorrow morning.']),
        ('negative', 'I NO WANT WATER', ['I do not want water.', "I don't want water."]),
        ('positive', 'I WANT WATER', ['I want water.']),
        ('two_scopes', 'YESTERDAY I WORK TOMORROW I SLEEP', ['I worked yesterday. I will sleep tomorrow.', 'Yesterday I worked. Tomorrow I will sleep.']),
        ('negation_scope', 'I NO WANT WATER I WANT EAT', ['I do not want water. I want to eat.', "I don't want water. I want to eat."]),
        ('referent', 'MY FAMILY HAPPY I LOVE THEY', ['My family is happy. I love them.']),
        ('doctor_sleep', 'I NEED DOCTOR I NO SLEEP', ['I need a doctor. I do not sleep.', "I need a doctor. I don't sleep."]),
        ('fragment_i', 'I', []), ('fragment_want', 'I WANT', []),
        ('fragment_owner', 'MY', []), ('fragment_give', 'I GIVE', []),
        ('ambiguous_morning', 'HELLO GOOD MORNING', []),
        ('late_negation', 'I WANT WATER NO', []),
    ]
    for name, gloss, references in diagnostics:
        g = gloss.split()
        rows.append(dict(id=name, family='diagnostic', glosses=g, scores=[.9]*len(g),
                         arrivals=[float(i) for i in range(len(g))], event_complete=True,
                         timing='synthetic_1s', references=references,
                         interpretation='Declared text diagnostic; no unique reference for ambiguous/fragment cases.'))
    # A delayed continuation and a recognition dropout share the same evidence available
    # to a commit-only policy. This is a stress case, not natural ASL timing ground truth.
    for name, gap in [('hesitation', 2.5), ('dropout', 4.0)]:
        rows.append(dict(id=name, family='stress', glosses=['I','WANT','WATER'],
                         scores=[.9]*3, arrivals=[0.,1.,1.+gap], event_complete=True,
                         timing='synthetic_gap', references=['I want water.']))
    assert len({r['id'] for r in rows}) == len(rows)
    for r in rows:
        assert len(r['glosses']) == len(r['scores'])
        if r['arrivals']:
            assert len(r['arrivals']) == len(r['glosses'])
            assert sorted(r['arrivals']) == r['arrivals'], r['id']
    (HERE/'cases.json').write_text(json.dumps(rows, indent=2)+'\n')
    model = ROOT/'artifacts/models/stage3_multisentence_tiny_v17_20260929'
    for path in sorted(model.iterdir()):
        if path.is_file(): sources[str(path.relative_to(ROOT))] = sha(path)
    (HERE/'identity.json').write_text(json.dumps(dict(sources=sources,
        cases_sha256=sha(HERE/'cases.json'), training_seconds=0,
        atlas_writes=False, protected_test_access=False), indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows), phone=len(pairs),
                         timed_phone=sum(r['family']=='phone' and r['event_complete'] for r in rows))))

if __name__ == '__main__': main()
