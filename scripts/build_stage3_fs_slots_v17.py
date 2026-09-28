"""Stage 3 rows with fingerspelled-word slots (FS0, FS1) so the renderer copies spelled words.

Derived from data/local/stage3_asl_corpus_v17/corpus.jsonl: a person noun phrase ("our father"),
subject HE, or a place (school, hospital, home) is replaced by a slot token in both the glosses and
the English. Each derived row keeps its source row's split; explicit name templates are added per
split. At run time the app substitutes the spelled letters for the slot after generation and falls
back to a literal sentence if the slot is not produced exactly once.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'data/local/stage3_asl_corpus_v17/corpus.jsonl'
OUT = ROOT / 'data/local/stage3_asl_corpus_v17/corpus_fs_slots.jsonl'
MERGED = ROOT / 'data/local/stage3_asl_corpus_v17/corpus_with_fs_slots.jsonl'

PERSON = {'FRIEND': 'friend', 'MOTHER': 'mother', 'FATHER': 'father', 'DOCTOR': 'doctor', 'CHILD': 'child',
          'MAN': 'man', 'WOMAN': 'woman', 'FAMILY': 'family'}
PLACE = {'SCHOOL': 'school', 'HOSPITAL': 'hospital', 'HOME': 'home'}
POSSESSIVE = {'MY', 'YOUR', 'OUR'}
TEMPLATES = [
    (['MY', 'NAME', 'FS0'], 'My name is FS0.'),
    (['YOUR', 'NAME', 'FS0'], 'Your name is FS0.'),
    (['HELLO', 'MY', 'NAME', 'FS0'], 'Hello, my name is FS0.'),
    (['HELLO', 'FS0'], 'Hello, FS0.'),
    (['FS0', 'MY', 'FRIEND'], 'FS0 is my friend.'),
    (['MY', 'FRIEND', 'NAME', 'FS0'], "My friend's name is FS0."),
    (['FS0', 'GOODBYE'], 'Goodbye, FS0.'),
    (['THANKYOU', 'FS0'], 'Thank you, FS0.'),
    (['I', 'GO', 'FS0', 'TOMORROW'], 'I am going to FS0 tomorrow.'),
    (['FS0', 'FS1', 'FRIEND'], 'FS0 and FS1 are friends.'),
    (['MY', 'NAME', 'FS0', 'YOUR', 'NAME', 'WHAT'], 'My name is FS0. What is your name?'),
    (['FS0', 'SICK'], 'FS0 is sick.'),
    (['FS0', 'HAPPY'], 'FS0 is happy.'),
    (['WHERE', 'FS0'], 'Where is FS0?'),
    (['WHO', 'FS0'], 'Who is FS0?'),
]


def replace_once(row, slot):
    """Replace one noun phrase / HE / place with `slot` in glosses and English, or return None."""
    g, e = list(row['glosses']), row['english']
    noise = set(row.get('noise_indices', []))
    options = [i for i, x in enumerate(g) if (x in PERSON or x in PLACE or x == 'HE') and i not in noise
               and not x.startswith('FS')]
    random.shuffle(options)
    for i in options:
        x = g[i]
        if g.count(x) != 1:
            continue
        if x == 'HE':
            if re.search(r'\b(him|his|himself)\b', e, re.I) or len(re.findall(r'\bhe\b', e, re.I)) != 1:
                continue
            pattern = r'\bhe\b'
            start = end = i
        else:
            word = PERSON.get(x) or PLACE[x]
            det = r'(?:(?:my|your|our|the|a)\s+)?'
            if len(re.findall(rf'\b{det}{word}s?\b', e, re.I)) != 1 or re.search(rf'\b{word}s\b', e, re.I):
                continue
            pattern = rf'\b{det}{word}\b'
            found = re.search(pattern, e, re.I).group(0).split()
            spoken = found[0].upper() if len(found) > 1 and found[0].lower() in ('my', 'your', 'our') else None
            before = g[i - 1] if i else None
            # The English possessive must be exactly the gloss right before the noun (or absent on both).
            if (spoken or None) != (before if before in POSSESSIVE else None):
                continue
            start, end = (i - 1, i) if spoken else (i, i)
        english = re.sub(pattern, slot, e, count=1, flags=re.I)
        if english.startswith(slot):
            pass
        elif re.match(rf'^[a-z]', english):
            english = english[0].upper() + english[1:]
        glosses = g[:start] + [slot] + g[end + 1:]
        conf = list(row['confidences'])
        confidences = conf[:start] + [0.9] + conf[end + 1:]
        shift = end - start
        noise_indices = [k if k < start else k - shift for k in row.get('noise_indices', [])]
        return dict(row, glosses=glosses, confidences=confidences, noise_indices=noise_indices, english=english,
                    structure=row['structure'] + '+fs_slot', meaning=row.get('meaning', '') +
                    f' {slot} is a fingerspelled name or word; copy {slot} exactly.')
    return None


def main():
    random.seed(20260928)
    rows = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    derived = []
    for row in rows:
        one = replace_once(row, 'FS0')
        if one is None:
            continue
        derived.append(one)
        two = replace_once(one, 'FS1')
        if two is not None and random.random() < .3:
            derived.append(two)
    for split, repeats in (('train', 12), ('validation', 2), ('test', 2)):
        for glosses, english in TEMPLATES:
            for _ in range(repeats):
                derived.append(dict(glosses=glosses, confidences=[round(random.uniform(.6, .95), 3) for _ in glosses],
                                    noise_indices=[], structure='fs_name_template', meaning='fingerspelled slot template',
                                    english=english, split=split))
    # One split per gloss sequence (the trainer rejects cross-split duplicates): test > validation > train.
    owner = {}
    for r in rows + derived:
        key = ' '.join(r['glosses'])
        rank = {'test': 0, 'validation': 1, 'train': 2}[r['split']]
        owner[key] = min(owner.get(key, 9), rank)
    names = {0: 'test', 1: 'validation', 2: 'train'}
    kept = [r for r in derived if names[owner[' '.join(r['glosses'])]] == r['split']]
    OUT.write_text(''.join(json.dumps(r) + '\n' for r in kept))
    MERGED.write_text(CORPUS.read_text() + ''.join(json.dumps(r) + '\n' for r in kept))
    from collections import Counter
    print(len(kept), Counter(r['split'] for r in kept), Counter('FS1' in r['glosses'] for r in kept))
    for r in kept[:6]:
        print(r['glosses'], '->', r['english'])


if __name__ == '__main__':
    main()
