# iOS-100 utility vocabulary proposal

**Status:** `proposal_requires_asl_review`  
**Basis:** cached ASL Citizen and PopSign metadata; no videos downloaded

This is an accuracy-first, conversational vocabulary. It deliberately keeps high-utility signs that are absent from PopSign instead of optimizing only for the easiest metadata intersection.

## Coverage summary

- Canonical labels: **100**
- Reuse an exact or declared-equivalent current v16 class: **91**
- Covered by ASL Citizen metadata: **98**
- Covered by PopSign game metadata: **44**
- Meet the working 20 train / 5 validation / 5 test estimate: **45/100**
- Require ASL-LEX/raw-label variant review: **12**
- Sum of per-class signer deficits: train **3**, validation **79**, test **0**

The deficit totals are planning units, not necessarily that many unique new people: one newly recorded signer can fill a deficit for many signs. Counts from independent anonymous datasets are added only as a working estimate; cross-dataset identity overlap cannot be proven from public metadata.

## Proposed 100

| Category | Signs |
| --- | --- |
| People And Reference | I, YOU, WE, THEY, HE, MY, YOUR, OUR |
| Questions | WHAT, WHERE, WHEN, WHO, WHY, HOW |
| Social And Safety | HELLO, GOODBYE, PLEASE, THANKYOU, SORRY, YES, NO, MAYBE, HELP |
| Needs And Thought | WANT, NEED, LIKE, LOVE, KNOW, UNDERSTAND, THINK, FEEL, HAVE |
| Actions And Communication | GIVE, TAKE, COME, GO, STOP, WAIT, TRY, USE, MAKE, FIND, LOOK, SEE, HEAR, LISTEN, TALK, SAY, TELL, ASK, ANSWER, LEARN, WORK, READ, WRITE, DRINK, SLEEP |
| Descriptions And States | GOOD, BAD, HAPPY, SAD, ANGRY, EXCITED, TIRED, SICK, HUNGRY, HOT, COLD, BIG, SMALL, MORE, LESS, SAME, DIFFERENT, EASY, IMPORTANT, READY |
| Time And Places | NOW, TOMORROW, YESTERDAY, MORNING, NIGHT, TIME, DAY, WEEK, YEAR, HOME, SCHOOL, HOSPITAL, DOCTOR |
| People And Family | FAMILY, FRIEND, MOTHER, FATHER, CHILD, MAN, WOMAN |
| Language | NAME, SIGN, LANGUAGE |

## Signs still below the working threshold

| Sign | Working train/val/test | Additional train/val/test signers |
| --- | ---: | ---: |
| I | 20/4/11 | 0/1/0 |
| YOU | 22/4/11 | 0/1/0 |
| WE | 23/3/11 | 0/2/0 |
| THEY | 21/3/11 | 0/2/0 |
| HE | 21/4/11 | 0/1/0 |
| MY | 20/4/11 | 0/1/0 |
| YOUR | 22/4/11 | 0/1/0 |
| OUR | 21/4/11 | 0/1/0 |
| WHAT | 22/3/11 | 0/2/0 |
| WHEN | 21/4/11 | 0/1/0 |
| HOW | 23/3/11 | 0/2/0 |
| SORRY | 22/3/10 | 0/2/0 |
| MAYBE | 21/4/11 | 0/1/0 |
| HELP | 19/4/11 | 1/1/0 |
| WANT | 26/4/11 | 0/1/0 |
| NEED | 22/3/11 | 0/2/0 |
| LOVE | 19/4/11 | 1/1/0 |
| KNOW | 21/4/11 | 0/1/0 |
| UNDERSTAND | 21/4/11 | 0/1/0 |
| FEEL | 22/4/11 | 0/1/0 |
| TAKE | 22/3/11 | 0/2/0 |
| COME | 19/3/11 | 1/2/0 |
| STOP | 20/3/11 | 0/2/0 |
| TRY | 21/4/11 | 0/1/0 |
| USE | 21/3/11 | 0/2/0 |
| TELL | 22/3/11 | 0/2/0 |
| ASK | 21/4/11 | 0/1/0 |
| ANSWER | 22/3/11 | 0/2/0 |
| LEARN | 22/4/11 | 0/1/0 |
| WORK | 21/3/11 | 0/2/0 |
| WRITE | 20/3/11 | 0/2/0 |
| GOOD | 22/4/11 | 0/1/0 |
| ANGRY | 21/3/11 | 0/2/0 |
| TIRED | 20/4/11 | 0/1/0 |
| COLD | 20/4/11 | 0/1/0 |
| BIG | 22/4/11 | 0/1/0 |
| SMALL | 20/3/11 | 0/2/0 |
| MORE | 22/4/11 | 0/1/0 |
| LESS | 20/3/11 | 0/2/0 |
| DIFFERENT | 22/4/11 | 0/1/0 |
| EASY | 23/3/11 | 0/2/0 |
| IMPORTANT | 21/3/11 | 0/2/0 |
| READY | 20/3/11 | 0/2/0 |
| DAY | 23/3/10 | 0/2/0 |
| WEEK | 21/4/11 | 0/1/0 |
| YEAR | 22/4/11 | 0/1/0 |
| SCHOOL | 23/3/11 | 0/2/0 |
| HOSPITAL | 23/4/11 | 0/1/0 |
| DOCTOR | 23/4/11 | 0/1/0 |
| FAMILY | 22/3/10 | 0/2/0 |
| FRIEND | 22/3/11 | 0/2/0 |
| WOMAN | 23/4/11 | 0/1/0 |
| NAME | 21/4/11 | 0/1/0 |
| SIGN | 20/4/11 | 0/1/0 |
| LANGUAGE | 21/4/11 | 0/1/0 |

## Mandatory review before training

- Every label and cross-dataset alias must be approved by an ASL-fluent or Deaf reviewer before video download or training.
- ASL Citizen numeric suffixes can represent lexical variants; do not merge them solely because their normalized English gloss matches.
- Review deictic pronouns, especially HE, because isolated pointing signs can depend on discourse location and may not support a stable English-gender class.
- Review likely visual or lexical confusion groups: I/ME, GOODBYE/BYE, MOTHER/MOM, FATHER/DAD, LOOK/SEE, TALK/SAY/TELL, GOOD/THANKYOU, and WANT/LIKE.
- Confirm that the current v16 MAKE_CREATE class is the intended lexical form before reusing it for canonical MAKE.
- PopSign commonly records one-handed mobile variants. Do not merge them with standard two-handed forms without sign-level review.

## Machine-readable detail

See `artifacts/reports/ios100_vocabulary_proposal.csv` for per-source signer counts, aliases, ASL-LEX codes, and deficits for all 100 signs.
