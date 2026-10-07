# GOOD / THANKYOU variant audit — 2026-10-08

Read-only research for the user's request to train GOOD as the two-handed form (flat hand
from the chin down onto the non-dominant palm) so it separates from THANKYOU. No data,
manifest, model or training was changed. Citizen test was not opened.

## Definitions

- Locked class GOOD = Citizen `GOOD`, ASL-LEX `B_01_052`: **one-handed**, open B, chin, straight
  movement away. THANKYOU = `H_02_053`: one-handed, open B, mouth, curved movement away.
  ASL-LEX has no separate two-handed GOOD entry, so the label cannot separate the variants.
- Switching GOOD to the two-handed form is a variant-contract change for the class (user decision).

## Method

Landmark screen (`analyze.py`, `screen.csv`) then visual review of every GOOD clip on 5-frame
contact sheets (`make_sheets.py`, `sheets/`). Verdicts: `visual_review_GOOD.csv`. The screen agreed
with the visual verdict on 169/179 decided clips; it misses two-handed clips whose base hand is
partly out of frame, so only visual verdicts are counted.

## GOOD clips by variant (visual)

| Source | Split | Two-handed (target) | One-handed | Unclear | Training status |
|---|---|---:|---:|---:|---|
| ASL Citizen | train | 2 | 12 | 1 | approved |
| ASL Citizen | val | 0 | 3 | 1 | validation only |
| SemLex | train | 6 | 13 | 2 | approved train-only |
| SemLex | val | 2 | 9 | 4 | validation only |
| Local deep-clean | train | 22 | 84 | 1 | approved (familiar signers) |
| Local deep-clean | val | 2 | 21 | 0 | validation only |
| MS-ASL gap audit | — | 2 | 1 | 0 | **not training-eligible** |
| **Usable now** | train | **30** | | | |
| | val | **4** | | | |

Local two-handed train clips: 18 from one local signer (blue shirt, who signs it this way every
time) and 4 web/dictionary clips. Local "deep-clean" data therefore includes web dictionary videos.
Sequence sources: GOOD MORNING phrase videos — 12-clip sample shows one local signer (white shirt)
using the two-handed GOOD (4/12), the others one-handed; ASLLRP-segmented (2), O5S5 (2) and
STEM (3) GOOD were not visually reviewed.

## THANKYOU

Available: Citizen 16/3, SemLex 13/5–6, local 119/24, THANKYOU FRIEND phrases 25/20, ASLLRP 3+1,
O5S5 2, PopSign 3 (audit only). All match the single one-handed ASL-LEX form; the landmark screen
found one possible two-handed symmetric clip. Not visually reviewed clip-by-clip.

## Implications

- Only 30 training / 4 validation two-handed GOOD clips exist, concentrated in one local signer.
- Citizen validation has **no** two-handed GOOD, so Citizen GOOD accuracy could not measure the new
  form; a two-handed-only GOOD class would also be judged wrong against the pinned Citizen labels.
- Next safe step if approved: a variant manifest (GOOD = two-handed, one-handed GOOD clips excluded,
  not relabelled), then an isolated retraining with its own validation definition.

## Experiments (2026-10-08, Variant C local replay from 96.83; validation only)

GOOD validation clips grouped by visual verdict; THANKYOU unchanged. Counts are correct/total.

| Model | THANKYOU C / S / L | THANKYOU→GOOD | Two-handed GOOD S / L | One-handed GOOD read as | Totals C / S / L (original labels) |
|---|---|---:|---|---|---|
| 96.83 isolated | 1/3 · 2/6 · 8/24 | 14 | 1/2 · 2/2 | mostly GOOD | 366 / 853 / 1857 |
| Variant C (current) | 1/3 · 2/6 · 23/24 | 5 | 2/2 · 2/2 | mostly GOOD (SemLex 6/9 THANKYOU) | 368 / 854 / 2843 |
| No local GOOD/THANKYOU | 1/3 · 2/6 · 13/24 | 10 | 2/2 · 0/2 | mixed | 368 / 857 / 2823 |
| **Two-handed GOOD only** | **3/3 · 4/6 · 24/24** | **0** | **2/2 · 2/2** | **THANKYOU (all 33)** | 366 / 859 / 2822 |

Excluding one-handed/unclear GOOD clips from scoring (they are out of scope once GOOD is defined as
two-handed): Citizen 364→366 of 374, SemLex 850→858 of 965, local 2823→2822 of 2875
(Variant C → two-handed GOOD). Only 4 two-handed GOOD validation clips exist; 18 of 30 training
clips are one signer. One-handed GOOD will be read as THANKYOU by design.
