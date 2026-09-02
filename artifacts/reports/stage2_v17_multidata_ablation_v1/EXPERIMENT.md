# Stage 2 public-data single-source and combined ablation

**Run date:** 2026-09-01 (Asia/Manila)  
**Decision:** retain the conservative combined checkpoint as a research candidate;
do not promote it for app deployment or autonomous phrase generation

## Data state

No duplicate download was needed. Every currently lawful and usable source identified
by the online survey was already present locally and hash/audit backed:

| Source | Local material | Supervision used here |
| --- | ---: | --- |
| 2M-Flores-ASL `dev` | 155 selected sentences; 1.3 GiB video; 127 train / 28 train-only validation for temporal pretraining | label-free masked temporal reconstruction; ordered gloss strings were not asserted as exact ASL-LEX variants |
| ASLLRP `OTHER`-CTC | 1,104 target-bearing spans; 879 train / 225 signer-held-out validation; 1.8 GiB video | exact pinned target glosses plus explicit `OTHER` context |
| NCSLGR public subset | 166 utterances, two native signers, 17/100 lexical-label coverage | transition-only; no compatible exact-variant CTC targets in the current artifacts |
| How2Sign train subset | 1,027 acquired clips, 1,026 usable, six source signer IDs | transition/self-supervision only; public artifacts do not provide ordered gloss targets for this decoder |

DSP video is unavailable under the currently published BU download terms, the Apple
ASL STEM Wiki annotation bundle was not located, and ASL Homework requires authorized
Databrary access. They were not downloaded or treated as training data.

## Matched test

Both supervised runs used seed 1701, the same 879 ASLLRP training spans, old real and
train-only synthetic replay, source sampling masses, frozen backbone, learning rates,
10 epochs, and validation selector. The only experimental difference was whether the
student began from the original Stage-2 checkpoint or from a 50% interpolation toward
the 2M-Flores temporal-pretrained encoder. Distillation always used the unchanged
original Stage-2 model as teacher.

| Validation metric | ASLLRP single source | 2M-Flores + ASLLRP | Difference |
| --- | ---: | ---: | ---: |
| New ASLLRP full sequence | 617/682 edits (90.47% WER), 0/225 exact | **542/682 edits (79.47% WER), 4/225 exact** | **75 fewer edits; 12.2% relative edit reduction** |
| New ASLLRP target-only | 450/284 edits (158.45% WER), 14/225 exact | **365/284 edits (128.52% WER), 25/225 exact** | **85 fewer edits; 18.9% relative edit reduction; 11 more exact** |
| Older ASLLRP phrase gate | 11/24 edits | 11/24 edits | preserved |
| Local phrase gate | 7/259 edits | **6/259 edits** | one fewer edit |
| Selected epoch | 5 | 6 | — |

WER above 100% is possible because insertions are counted. The combined checkpoint
passed the predeclared legacy guard (`local <= 7` and older ASLLRP `<= 11`). Both saved
checkpoints reproduced their recorded metrics exactly after a cold CPU reload. No
Citizen, SemLex, local sealed, RIT, 2M-Flores `devtest`, How2Sign validation, or
How2Sign test split was accessed.

The 2M-Flores-only experiment remains a different-task control: its CTC head was
frozen, and its selected train-only validation reconstruction score was 0.15054 at
epoch 13. A previous direct replay evaluation improved the contextual diagnostic from
43 to 41 edits but regressed the older ASLLRP gate from 11 to 12, so full-strength
transfer was rejected. In this experiment, 25%, 50%, and 75% interpolation screens
showed that 50% was the strongest initialization that preserved both legacy gates
before supervised training. Full-strength initialization again regressed them.

## Interpretation and consultation point

The improvement establishes that genuine sentence motion from 2M-Flores transfers to
the exact-variant ASLLRP task. It also establishes that the main problem is not simply
missing downloads: the ordinary high-rate joint fine-tune forgot old phrases, whereas
the conservative frozen-backbone recipe preserved them. The remaining 79.47% full WER
and 1.78% exact-sequence accuracy are still far below a usable continuous recognizer.
This is both an objective/architecture problem and a supervision-diversity problem.

Do not use the checkpoint to mass-produce self-labeled training truth. The next bounded
experiment should keep the genuine 30-phrase, 13-native-signer collection as the
validation anchor and use How2Sign/NCSLGR only to regularize transition dynamics.
Generated combinations should first be exported as review candidates with explicit
source spans, confidence, continuity metrics, and a `synthetic` flag. Only native-
reviewed samples may be admitted to training, and none may enter validation or test.

## Artifacts

- Combined checkpoint: `artifacts/models/stage2_v17_2m_asllrp_other_ctc_conservative_v1/best_model.pth`
  (SHA-256 `20ea6a6952ef322b5d8144cba936f34f5ef94bf1c3cd85ea28778d3be3ceb3ca`)
- ASLLRP-only checkpoint: `artifacts/models/stage2_v17_asllrp_other_ctc_conservative_single_v1/best_model.pth`
  (SHA-256 `58ff4f2a41c4261fdd66346840fab475f3a6f6981ca756f1a5e129d528172ea6`)
- 2M-Flores temporal checkpoint: `artifacts/models/stage2_v17_2m_flores_temporal_pretrain_v1/temporal_pretrained.pth`
  (SHA-256 `ba543b1fd9fa9dd5827b5c67b5f4b0b4a52748187d45513b17a87d64274519ab`)

