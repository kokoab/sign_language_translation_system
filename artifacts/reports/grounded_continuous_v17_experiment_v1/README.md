# Grounded continuous recognition experiment (v17)

## Outcome

The experiment is complete, and it answers the immediate question: **better continuous
supervision helps, but the present phrase holdings are not sufficient to replace the
stable isolated live path yet.** The selected causal model does not assume that every
window is a phrase: it can emit CTC blank, one of the 100 locked glosses, or explicit
`OTHER`. It uses only past/current evidence.

Timed sign boundaries from the local NCSLGR subset improved the matched three-seed
composite score in all three runs. Mean NCSLGR WER improved from 86.67% to 80.00%, mean
local signer-held-out WER from 39.44% to 39.07%, and isolated exact accuracy from 81.27%
to 82.47%. The mean false-emission rate on transition/background windows worsened from
13.64% to 15.44%, so boundary rejection remains the main failure mode. Mean exact phrase
accuracy is only 20.75%; this is research evidence, not demo-ready accuracy.

| Matched three-seed mean | CTC only | + timed alignment |
| --- | ---: | ---: |
| Composite selection score (lower is better) | 0.4058 | **0.3922** |
| All exact phrases | 19.95% | **20.75%** |
| All phrase WER | 43.59% | **42.67%** |
| Local held-out signer exact | 21.50% | **23.00%** |
| Local held-out signer WER | 39.44% | **39.07%** |
| NCSLGR WER | 86.67% | **80.00%** |
| Isolated exact | 81.27% | **82.47%** |
| Transition false emission | **13.64%** | 15.44% |

The selected seed-17081 checkpoint obtained 25.5% exact / 37.04% WER on the 200 local
held-out-signer phrases, 41.67% WER on 12 exact ASLLRP phrases, 78% WER on 37 NCSLGR
utterances, 82.37% exact isolated recognition, and 11.36% transition false emissions.
It is the best balanced research checkpoint, not a replacement for the current live
model.

## Data actually used

- Local phrases: 780 raw recordings across nine exact sequences. Automated face audit
  found three phrase-recording identities. Two identities train; one identity validates.
- ASLLRP: 44 train and 12 validation exact contiguous spans. These remain useful but
  are small and visually/domain-limited.
- NCSLGR: 125 utterances with at least one exact raw-gloss match to the locked 100-class
  vocabulary; 88 from one participant train and 37 from the other validate. Published
  SignStream intervals provide the only direct frame-level transition supervision.
- Isolated replay and explicit blank/transition windows keep the model from treating
  all activity as a multi-gloss phrase.

How2Sign/OpenASL/other downloaded videos were not assigned invented gloss targets. They
can support self-supervised motion or background pretraining later, but the available
How2Sign package has no reliable gloss annotation for this exact 100-class evaluation.

## Signer correction

The wider local holdings do contain seven or more visible people. The nine phrase
folders used here do not: their 780 clips repeatedly show three identities. Forcing
seven clusters splits the same green-shirt signer by framing/expression and produces
2/1/1-video pseudo-signers. The automatic 2-to-10 comparison selects three clusters
(sizes 300/298/182, silhouette 0.7371). This matters because calling those fragments
new signers would leak the same person into training and validation.

## Stage-1 adaptation result

A separate, retention-gated Stage-1 fine-tune used mild spatial/temporal noise, carried
frame gaps, isolated replay, and precise phrase cores. It improved held-out local cores
from 91.48% to 92.59%, ASLLRP cores from 83.33% to 87.50%, and NCSLGR cores from 10% to
24%. Citizen validation changed from 94.18% to 93.39% and the other isolated validation
set from 85.79% to 84.36%.

That adapted checkpoint made the end-to-end streaming decoder worse in the matched
ablation. It is therefore retained as an experiment and **not promoted** to the live
prototype. More isolated noise alone is not the missing solution; correctly timed
continuous targets and background/UNKNOWN rejection are.

## What to do with the incoming 30 phrases

Record all 13 native signers naturally, without pauses inserted between glosses. A few
deliberately slower examples are useful, but the main distribution should be normal
conversational signing. Keep the same signer entirely within one split (proposed:
10 train, 2 validation, 1 sealed test).

For each performance, retain one phrase-level gloss sequence and boundary timestamps
for each gloss. Automatic motion/Stage-1 proposals can create the first-pass boundaries,
but a person should correct them. Three synchronized camera views are correlated views
of one performance, not three independent samples, and must stay in the same split.

Thirty phrases are enough for the next decisive prototype experiment if they are
chosen to cover varied adjacent-gloss pairs, one-/two-handed transitions, face/lip
contrasts, repeated glosses, and genuine between-sign motion. They are not enough to
claim open-domain continuous ASL translation. With 13 signers and three performances
per phrase, the useful unit is 1,170 signer-performance sequences (plus synchronized
views), which is far stronger than simply adding more copies from the same three local
phrase identities.

## Reproducible assets

- Signer audit: `artifacts/reports/local_phrase_signer_audit_v17_v4_auto/`
- Grounded split: `data/local/stage2_v17_grounded_signer_split/`
- NCSLGR manifest: `active/v17/ncslgr_supervised_manifest_v17.json`
- Causal aligned trainer: `active/v17/train_unified_streaming_aligned_grounded_v17.py`
- Selected research checkpoint:
  `artifacts/models/unified_streaming_aligned_grounded_v17_v1/best_model.pth`
- Retained Stage-1 adaptation experiment:
  `artifacts/models/stage1_v17_grounded_adapt_v1/best_model.pth`

No official test split was accessed. Accepted live inference files were not overwritten.
