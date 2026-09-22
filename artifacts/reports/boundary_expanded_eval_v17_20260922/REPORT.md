# Expanded held-out boundary evaluation

No promotion. Frozen Reel, identical commit rules in every arm.
12 reused ASLLRP development videos (24 signs) plus 60 held-out local signer02 clips (162 signs).
Familiar-signer reused development pool; not unseen-signer, not protected test, not live latency.

| Arm | Epoch | ASLLRP12 WER | local60 WER | Combined WER | Combined correct |
| --- | --- | --- | --- | --- | --- |
| frozen_pretrained_bio | — | 75.00% | 34.57% | 39.78% | 127/186 |
| finetune_epoch7 | 7 | 62.50% | 53.09% | 54.30% | 96/186 |
| augmented_epoch8 | 8 | 70.83% | 51.23% | 53.76% | 100/186 |

Interval recall and gap-contained commits are ASLLRP-only; local phrases carry no curated
boundary annotation. Retention is measured against the recorded Reel baseline on ASLLRP only.
Subsets are summarized separately; the combined column pools two different sources and is labelled as such.
See evaluation.json. Do not compare these figures against the older standalone 211-phrase convention.
