# Better public continuous-ASL data acquired — 2026-09-13

## Decision

SoMe ASL is excluded. The user's visual review found it predominantly one-handed and
unsuitable for this project. Its already-downloaded files are preserved but must not
enter a training manifest.

Two better, public sources are now present locally. Neither required an account, access
request, or message to a publisher.

## O5S5: primary new source

All six O5S5 narratives that currently have public ELAN transcripts were downloaded
with their matching raw videos:

- 6 signers and 6 frontal upper-body narratives;
- 2,586,024,974 video bytes (2.59 GB), 1,303.82 seconds (21.73 minutes), and 60,944
  fully decoded frames;
- 540p, 720p, 1080p, and 4K sources at approximately 30–60 fps;
- 3,959 timed manual annotations: 2,601 `RightHand_IDg` and 1,358 `LeftHand_IDg`;
- 591 distinct raw gloss strings;
- 53 frozen Citizen100 classes across 256 deduplicated occurrences under an exact
  ASL-LEX `SignBankAnnotationID` join.

Five videos decode through their reported frame count in OpenCV. The RD container
reports seven more tail frames than OpenCV can decode; its last annotation ends 0.91
seconds before the file duration, so those tail frames are explicitly excluded. Every
timed hand annotation fits its paired video. SHA-256 hashes are in
`corpus_quality_audit.json`. The visual contact sheet confirms clear frontal framing,
visible faces, and both hands across all six signers.

O5S5 is published by Gallaudet as open access under CC BY-NC-SA 4.0. O5S5 ID glosses
use ASL Signbank; the frozen Citizen classes already pin an ASL-LEX code whose official
table supplies the same `SignBankAnnotationID`. Exact equality therefore admits 256
positive occurrences across 53 classes without English-label guessing. Left/right
tier copies are paired one-to-one and deduplicated. Unannotated gaps remain ineligible
as background because transcript completeness is not established.

The six videos were replayed through the real live Apple Vision observer: 26,079
timestamped observations, with Apple Vision hands present in all 256 admitted target
occurrences. The signer-disjoint split reserves LG for validation and uses CK, Doug
Ridloff, JAH, LR, and RD for training. Combined with existing ASLLRP supervision, the
loader produces 6,819 training context windows across 73/100 classes; O5S5 contributes
1,005 of those windows across 49 classes and no background windows.

Sources: [Gallaudet O5S5 dataset and license](https://ida.gallaudet.edu/o5s5/index.html),
[public OSF transcripts](https://osf.io/769sw/).

## RWTH-BOSTON-104: supplemental sequence source

The complete official camera-0 distribution and support files were downloaded:

- 201 continuous sentence videos, 80,759,696 video bytes;
- 525.78 seconds, 15,746 fully decoded frames, 336×312 at 25 fps in the downloaded
  MPEG files;
- 3 signers, 888 gloss tokens, and 113 observed gloss strings;
- official 161-video train / 40-video test partition;
- 19 exact frozen Citizen100 raw-label strings across 148 token occurrences.

The official partition is not signer-disjoint: all three signers occur in both train
and test. The footage is old, grayscale, and low-resolution. Use it only as
training-only auxiliary temporal pretraining, never as evidence of signer
generalization or as a replacement evaluation set. Direct locked-head supervision
still requires exact visual-variant review.

Source: [official RWTH-BOSTON-104 page and download](https://www-i6.informatik.rwth-aachen.de/aslr/database-rwth-boston-104.php).

## Ready-to-use local material

- O5S5 raw videos and annotations:
  `data/local/open_asl_alternatives_20260913/o5s5/`
- RWTH raw videos, sentence corpora, lexicons, and language models:
  `data/local/open_asl_alternatives_20260913/rwth_boston_104/`
- Full measured audit and hashes: `corpus_quality_audit.json`
- O5S5 legacy English-string shortlist: `o5s5_locked100_candidate_events.csv` (290
  tier rows; superseded for admission)
- Authoritative exact-ID manifest, Apple Vision audit, and ready combined supervision:
  `../o5s5_citizen100_v17/`
- RWTH sentence manifest: `rwth_boston104_sentences.csv` (201 rows)
- O5S5 visual QA: `o5s5_contact_sheet.jpg`
- Reproducible checks: `scripts/audit_open_asl_alternatives_v17.py` and
  `scripts/acquire_rwth_boston104_v17.py`

Run the complete audit with:

```bash
venv/bin/python scripts/audit_open_asl_alternatives_v17.py
```

## What this solves

This acquisition now adds exact direct positives from real, natural, multi-sign,
visibly two-handed sequences and is ready for a bounded development-only experiment.
It does not cover all 100 classes in continuous context and does not prove
portrait-iPhone generalization. No protected test split was accessed and no model
training was started.
