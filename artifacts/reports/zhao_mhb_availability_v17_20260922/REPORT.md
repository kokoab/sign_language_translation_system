# Zhao/MHB availability review — 2026-09-22

## Outcome

Found author-owned component code and checkpoint files. This corrects the earlier
“no verified weights found” search result, but does not establish a complete MHB release.
No live promotion, training, dataset payload acquisition, or accuracy evaluation.

## Recovered components

Author identity: https://illuminatingfish.github.io/index.html

- https://github.com/IlluminatingFish/ASL-Handshape : one saved GCN ensemble checkpoint,
  1,099,293 bytes; weights-only inspection finds 65 tensors and an 88-output classifier.
  The entry point uses 11 hand nodes and hard-coded CUDA allocations. Label ordering,
  the extra category relative to the paper's 87 categories, and raw-video preprocessing
  remain unverified. Do not interpret integer outputs as named handshapes yet.
- https://github.com/IlluminatingFish/SegmentASLTransformer : five checkpoint files;
  downloaded only `transformer_model_f1_0.8689.pth` (5,400,744 bytes), selected by the
  upstream filename convention, not our validation outcomes. Its filename is not a
  measured result on our data. No README, license file, or GitHub releases found in
  the inspected repository trees. This does not establish rights for product distribution.

Commits, exact weight URLs, sizes and SHA256 are in provenance.json; recursive GitHub
file inventories are saved separately. Only selected source files and model weights
were downloaded. Repository data arrays, labels and prediction dumps were not acquired.

## Verified locally

`venv/bin/python artifacts/reports/zhao_mhb_availability_v17_20260922/check.py`

Strict segment checkpoint loading: 34 tensors. Synthetic input [1,32,189] gives
finite [1,32,3] output on MPS; maximum CPU/MPS difference 0.00000572205.
Hash checks and handshape output dimension checks pass. This is compatibility only.

## Why a real comparison would currently be invalid

The supplied test.py needs 27-node skeleton sequences and per-frame `preseg` predictions
loaded from `Bloss_predicted_indices.txt`. The source producing those predictions is
not present in the inspected tree. Its skeleton input array is also absent; the exact
raw-video node selection/normalization is not specified by this release.

The active model.py concatenation is actually additive skeleton plus preseg embeddings.
Handshape/label embedding use is commented out; test.py supplies random label indices
and probabilities, which this forward ignores. A separate model_handshape.py does use
labels, but has a different module layout and is not the architecture loaded above.
The active model has full-sequence attention, no causal mask, and next-frame differences
in test preprocessing. It is not a streaming drop-in. This Transformer should not be
called the paper's verified handshape-aware temporal graph model.

Replacing missing preseg with zeros, our wrist gate or ground-truth boundaries would
change the question or leak the answer. We therefore stopped before reporting accuracy.
A valid reproduction needs the upstream preseg producer/checkpoint, feature contract,
category maps and confirmation of which code/checkpoint corresponds to MHB.

## Paper and data register

Final paper: https://www.sign-lang.uni-hamburg.de/lrec/pub/26014.pdf

The paper trains boundaries from annotated ASLLRP-S starts/ends and combines handshape
pretraining with segmentation. Its handshape sources are ASLLVD, DSP, ASLLRP-S and the
NCSLGR 87-handshape demonstrations. Recognition uses isolated and sentence-derived signs.
It reports a random 4:1 split. The 80.23–83.30% recognition figures concern matched,
sufficiently represented segments; only 3,783/6,595 reference signs match its boundary
criterion. These are not whole-stream WER or evidence of pause-free phone performance.

Official access points for later review:

- ASLLVD: https://www.bu.edu/asllrp/av/dai-asllvd.html — citation videos with sign timing
  and start/end handshape labels; overlaps an existing project source.
- Sign Bank: https://dai.cs.rutgers.edu/dai/s/signbank — source identities, versions and
  permissions must be reconciled before admission; DSP inclusion is not blanket access.
- Handshape demonstrations: https://www.bu.edu/asllrp/cslgr/pages/ncslgr-handshapes.html
  — handshape supervision, not continuous transition ground truth.
- Continuous corpora: https://www.bu.edu/asllrp/rpt19/asllrp19.pdf — SignStream 3 includes
  handshape annotations; older NCSLGR SignStream 2 does not. Do not conflate the older
  continuous corpus with the separate handshape demonstration collection.

Read ../dataset_reconciliation_v17_20260922/REPORT.md before proposing acquisition:
we already have ASLLRP contextual signs and ASLLVD supplements. New caches or alternate
views would not constitute independent examples. No data acquisition is authorized here.

## Recommendation after the candidate review

None of the completed transfer probes qualifies for default Reel deployment. Their
protocols differ, so they do not rank encoder quality. The shared failure is insufficient
retention of correct signs while rejecting transitions, not proof that all local data is bad.

Next prepare an ASL temporal boundary experiment using existing verified complete timelines:
predict sign starts/ends continuously from hand/finger and body evidence, retain the
working identity classifier, and compare against unchanged Reel on complete videos.
First establish coverage and a fixed evaluation set with low-motion signs, adjacent signs
without pauses, and out-of-vocabulary/unknown spans handled explicitly. Keep unseen-label
regions unknown, not background. Use all eligible training boundaries irrespective of
locked100 identity, only where source/signer/annotation contracts permit. This is a proposed
new recipe, not permission to bypass existing training gates or claim it will fix Reel.

The first ablation should compare a temporal skeleton boundary model with and without
handshape features under one protocol. SHuBERT remains a possible offline representation
reference; its tiny interval readout did not establish that its encoder should be discarded.
Do not spend another run on a foreign blank-head veto or tune the three reused gaps.
