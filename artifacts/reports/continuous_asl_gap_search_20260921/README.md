# Connected ASL data search — acquisition result

## Decision

**Acquire ASL-Homework-RGBD next.** It is the only identified corpus that directly
matches the failure we need to fix: broad signer coverage plus real connected RGB and
human sign boundaries. The official release contains 935 continuous recordings from
45 people (24 fluent, 21 learners), 1920×1080 RGB, stable participant codes, and ELAN
tiers for “Signing Happening,” exact gloss start/end times, nonmanuals, and production
errors.

Full access is through [Databrary volume 1249](https://nyu.databrary.org/volume/1249)
and is restricted to authorized Databrary researchers. I found no legitimate public
mirror. The publisher's [dataset page](https://latlab.ist.rit.edu/lrec2022/) exposes
only the F13 sample that is already local. The corpus index lists CC BY 4.0, but the
terms presented by Databrary at download time must govern the transfer.

Download only **RGB + EAF + demographics**. Depth, Kinect skeleton, face meshes, and
all learner recordings are unnecessary for the first pass. Audit the EAFs before
transferring video, then prioritize fluent signers whose verified glosses overlap the
locked 100.

## What I obtained now

The official NCSLGR full catalog and legacy SignStream database bundle are saved under
`data/local/continuous_asl_gap_search_20260921/ncslgr_catalog/`; ZIP integrity and
SHA-256 hashes passed. The catalog contains 1,887 utterances across 38 collections.
Against the earlier strict locked-string screen, 76 target-bearing parent utterances
exist outside the 166 clips already local. Current public endpoints expose 59 of those
parents, representing 50 unique front videos and 213.17 MiB.

I did not download those videos: their per-sign XML and complete participant metadata
still require the free DAI account, and video alone cannot safely train locked spans.
NCSLGR has only eight corpus participants, so it is a useful secondary source rather
than the main generalization fix. Use the
[DAI login/account page](https://dai.cs.rutgers.edu/dai/s/index?redirect=dai).

## Data already present

The strongest multi-signer material is already here, so it should not be downloaded
again:

- ASL STEM Wiki: **111 approved bounded spans, 31 glosses, 18 participants**; only 18
  glosses have at least three approved participants.
- ASLLRP: high-quality timed signing, but only four main modern-corpus signers.
- 2M-Flores: 155 selected genuine sentences, but the released metadata behaves as one
  local signer ID and cannot establish signer generalization.
- O5S5, RWTH-Boston-104, Cokely, local phrases, and public NCSLGR are already exhausted
  as small auxiliary sources.

## Rejected alternatives

- How2Sign offers RGB and English sentence alignment, but its public download page does
  not expose the human sign-level gloss annotations needed here. A 33+ GB transfer
  would not solve this supervision gap.
- FLEURS-ASL and the large ASL STEM release have model-generated pseudo annotations;
  they are not replacements for the requested confident human labels.
- OpenASL and YouTube-ASL add signer diversity but provide translation/caption targets,
  not trustworthy locked-gloss timing.

## Exact acquisition sequence

1. Obtain authorized access to Databrary volume 1249.
2. Download the EAF files and participant demographics first; do not fetch media yet.
3. Produce a locked-100 coverage table by fluent signer and reject mismatched variants.
4. Freeze signer-disjoint train/validation roles.
5. Download only matching RGB recordings, then run the unchanged Apple Vision v17
   extractor and preserve the publisher's boundaries.
6. Train one bounded Stage-2 comparison. No new decoder rule or boundary relabeling is
   justified before this data gate.

[Source matrix](sources.csv) · [Verification record](verification.json) ·
[prepared access requests](../continuous_asl_acquisition_20260912/ACCESS_REQUESTS.md)
