# Local phrase signer audit

This audit groups the 780 videos in `data/raw_videos/PHRASES` by the face of the
camera-centered signer. It stores only anonymous assignments and audit scores; it does
not persist face embeddings.

The automatically selected solution is **three phrase-recording identities** with
298, 300, and 182 videos. All three occur in all nine phrase folders. The cosine
silhouette is 0.7371.

The wider local video collection does contain more people. That does not make them
additional signers in these nine phrase recordings. In particular, forcing seven
clusters produced sizes 298, 278, 180, 20, 2, 1, and 1. Visual review showed that the
20-video group is the same green-shirt signer under a different framing/expression,
the two-video group is that same signer, and the singleton groups are the dark-room
signer. Treating those fragments as separate people would create signer leakage.

For the grounded continuous experiment, anonymous signers 01 and 03 are training data
and signer 02 is validation data. The two low-margin clips in signer 03 remain excluded
from split evidence. Re-run the audit with:

```bash
venv/bin/python scripts/cluster_local_phrase_signers_v17.py \
  --video-root data/raw_videos/PHRASES \
  --output artifacts/reports/local_phrase_signer_audit_v17_<new_version>
```

`signer_clusters.json` includes the complete 2-to-10-cluster comparison. The 7-cluster
result was retained separately only as an audit failure case, not as supervision.
