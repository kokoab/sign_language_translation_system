# Matched-variant WATER COLD diagnostic replay

ASLLRP signer JONATHAN, collection Jonathan_2012-11-27_sc95; existing development
validation example asllrp:30336.mp4:span00, not an unseen or protected test.
The acquisition folder says train_candidate; the later evaluation uses this clip
in its exact validation domain. No data split or checkpoint was changed.

Clip: data/local/asllrp_contiguous_phrases_v17/spans/train_candidate/asllrp/WATER_COLD/30336_span00.mp4
Duration 1.368033 seconds; manual bounds plus five context frames. This is a short
contiguous phrase fragment, not a fully translated sentence.
SHA256: 8101ded721fd7a88cbc55b512b98e33356efe67272c5ec2fb79245fd4d6b3ae3

Lexical mapping was checked against the locked manifest and official ASL-LEX table:
WATER -> A_02_031 -> SignBankAnnotationID WATER;
COLD -> C_02_068 -> SignBankAnnotationID COLD.
The ASLLRP annotation entry/variants match those IDs. This is documented lexical
matching, not a new expert visual certification. Four frames were decoded/inspected.
Source parent https://dai.cs.rutgers.edu/ss3front/30336.mp4 returned HTTP200 video/mp4.
Parent contains surrounding signing; use the local crop for the two-sign reference.

Executed command:
```sh
venv/bin/python scripts/live_reel_continuous_v17.py \
  --video data/local/asllrp_contiguous_phrases_v17/spans/train_candidate/asllrp/WATER_COLD/30336_span00.mp4 \
  --finish-at-eof --no-display --no-speech --naturalizer literal \
  --expected-sequence WATER COLD \
  --output-root artifacts/reports/stage2_v17_same_variant_replay_v1
```

Exit0. Raw-video Stage2 suggestion WATER COLD, exactly matching the reference
(0/2 sign edits). Committed hypothesis empty, no utterance committed; summary exact
false compares committed hypothesis, not Stage2. Zero stale camera frames dropped.
History: [history.json](20260910_071445_185192/history.json).
Both accepted and repaired frozen-cache predictions were also WATER COLD. This
single successful existing development example proves neither generalization nor
an improvement due to the repair; it is a useful positive diagnostic control.
No training, threshold tuning, protected evaluation or new video acquisition.
