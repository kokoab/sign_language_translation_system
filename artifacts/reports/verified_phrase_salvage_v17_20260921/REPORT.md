# Conservative recovery from excluded phrases — 2026-09-21

Recovered **one additional continuous two-sign training subspan**, without changing
source annotations or weakening lexical mapping requirements. Canonical phrase total:
**494 = 283 training + 211 validation**. This is a small gain, not a remedy for the
previously identified coverage or unseen-OOV evaluation shortfalls. Training stays blocked.

## What passed

ASLLRP source `23882926.mp4`, annotations `343321` and `343322`: WHEN (locked code
C_03_086) followed by NOT (established OOV code E_01_100). Targets are WHEN, OTHER;
the 100-sign vocabulary is unchanged. Both annotations are complete and nonoverlapping,
and their frame intervals adjoin exactly. The original excluded crop remains excluded.

Re-extracted original frames [5,16) from the hash-verified existing source video using
the original orientation and frozen Apple landmark configuration. All 11 decoded frames
are retained. Observed-hand frame fraction before trimming is 1.0. The source is not
joined to another video, slowed down, or stretched to create extra temporal evidence.
The standard 32-step feature representation and stride4/window8 reconstruction produce
two CTC observations, sufficient for the two distinct targets. Archive metadata pins
source video/archive hashes, exact source interval, annotation IDs and ASL-LEX codes.

New file:
`data/local/approved_phrases_v17_20260921_v2/other/train/asllrp_verified_span/343321_343322.npz`.
This is a training subspan, not an independent source video, new signer, or test example.

## Where recovery stopped

A scan of excluded ASLLRP clips found 507 runs of at least two complete, nonoverlapping,
established-identity annotations containing a known sign. Only two had no gaps between
annotations. For this recovery pass, unlabeled gaps were conservatively excluded rather
than assumed free of additional signs. That restriction does **not** establish that all
other gaps contain signs or that those 505 runs are permanently unusable; their complete
interval supervision was not certified in this pass. The earlier 493 approved clips keep
their existing admission basis; this is a stricter salvage rule, not a retrospective
assertion that every established phrase needs gap-free annotations.

The other gap-free candidate, STILL + HAVE (annotations386202/386203), lasts seven frames.
It gives only one stride4/window8 observation for two targets. It remains excluded; no
retiming or architecture change was made to force admission. Flores/NCSLGR ambiguous
cross-corpus identity links and the local quality-flagged clip were not guessed or
re-admitted. No new acquisitions or fluent-review substitute was introduced.

## Enforced handoff

New canonical manifest: `active/v17/approved_phrase_manifest_20260921_v2.json`.
Its view symlinks the previous493 approved archives and adds the one raw-re-extracted
subspan. Version1 and every original dataset are preserved. Exclusion records continue
to describe the full original clips, including the parent of the new admitted subspan.
The manifest pins version1, the annotation evidence and the salvage script itself.

The shared default manifest/root and AGENTS.md now use v2; existing CTC trainers and
launchers inherit it. Membership/hash checks pass. `training_ready=false` and all recipe,
auxiliary supervision and evaluation blockers remain. Verify with:

```sh
venv/bin/python -m active.v17.approved_phrase_data_v17
```

Actual shared loader counts: known-phrase train276/validation211; OTHER-containing
train7/validation0. Signers and exact video hashes remain cross-role disjoint. The new
source has its own `asllrp_verified_span` name; a future training recipe must explicitly
include that source in metrics/sampling rather than assume the old source taxonomy.
41 focused tests pass, including gap/ambiguity/overlap/cut-boundary rejection. Compile
and diff checks pass. No training, benchmark rerun, download or protected-test access.

Stop here for this recovery rule. More substantial expansion needs defensible source
annotation coverage or separately admitted isolated event supervision; neither is
silently counted as repaired phrases.
