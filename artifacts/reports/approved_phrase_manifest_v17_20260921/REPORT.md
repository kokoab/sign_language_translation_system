# Enforced repaired phrase admission — 2026-09-21

The approved whole-phrase manifest is now
`active/v17/approved_phrase_manifest_20260921.json`. Original datasets remain unchanged.
Approved file view: `data/local/approved_phrases_v17_20260921/{phrases,other}`.
These are symlinks; preserve their original and recovered targets.

| Source | Training | Validation |
|---|---:|---:|
| User-reviewed local phrases | 232 | 199 |
| Established ASLLRP contiguous phrases | 44 | 12 |
| Fully resolved ASLLRP OTHER sequences | 6 | 0 |
| **Total** | **282** | **211** |

56 admitted clips use recovered evidence. Excluded whole sequences: 1 local clip with
insufficient hand detections in a rebuilt window, 1,098 ASLLRP clips with unresolved
annotations, and 266 cross-corpus identity-unresolved clips (125 NCSLGR / 141 Flores).
The manifest records each exclusion and preserves original/recovered locations. No
uncertain annotation becomes OTHER or blank. No transcript token was removed while
retaining its corresponding video. Trusted event cores inside excluded clips are not
included in this whole-sequence view; they remain available for a separately specified
supervision contract. Zero independent unseen-OOV evaluation examples are admitted.

## Enforcement and next-session command

From the repository root:

```sh
venv/bin/python -m active.v17.approved_phrase_data_v17
```

The verifier checks exact NPZ membership, every admitted file hash, and admission-evidence
and vocabulary hashes. Additional, removed or altered files fail verification. Both
`train_unified_streaming_ctc_v17.py` and `train_unified_streaming_aligned_grounded_v17.py`
default to the approved roots and canonical `--dataset-manifest`. They verify at the
start of `run`, before model loading. Old-root overrides, Flores supplements and full-local
supplements are rejected. Their checkpoint and result writers record manifest path/hash.
No new checkpoint was produced; writer integration was checked without training.

Flores and YouTube-motion experiment run/preflight launchers also check the gate before
model work, including before motion pretraining. Historical analysis and unrelated
trainers are not all rewritten: AGENTS.md and current ground truth explicitly require
this contract for any future phrase trainer and forbid bypass through legacy scripts.
Existing reports/results remain historical, pinned to their previous data versions.

## Data admission is complete; training readiness is not

**`training_ready` is false.** Verification succeeds, but an attempted training run
fails with the recorded reasons. The previous recipes score NCSLGR and ASLLRP OTHER
validation, both absent from this approved view. Blank/rest supervision, auxiliary input
pinning and independent unseen-OOV evaluation also remain unresolved. The manifest is
explicitly a phrase admission contract, not certification of isolated or RGB datasets.

Next session should develop a recipe compatible with the retained sources, define and
pin any auxiliary inputs and evaluation roles, and issue a reviewed new manifest version.
Do not merely set the flag to true or reinstate excluded data to satisfy the old metrics.
This gate implements the previously agreed training deferral; no training is launched.

## Validation

40 focused tests pass, including file/evidence tampering, unexpected membership,
old-root rejection and entry-point/launcher checks before model work. Actual phrase
loading succeeds for all 493 clips with stride4/window8. Build-time checks preserve
source-level signer disjointness and prevent cross-role exact video overlap. Compilation
and `git diff --check` pass. Measured counts and manifest digest are in `verification.json`.
Builder: `scripts/build_approved_phrase_manifest_v17.py`; it refuses an existing version.
