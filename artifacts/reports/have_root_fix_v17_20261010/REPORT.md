# HAVE omission root cause and fix — 2026-10-10

The five measured HAVE omissions in the matched take-11 replay are recovered. Each is followed by BIG in the correct order. Production Python and Swift agree on the exact three session traces with the same FP16 Core ML models. The signed iOS Release build is **installed on the iPhone 13**; automatic launch was blocked because the phone was locked. One pre-existing first-HAVE→ANGRY substitution in take 11 remains.

## Root cause

The decoder discarded short, accepted HAVE evidence while waiting for the sign to finish. Candidate endpoints move with the growing boundary window; a useful core could disappear from the candidate set. A larger window containing HAVE's exit and the next BIG then won as BIG. The existing six-frame commitment delay and dip-bridging rules allowed that merge. Simply reducing the delay reintroduced false fragments.

At approximately 7s, another problem compounded this: the TIRED rescue swapped scores, leaving HAVE at0.3267 and TIRED at0.0989 although motion had ruled TIRED out. Combining that evidence yields HAVE0.4256, above the unchanged0.4 acceptance threshold. The later full-contact window was ANGRY0.3786/HAVE0.1243; the earlier handoff incorrectly treated0.3786 as HAVE evidence.

A separate commitment bug could output later BIG before earlier pending HAVE. At20.8s, HAVE417–432 preceded BIG433–437, but BIG appeared at22.17s and HAVE at22.30s. The early loop scanned past unresolved earlier signs. This also explains real NAME MY inversions in the development sentences.

## Final implementation

- Retain only accepted HAVE cores while their ends are unresolved. Preserve the accepted evidence to the first two-frame transition (`O + UNK + B >=0.5`) within the existing30-frame span limit. Conflicting exit/crossing spans cannot erase that completed contact.
- An earlier open recognized word keeps ownership of its ending. The rescue requires new core observations and excludes continued two-hand lowering; a repeated held frame cannot create another sign.
- When the existing TIRED motion rule fires on such a new contact, combine HAVE/TIRED evidence instead of leaving confidence split between them.
- Early words cannot pass unresolved earlier signs. No sentence-template completion is used.

Original recognizer, boundary and cue model weights are retained; word threshold0.4, lag6 and dip-bridging settings remain. Python uses `stream.have_contact_closure` in `config_fixed.json`; Swift `v3Words` enables the corresponding behavior. The fixed Python config also pins the fragment+reject Core ML package instead of inheriting the old LocalA export path.

Changed production files:
- `active/v17/segmental_runtime_v17.py`
- `test/test_segmental_runtime_v17.py`
- `/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios/Runner/LiveReel/LiveReelDecoder.swift`

Task-only patches and original source snapshots are in this directory. Other recorded Swift source hashes are unchanged. The pre-existing worktree reorganization and `scripts/app_shell_v17.py` edit are preserved.

## Measured results

Actual timestamps and FP16 inputs are pinned; older frame-index/fps results are not comparable. Take11 has654 observations; exact10:06 live trace406;09:55 actual-PTS trace580.

| Check | Before | Fixed |
|---|---|---|
| Take11 accepted HAVEs |5|10|
| Five measured silent HAVE regions (~4.1,7.0,11.6,20.9,28.4s)|missing|HAVE followed by BIG in all5|
| Take11 ANGRY substitutions |1|1 (first phrase, unresolved)|
| Exact10:06 live input |HAVE×3, I, BIG, HAVE|HAVE×7, I, HAVE, BIG, HAVE|
|09:55 matched actual-PTS replay|HAVE×8|HAVE×9|
| Pair validation103 |correct82, exact58, extra26, silent18|identical words for every clip|
| Citizen validation378 |correct332, exact301, extra39, silent23|identical words for every clip|
| Development sentences89/226 words |S6/D17/I11,34 errors (15.04%)|S6/D15/I8,29 errors (12.83%)|

Pair validation contains25ANGRY,36TIRED,38HAPPY,4HAVE; zero falseHAVE both before and after. Citizen's one pre-existing falseHAVE is the same clip; none added. Only three development hypotheses changed: two NAME MY→MY NAME repairs and removal of an extra HELP. These are development results, **not held-out test accuracy**; no protected test was evaluated.

The seven isolated HAVE contacts in10:06 were checked against actual-PTS video frames. OpenCV `POS_MSEC` seeking was unreliable on that video; use `live_contacts_pts_review.jpg`, not the earlier non-PTS frame sheet. Replays use the saved observations and embeddings, avoiding any frame-selection ambiguity.

## Validation and reproduction

- 23 focused tests pass: `venv/bin/python -m unittest test.test_segmental_runtime_v17 -v`.
- New early-order regression fails on the saved original runtime (BIG jumps pending HAVE) and passes on the fix.
- Python/Swift contact and rescue math:5 fixtures, zero probability delta.
- Final production Swift replay words, boundaries and scores match the successful experiment on all3traces.
- Production Python matches the experimental Python runtime on the six previously failing negative clips and the old cached take11 inputs. That old cache is a separate sampling path, not the actual-PTS positive benchmark.
- Production Python Core ML replays of the actual traces match production Swift words and frame boundaries on all3sessions (maximum score delta1.28e-5); see `python_coreml_replays.json`.
- iOS unsigned Release build: `ios_build.log` ends BUILD SUCCEEDED. Build output is `/Users/frnzlo/slt_have_contact_build/Build/Products/Release-iphoneos/Runner.app`; no installation or device-run claim.
- Repository-wide `git diff --check` reports only pre-existing whitespace at `scripts/app_shell_v17.py:632`. Task-scoped check passes.

Use `verify_final.py` for saved-output invariants. `replay_python_coreml.py` reproduces the production exact-input replays. `production_harness/` contains the matching production Swift sources and standalone diagnostic entry points. `evaluate_contact2.py` records the uncached paired570-clip development check. Its saved key `closed_merge_odds` is a historical experiment name: **the final run uses legacy cue models and min fusion, with no odds fusion**. `final_results.json` gives concise final metrics and source hashes.

## Rejected experiments and remaining limits

Earlier global lag/threshold/cost/context/density changes were unsafe or ineffective. Unconditional contact closure added5falseHAVEs; ownership and contact-evidence guards remove these failures. Keeping all historical candidates added unrelated fragments, so only accepted HAVE evidence is retained.

The handshape canonical rotation has a sign error shared by training and inference. An isolated-only refit (68original training clips) corrected it, but did not solve the omissions. It is **not promoted**; preserve the legacy model/geometry pairing until a separate versioned migration. No new phrase training was done and the canonical training gates remain unchanged.

`raw_cache` contains post-correction scores, despite its name. The first cached geometry comparison was invalidated. Final comparisons start each variant with an empty post-correction cache and share only recognizer outputs keyed by complete input hashes.

This fixes the measured omission mechanism; it does not guarantee every future HAVE or resolve the first ANGRY substitution. Next safe action is to unlock the phone, open ASL Translator, and capture new live recordings with exact traces. Do not claim on-device performance from this Mac replay/build.

## Installation — 2026-10-10

User requested installation. Signed Release build passed and its code signature verified. Installed in place on the paired iPhone 13 without uninstalling. Device app inventory confirms the same installation URL returned by the installer for `com.kokoab.sltMobileApp` (1.0.0, build 1). Launch was denied solely because the device was locked; no new live recognition or performance claim. Validated source hashes were unchanged. See `installation.json`, `ios_signed_build.log`, `ios_install.json`, `ios_installed_app.json`, and `ios_launch.json`.
