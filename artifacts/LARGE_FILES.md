# Large artifact files — read this, not them

Every file under `artifacts/` at or above 1 MB — 362 files, largest first. Smaller-but-still-costly files (200 KB–1 MB) are not listed; find them with:

```sh
find artifacts -size +200k \( -name '*.json' -o -name '*.csv' \) | xargs ls -S
```

**Never read one of these whole.** The largest is ~1M tokens and will end your
session. Find the field you need below, then extract just that field:

```sh
jq '.some_key' artifacts/reports/<dir>/result.json     # one field
jq -r 'keys[]' artifacts/reports/<dir>/result.json      # what is in it
head -5 artifacts/reports/<dir>/audit.csv               # CSV shape
```

Regenerate: `venv/bin/python scripts/index_large_artifacts_v17.py`

---

| File | MB | Contents |
|---|---:|---|
| `artifacts/reports/clean_boundary_subset_20260920/curated_manifest.json` | 33.0 | nested: `events`, `eligible_cores`, `boundary_windows`, `audit` |
| `artifacts/reports/fist_letters_v17_20260929/swift_parity_cases.json` | 32.5 | array, 598 items of objects keyed `landmarks`, `logits`, `refined`, `features` |
| `artifacts/generated/paper_revision_20261007/trusted-read-02/document-result.json` | 24.5 | `isError`=false — nested: `content`, `structuredContent` |
| `artifacts/reports/confident_supervision_v17_20260920/confident_supervision.json` | 17.2 | `format`=slt_confident_continuous_supervision_v17; `version`=1 — nested: `contract`, `provenance`, `rows`, `decisions`, `class_coverage`, `audit` |
| `artifacts/reports/good_thankyou_variant_audit_v17_20261008/two_handed_good_inputs/local_train_two_handed_good.json` | 14.3 | `format`=slt_v17_local_deep_clean_final_v1; `split`=train; `split_eligibility`=train_only_after_human_review; `selected_clips`=13296; `selected_classes`=94; `signer_disjoint`=false; `signer_overlap_user_approved`=true; `citizen_test_accessed`=false — nested: `… |
| `artifacts/reports/good_thankyou_variant_audit_v17_20261008/local_train_no_good_thankyou.json` | 14.1 | `format`=slt_v17_local_deep_clean_final_v1; `split`=train; `split_eligibility`=train_only_after_human_review; `selected_clips`=13155; `selected_classes`=92; `signer_disjoint`=false; `signer_overlap_user_approved`=true; `citizen_test_accessed`=false — nested: `… |
| `artifacts/reports/citizen_variant_local_filter_v17_20261008/local_train_citizen_variant.json` | 12.8 | `format`=slt_v17_local_deep_clean_final_v1; `split`=train; `split_eligibility`=train_only_after_human_review; `selected_clips`=11956; `selected_classes`=85; `signer_disjoint`=false; `signer_overlap_user_approved`=true; `citizen_test_accessed`=false — nested: `… |
| `artifacts/reports/live_correctness_v17_20260929/motion_probe.json` | 12.3 | nested: `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/paired_phase.json` | 11.7 | nested: `compute_ms`, `records` |
| `artifacts/reports/fingerspell_trigger_ios_v17_20260930/trigger_fixture.json` | 10.1 | array, 63 items of objects keyed `name`, `raw`, `times`, `rows`, `probs`, `fires` |
| `artifacts/reports/live_reel_mobile_v17_20260928/fixtures_long/fixture_3.json` | 9.4 | `id`=session:20260928_213626_432391; `video`=/Volumes/secret/SLT/SLT/artifacts/app_sessions/20260928_2136… — nested: `reference`, `frames`, `bio_words`, `bio_letters`, `spans`, `crops`, `words` |
| `artifacts/reports/live_correctness_v17_20260929/parity_fixtures/fixture_3.json` | 9.4 | `id`=session:20260928_213626_432391; `video`=/Volumes/secret/SLT/SLT/artifacts/app_sessions/20260928_2136… — nested: `reference`, `frames`, `bio_words`, `bio_letters`, `spans`, `crops`, `words` |
| `artifacts/reports/my_transition_recovery_v17_20261011/balanced_validation.json` | 8.3 | nested: `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/paired_validation.json` | 8.2 | nested: `records` |
| `artifacts/reports/annotation_identity_audit_v17_20260921/event_ledger.json` | 8.1 | array, 9104 items of objects keyed `identity`, `archive`, `role`, `signer`, `video_path`, `source_video_sha256`, `parent_utterance`, `annotation_id` |
| `artifacts/reports/my_transition_recovery_v17_20261011/merge_phase.json` | 7.8 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/context_final_phase.json` | 7.8 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/preserved_phase.json` | 7.7 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/jitter_all.json` | 7.7 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/joint_all.json` | 7.6 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/narrow_final_all.json` | 7.6 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/cadence_final_all.json` | 7.6 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/merge_all.json` | 7.6 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/context_final_all.json` | 7.6 | nested: `compute_ms`, `records` |
| `artifacts/models/stage2_v17_transition_adapt_v2/no_stem/seed_1701/history.json` | 7.5 | array, 21 items of objects keyed `epoch`, `initialized_candidate` |
| `artifacts/models/stage2_v17_transition_adapt_v2/with_stem/seed_1701/history.json` | 7.5 | array, 21 items of objects keyed `epoch`, `initialized_candidate` |
| `artifacts/models/stage2_v17_transition_adapt_v1/no_stem/seed_1701/history.json` | 7.5 | array, 21 items of objects keyed `epoch`, `initialized_candidate` |
| `artifacts/models/stage2_v17_transition_adapt_v2/with_stem/seed_1702/history.json` | 7.5 | array, 21 items of objects keyed `epoch`, `initialized_candidate` |
| `artifacts/reports/my_transition_recovery_v17_20261011/confidence_consensus_all.json` | 7.5 | nested: `compute_ms`, `records` |
| `artifacts/models/stage2_v17_transition_adapt_v2/no_stem/seed_1702/history.json` | 7.5 | array, 21 items of objects keyed `epoch`, `initialized_candidate` |
| `artifacts/models/stage2_v17_transition_adapt_v1/with_stem/seed_1701/history.json` | 7.5 | array, 21 items of objects keyed `epoch`, `initialized_candidate` |
| `artifacts/models/stage2_v17_transition_adapt_v1/no_stem/seed_1702/history.json` | 7.5 | array, 21 items of objects keyed `epoch`, `initialized_candidate` |
| `artifacts/reports/my_transition_recovery_v17_20261011/confidence_all.json` | 7.4 | nested: `compute_ms`, `records` |
| `artifacts/reports/reel_decision_probe_v17_20260922/evidence.json` | 7.4 | `test_accessed`=false — nested: `summary`, `results`, `skipped`, `models`, `settings`, `limitations` |
| `artifacts/reports/live_correctness_v17_20260929/probe.json` | 7.2 | nested: `records` |
| `artifacts/reports/live_correctness_v17_20260929/geometry_replay.json` | 7.2 | nested: `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/boost_phase.json` | 7.1 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/narrow_final_validation.json` | 6.5 | nested: `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/preserved_validation.json` | 6.4 | nested: `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/context_final_validation.json` | 6.4 | nested: `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/joint_validation.json` | 6.4 | nested: `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/cadence_final_validation.json` | 6.4 | nested: `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/merge_validation.json` | 6.4 | nested: `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/confidence_validation.json` | 6.4 | nested: `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/cadence_screen.json` | 6.1 | nested: `compute_ms`, `records` |
| `artifacts/models/stage2_v17_transition_adapt_v1/with_stem/seed_1702/history.json` | 6.1 | array, 17 items of objects keyed `epoch`, `initialized_candidate` |
| `artifacts/reports/my_transition_recovery_v17_20261011/positive_final_phase.json` | 5.8 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/boost_validation.json` | 5.7 | nested: `records` |
| `artifacts/vendor/slrt/CiCo/data_preparation/how2sign_realigned_train.csv` | 5.6 | 31165 rows — columns: `VIDEO_ID	VIDEO_NAME	SENTENCE_ID	SENTENCE_NAME	START_REALIGNED	END_REALIGNED	SENTENCE` |
| `artifacts/reports/live_reel_mobile_v17_20260928/fixtures_long/fixture_2.json` | 5.6 | `id`=local:HELLO_HOW_YOU:833e2480; `video`=/Volumes/secret/SLT/SLT/data/raw_videos/PHRASES FIXED/HELLO_… — nested: `reference`, `frames`, `bio_words`, `bio_letters`, `spans`, `crops`, `words` |
| `artifacts/reports/live_correctness_v17_20260929/parity_fixtures/fixture_2.json` | 5.6 | `id`=local:HELLO_HOW_YOU:833e2480; `video`=/Volumes/secret/SLT/SLT/data/raw_videos/PHRASES FIXED/HELLO_… — nested: `reference`, `frames`, `bio_words`, `bio_letters`, `spans`, `crops`, `words` |
| `artifacts/reports/live_reel_mobile_v17_20260928/fixtures_long/fixture_1.json` | 5.6 | `id`=local:HELLO_HOW_YOU:6446e671; `video`=/Volumes/secret/SLT/SLT/data/raw_videos/PHRASES FIXED/HELLO_… — nested: `reference`, `frames`, `bio_words`, `bio_letters`, `spans`, `crops`, `words` |
| `artifacts/reports/live_correctness_v17_20260929/parity_fixtures/fixture_1.json` | 5.6 | `id`=local:HELLO_HOW_YOU:6446e671; `video`=/Volumes/secret/SLT/SLT/data/raw_videos/PHRASES FIXED/HELLO_… — nested: `reference`, `frames`, `bio_words`, `bio_letters`, `spans`, `crops`, `words` |
| `artifacts/reports/boundary_expanded_eval_v17_20260922/tcn_comparison/evaluation.json` | 5.5 | `status`=complete; `completed`=72; `elapsed_seconds`=420.37081145900004; `promoted`=false; `limitation`=Conditional full-video comparison uses native BoundaryRecogn… — nested: `runs`, `reel` |
| `artifacts/models/stage2_v17_live_adapt_v1/seed_17101/history.json` | 5.4 | nested: `design`, `baseline`, `original_baseline`, `epochs`, `best_key` |
| `artifacts/models/stage2_v17_revisable_core_v1/seed_17101/history.json` | 5.4 | nested: `design`, `baseline`, `original_baseline`, `epochs`, `best_key` |
| `artifacts/reports/live_reel_mobile_v17_20260928/fixtures_long/fixture_0.json` | 5.4 | `id`=local:PLEASE_HELP_ME:PLEASE_HELP_ME_16; `video`=/Volumes/secret/SLT/SLT/data/raw_videos/PHRASES FIXED/PLEASE… — nested: `reference`, `frames`, `bio_words`, `bio_letters`, `spans`, `crops`, `words` |
| `artifacts/reports/live_correctness_v17_20260929/parity_fixtures/fixture_0.json` | 5.4 | `id`=local:PLEASE_HELP_ME:PLEASE_HELP_ME_16; `video`=/Volumes/secret/SLT/SLT/data/raw_videos/PHRASES FIXED/PLEASE… — nested: `reference`, `frames`, `bio_words`, `bio_letters`, `spans`, `crops`, `words` |
| `artifacts/models/stage2_v17_revisable_v1/seed_17101/history.json` | 5.4 | nested: `design`, `baseline`, `original_baseline`, `epochs`, `best_key` |
| `artifacts/reports/live_reel_cached_stage1_v17/20260904_095151_081931/history.json` | 5.1 | `format`=slt_live_reel_stage1_v17_session; `version`=1; `prototype_scope`=growing activity-anchored Stage-1 candidates with label-stab…; `started_utc`=2026-09-04T01:51:51.082115+00:00; `source`=camera:0; `mode`=cascade; `score_semantics`=uncalibrated Stage-1 s… |
| `artifacts/reports/my_transition_recovery_v17_20261011/boost_new.json` | 4.9 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/preserved_offsets.json` | 4.9 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/positive_genuine.json` | 4.7 | nested: `compute_ms`, `records` |
| `artifacts/reports/live_reel_mobile_v17_20260928/fixtures/fixture_2.json` | 4.7 | `id`=local:HELLO_HOW_YOU:f6a616f4; `video`=/Volumes/secret/SLT/SLT/data/raw_videos/PHRASES FIXED/HELLO_… — nested: `reference`, `frames`, `bio_words`, `bio_letters`, `spans`, `crops`, `words` |
| `artifacts/reports/live_reel_mobile_v17_20260928/fixtures/fixture_1.json` | 4.7 | `id`=local:HELLO_HOW_YOU:f0b6ec18; `video`=/Volumes/secret/SLT/SLT/data/raw_videos/PHRASES FIXED/HELLO_… — nested: `reference`, `frames`, `bio_words`, `bio_letters`, `spans`, `crops`, `words` |
| `artifacts/reports/live_reel_mobile_v17_20260928/fixtures/fixture_3.json` | 4.6 | `id`=local:HELLO_HOW_YOU:069536d5; `video`=/Volumes/secret/SLT/SLT/data/raw_videos/PHRASES FIXED/HELLO_… — nested: `reference`, `frames`, `bio_words`, `bio_letters`, `spans`, `crops`, `words` |
| `artifacts/reports/live_reel_mobile_v17_20260928/fixtures/fixture_0.json` | 4.6 | `id`=local:HELLO_HOW_YOU:64f34df5; `video`=/Volumes/secret/SLT/SLT/data/raw_videos/PHRASES FIXED/HELLO_… — nested: `reference`, `frames`, `bio_words`, `bio_letters`, `spans`, `crops`, `words` |
| `artifacts/reports/my_like_regression_v17_20261011/rollback_review/contacts.inputs.json` | 4.6 | array, 12 items of objects keyed `boxes`, `embeddings`, `end`, `id`, `landmarks`, `phone_gloss`, `start`, `take` |
| `artifacts/reports/my_transition_recovery_v17_20261011/boost_genuine.json` | 4.5 | nested: `compute_ms`, `records` |
| `artifacts/reports/stage2_v17_live_matched_v1/cache.json` | 4.4 | `format`=slt_stage2_live_matched_cache_v17; `protected_test_accessed`=false — nested: `contract`, `rows` |
| `artifacts/reports/stage2_v17_live_matched_v1/cache_progress.json` | 4.4 | `completed`=1647; `total`=1647; `elapsed_seconds`=3386.57542425 — nested: `results` |
| `artifacts/reports/have_root_fix_v17_20261010/causal.json` | 4.3 | `frames`=654 — nested: `results` |
| `artifacts/reports/my_transition_recovery_v17_20261011/completion_screen.json` | 4.2 | nested: `compute_ms`, `records` |
| `artifacts/reports/sept30_fix_port_v17_20261011/phone/live_reel_traces/20261011_220739.json` | 4.2 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/my_transition_recovery_v17_20261011/confirm_screen.json` | 4.2 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/decision_all.json` | 4.1 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/bounded_screen.json` | 4.1 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/positive_bounded_new.json` | 4.0 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/rank_screen.json` | 4.0 | nested: `compute_ms`, `records` |
| `artifacts/reports/boundary_expanded_eval_v17_20260922/evaluation.json` | 4.0 | `promoted`=false; `limitation`=12 reused ASLLRP development videos plus the60 held-out loca… — nested: `runs`, `reel` |
| `artifacts/reports/my_transition_recovery_v17_20261011/narrow_boundary_screen.json` | 3.8 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/stress_joint.json` | 3.8 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/stress_merge.json` | 3.8 | nested: `compute_ms`, `records` |
| `artifacts/reports/boundary_adaptation_diagnostic_v17_20260922/evaluation.json` | 3.8 | `status`=complete; `elapsed_seconds`=489.261869709; `frozen_reference_reproduced`=true; `promoted`=false; `limitation`=Cached MediaPipe pose, batched bounded-window offline compos… — nested: `runs`, `reel`, `timings` |
| `artifacts/reports/my_transition_recovery_v17_20261011/boundary_screen.json` | 3.7 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_like_regression_v17_20261011/long_word_run_trace.json` | 3.7 | nested: `frames` |
| `artifacts/reports/my_have_phone_v17_20261011/device_inputs/long_word_run.json` | 3.7 | nested: `frames` |
| `artifacts/reports/have_omission_v17_20261010/round2.json` | 3.7 | `frames`=654 — nested: `results` |
| `artifacts/reports/my_transition_recovery_v17_20261011/context_screen.json` | 3.6 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/context_score_screen.json` | 3.6 | nested: `compute_ms`, `records` |
| `artifacts/reports/have_omission_v17_20261010/take11_pts_trace.json` | 3.6 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/have_root_fix_v17_20261010/take11_trace.json` | 3.6 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/my_have_phone_v17_20261011/device_inputs/take11.json` | 3.6 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/my_transition_recovery_v17_20261011/positive_bounded_genuine.json` | 3.6 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/jitter_floor_screen.json` | 3.5 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/splice_boost.json` | 3.5 | nested: `records` |
| `artifacts/reports/o5s5_augmented_v17_20260914/combined_supervision.json` | 3.5 | `format`=slt_stage1_window_supervision_v17; `version`=1; `raw_format`=apple_vision_isotropic_xy_confidence_v1; `all_signs_source_sha256`=8d0b9f65fe5b3441dcf65428ff32b029450ba4e9076a85b1b7ef17cf9b86…; `frozen_inputs_sha256`=a300e66c5c677cef7d4bc63fcd55b575254e4… |
| `artifacts/reports/o5s5_citizen100_v17/combined_supervision.json` | 3.5 | `format`=slt_stage1_window_supervision_v17; `version`=1; `raw_format`=apple_vision_isotropic_xy_confidence_v1; `all_signs_source_sha256`=8d0b9f65fe5b3441dcf65428ff32b029450ba4e9076a85b1b7ef17cf9b86…; `frozen_inputs_sha256`=a300e66c5c677cef7d4bc63fcd55b575254e4… |
| `artifacts/reports/local_deep_clean_v17/train_v17_audit.csv` | 3.5 | 13381 rows — columns: `feature_path`, `video_path`, `split`, `label`, `orientation`, `width`, `height`, `source_frames_before_trim` +17 more |
| `artifacts/reports/similar_sign_local_audit_v17_20261008/local_all_preds_9683_iso.json` | 3.5 | array, 16277 items of objects keyed `split`, `item`, `label`, `pred`, `p`, `p_label`, `raw` |
| `artifacts/reports/temporal_repair_training_v17_20261011/temporal_train_rows.json` | 3.4 | array, 9528 items of objects keyed `id`, `parent`, `group`, `source`, `signer_id`, `label`, `phase`, `capture_offset` |
| `artifacts/reports/continuous_rebuild_v17_v1/joint_context_v2.json` | 3.4 | `test_accessed`=false; `promoted`=false; `claim`=development evaluation of exact CTC mass plus motion-conditi… — nested: `by_weight`, `details` |
| `artifacts/reports/stage2_v17_revisable_core_v1/supervision.json` | 3.3 | `format`=slt_stage2_live_transition_supervision_v17; `version`=1; `protected_test_accessed`=false — nested: `contract`, `inputs`, `items`, `audit`, `limitations` |
| `artifacts/reports/have_root_fix_v17_20261010/scoring.json` | 3.3 | `frames`=654 — nested: `results` |
| `artifacts/reports/stage2_v17_revisable_v1/supervision.json` | 3.2 | `format`=slt_stage2_live_transition_supervision_v17; `version`=1; `protected_test_accessed`=false — nested: `contract`, `inputs`, `items`, `audit`, `limitations` |
| `artifacts/reports/boundary_local_calibration_v17_20260922/calibration.json` | 3.2 | `selected`=frozen_min3 — nested: `summaries`, `records` |
| `artifacts/reports/have_omission_v17_20261010/round3.json` | 3.1 | `frames`=654 — nested: `results` |
| `artifacts/reports/sept30_fix_port_v17_20261011/first_trace.json` | 3.1 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/geometry_motion_sweep.json` | 3.1 | nested: `rows`, `sweep` |
| `artifacts/reports/stage1_window_v17/supervision_manifest.json` | 3.0 | `format`=slt_stage1_window_supervision_v17; `version`=1; `raw_format`=apple_vision_isotropic_xy_confidence_v1; `all_signs_source_sha256`=8d0b9f65fe5b3441dcf65428ff32b029450ba4e9076a85b1b7ef17cf9b86…; `frozen_inputs_sha256`=a300e66c5c677cef7d4bc63fcd55b575254e4… |
| `artifacts/reports/have_root_fix_v17_20261010/0955_trace.json` | 3.0 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/my_have_phone_v17_20261011/device_inputs/take0955.json` | 3.0 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/my_transition_recovery_v17_20261011/rank_swap_new.json` | 2.9 | nested: `compute_ms`, `records` |
| `artifacts/reports/local_familiar_signer_v17_20260922/split_manifest.json` | 2.9 | `format`=local_familiar_signer_split_v17; `training_ready`=false; `source_manifest`=data/local/combined_dataset_v17_20260922/manifest.json; `source_manifest_sha256`=089da8b13943b65663691cc7e23c8bd1dcb279fceca56ab1675fa768d19c…; `seed`=17521; `policy`=Keep all… |
| `artifacts/reports/boundary_data_audit_20260920/window_details.json` | 2.9 | array, 6641 items of objects keyed `role`, `source`, `item`, `video`, `label`, `start`, `end`, `window_start` |
| `artifacts/reports/my_transition_recovery_v17_20261011/calibrated_new.json` | 2.8 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/preserved_new.json` | 2.8 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/optimized_equivalence.json` | 2.8 | nested: `compute_ms`, `records` |
| `artifacts/reports/citizen_variant_local_filter_v17_20261008/local_val_citizen_variant.json` | 2.8 | `format`=slt_v17_local_deep_clean_final_v1; `split`=val; `split_eligibility`=validation_nonsigner_disjoint_user_approved; `selected_clips`=2603; `selected_classes`=85; `signer_disjoint`=false; `signer_overlap_user_approved`=true; `citizen_test_accessed`=false… |
| `artifacts/reports/have_root_fix_v17_20261010/bias.json` | 2.8 | `frames`=654 — nested: `results` |
| `artifacts/reports/my_transition_recovery_v17_20261011/preserved_splice.json` | 2.7 | nested: `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/context_final_splice.json` | 2.7 | nested: `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/joint_splice.json` | 2.7 | nested: `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/merge_splice.json` | 2.7 | nested: `records` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_replay_100646.json` | 2.7 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/my_transition_recovery_v17_20261011/ambiguous_new.json` | 2.7 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/current_offsets.json` | 2.7 | nested: `compute_ms`, `records` |
| `artifacts/reports/have_root_fix_v17_20261010/merge.json` | 2.7 | `frames`=654 — nested: `results` |
| `artifacts/reports/temporal_repair_training_v17_20261011/training_results.json` | 2.7 | `seconds`=25.237917184829712; `encoders_unchanged`=true — nested: `runs`, `baseline`, `selected`, `diagnostic`, `checkpoints` |
| `artifacts/reports/my_transition_recovery_v17_20261011/splice_current.json` | 2.6 | nested: `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/mix_screen.json` | 2.6 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/rescore_old_genuine.json` | 2.6 | nested: `compute_ms`, `records` |
| `artifacts/reports/live_reel_stage1_v17/20260902_213908_704489/history.json` | 2.6 | `format`=slt_live_reel_stage1_v17_session; `version`=1; `prototype_scope`=growing activity-anchored Stage-1 candidates with label-stab…; `started_utc`=2026-09-02T13:39:08.704777+00:00; `source`=camera:0; `mode`=cascade; `score_semantics`=uncalibrated Stage-1 s… |
| `artifacts/reports/stage2_data_learnability_audit_v17/all_window_predictions.csv` | 2.5 | 8978 rows — columns: `role`, `source`, `signer`, `item_id`, `identity`, `category`, `target`, `schedule` +14 more |
| `artifacts/reports/my_transition_recovery_v17_20261011/boost_screen.json` | 2.5 | nested: `compute_ms`, `records` |
| `artifacts/reports/stage1_window_v17/runtime_epoch01/hungry_recovered/20260912_085551_100597/history.json` | 2.4 | `format`=slt_live_reel_stage1_v17_session; `version`=1; `prototype_scope`=revisable timestamped Stage-1 windows with NO_EMIT; `started_utc`=2026-09-12T00:55:51.100782+00:00; `source`=artifacts/generated/revisable_hungry_diagnosis/recovered_pre…; `mode`=cascade… |
| `artifacts/reports/my_transition_recovery_v17_20261011/rescore_screen.json` | 2.4 | nested: `compute_ms`, `records` |
| `artifacts/models/stage3_v17_asl_order_v1/tokenizer.json` | 2.4 | `version`=1.0 — nested: `truncation`, `padding`, `added_tokens`, `normalizer`, `pre_tokenizer`, `post_processor`, `decoder`, `model` |
| `artifacts/models/stage3_v17_t5_efficient_tiny_locked100_v1/tokenizer.json` | 2.4 | `version`=1.0 — nested: `truncation`, `padding`, `added_tokens`, `normalizer`, `pre_tokenizer`, `post_processor`, `decoder`, `model` |
| `artifacts/models/stage3_v17_asl_order_fs_v1/tokenizer.json` | 2.4 | `version`=1.0 — nested: `truncation`, `padding`, `added_tokens`, `normalizer`, `pre_tokenizer`, `post_processor`, `decoder`, `model` |
| `artifacts/models/stage3_composition_v17_20260929/tokenizer.json` | 2.4 | `version`=1.0; `truncation`=null — nested: `padding`, `added_tokens`, `normalizer`, `pre_tokenizer`, `post_processor`, `decoder`, `model` |
| `artifacts/models/stage3_composition_v17_20260929_v2/tokenizer.json` | 2.4 | `version`=1.0; `truncation`=null — nested: `padding`, `added_tokens`, `normalizer`, `pre_tokenizer`, `post_processor`, `decoder`, `model` |
| `artifacts/model_assets/models/output/slt_conversational_t5_model/tokenizer.json` | 2.4 | `version`=1.0; `padding`=null — nested: `truncation`, `added_tokens`, `normalizer`, `pre_tokenizer`, `post_processor`, `decoder`, `model` |
| `artifacts/models/stage3_multisentence_tiny_v17_20260929/tokenizer.json` | 2.4 | `version`=1.0; `truncation`=null; `padding`=null — nested: `added_tokens`, `normalizer`, `pre_tokenizer`, `post_processor`, `decoder`, `model` |
| `artifacts/models/stage3_multisentence_tiny_run1_v17_20260929/tokenizer.json` | 2.4 | `version`=1.0; `truncation`=null; `padding`=null — nested: `added_tokens`, `normalizer`, `pre_tokenizer`, `post_processor`, `decoder`, `model` |
| `artifacts/reports/my_transition_recovery_v17_20261011/current_context.json` | 2.4 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/cadence_negative.json` | 2.4 | nested: `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/mix_genuine.json` | 2.4 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/localA_context.json` | 2.3 | nested: `compute_ms`, `records` |
| `artifacts/model_assets/models/manifest.json` | 2.3 | `A_65bb1e90_96c570.npy`=A; `A_040ae1f0_ccd1f6.npy`=A; `A_3c170d03__from_MARIAH_A__0f5b2d6f5f_df63e2.npy`=A; `A_9677a23b_d331bc.npy`=A; `A_03aafef9_30c0b8.npy`=A; `A_3f4dd5bf_9200f1.npy`=A; `A_58e49874_4bc7b7.npy`=A; `A_8bfb1f6f_30747f.npy`=A |
| `artifacts/reports/manifest.json` | 2.3 | `A_65bb1e90_96c570.npy`=A; `A_040ae1f0_ccd1f6.npy`=A; `A_3c170d03__from_MARIAH_A__0f5b2d6f5f_df63e2.npy`=A; `A_9677a23b_d331bc.npy`=A; `A_03aafef9_30c0b8.npy`=A; `A_3f4dd5bf_9200f1.npy`=A; `A_58e49874_4bc7b7.npy`=A; `A_8bfb1f6f_30747f.npy`=A |
| `artifacts/reports/my_transition_recovery_v17_20261011/merge_screen.json` | 2.2 | nested: `compute_ms`, `records` |
| `artifacts/reports/supplement_finalization_v17_20260922/semlex.json` | 2.2 | `format`=finalized_supervised_supplement_v17; `source`=semlex; `signer_policy`=preserve existing roles; overlap does not disqualify — nested: `counts`, `records`, `evidence_sha256`, `exclusions` |
| `artifacts/generated/xcode_dd_gosplit3/Build/Intermediates.noindex/XCBuildData/60b6dad69784e2e2ed623b8644b48cbb.xcbuilddata/manifest.json` | 2.2 | nested: `client`, `targets`, `nodes`, `commands` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_exact_100646_k4.json` | 2.2 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_exact_100646_k3.json` | 2.2 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_exact_100646_k2.json` | 2.2 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_exact_100646_k0.json` | 2.2 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_exact_100646_k1.json` | 2.2 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_exact_100646_m1.json` | 2.2 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_exact_100646_m2.json` | 2.2 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_pts_100646_m0.json` | 2.2 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_exact_100646_m3.json` | 2.2 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_exact_100646_m4.json` | 2.2 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_exact_100646_m5.json` | 2.2 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_pts_100646_m6.json` | 2.2 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_pts_100646_m2.json` | 2.2 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_exact_100646_m6.json` | 2.2 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/my_have_phone_v17_20261011/device_inputs/live1006.json` | 2.2 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/generated/xcode_dd_gosplit3/Build/Intermediates.noindex/XCBuildData/366701a6f77369c0d8d51da3f4461e10.xcbuilddata/manifest.json` | 2.1 | nested: `client`, `targets`, `nodes`, `commands` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_pts_100646_m4.json` | 2.1 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_pts_100646_m8.json` | 2.1 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/reports/go_split_root_cause_v17_20261009/harness/trace_pts_100646_m10.json` | 2.1 | `format`=slt_live_trace_v1 — nested: `frames` |
| `artifacts/generated/xcode_dd_gosplit2/Build/Intermediates.noindex/XCBuildData/3c1cf63e51b8292cbc1dadd573e5c406.xcbuilddata/manifest.json` | 2.1 | nested: `client`, `targets`, `nodes`, `commands` |
| `artifacts/generated/xcode_dd_gosplit/Build/Intermediates.noindex/XCBuildData/7a36427dfc9310f3f26820aaaed07b04.xcbuilddata/manifest.json` | 2.1 | nested: `client`, `targets`, `nodes`, `commands` |
| `artifacts/model_assets/weights/slt_final_t5_model/tokenizer.json` | 2.1 | `version`=1.0; `truncation`=null; `padding`=null; `normalizer`=null — nested: `added_tokens`, `pre_tokenizer`, `post_processor`, `decoder`, `model` |
| `artifacts/models/stage3_v17_reference_replay_v1/tokenizer.json` | 2.1 | `version`=1.0; `normalizer`=null — nested: `truncation`, `padding`, `added_tokens`, `pre_tokenizer`, `post_processor`, `decoder`, `model` |
| `artifacts/reports/my_transition_recovery_v17_20261011/stress_baseline.json` | 2.1 | nested: `compute_ms`, `records` |
| `artifacts/model_assets/models/manifest_v16_files.json` | 2.1 | `ABOUT_019aed92.npy`=ABOUT; `ABOUT_02463f2f.npy`=ABOUT; `ABOUT_02517b30.npy`=ABOUT; `ABOUT_04a9a375.npy`=ABOUT; `ABOUT_04c0299b.npy`=ABOUT; `ABOUT_05e9b4d5.npy`=ABOUT; `ABOUT_067dff10.npy`=ABOUT; `ABOUT_069af01c.npy`=ABOUT |
| `artifacts/reports/o5s5_augmented_v17_20260914/evaluations/epoch_01.json` | 2.1 | `checkpoint`=artifacts/models/stage1_window_o5s5_v17_seed17111/epoch_01.p…; `checkpoint_sha256`=22a898a7d09df42444f7fee02365b482d9fef78512e82b334156fa12239b…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=1; `latency_… |
| `artifacts/reports/stage1_window_v17/mps_parity_epoch01.json` | 2.1 | `checkpoint`=artifacts/models/stage1_window_v17_seed17111/epoch_01.pth; `checkpoint_sha256`=b3916056ec2f755c69b9c71ee167fa2018c923db0f91843639db1efc2839…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=1; `latency_incl… |
| `artifacts/reports/stage1_window_v17/evaluations/epoch_01.json` | 2.1 | `checkpoint`=artifacts/models/stage1_window_v17_seed17111/epoch_01.pth; `checkpoint_sha256`=b3916056ec2f755c69b9c71ee167fa2018c923db0f91843639db1efc2839…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=1; `latency_incl… |
| `artifacts/reports/english_comparison_20260917/manifest.json` | 2.0 | `base_sha256`=25a4e0551b190890712c79be549ed2381386d72d7c6c0611565e8b538125…; `source_manifest_sha256`=5f9f6097689f83f7e71993d94bd56e0cfeb9e423657beb21ee2e48126d24…; `replay_manifest_sha256`=a53bafc7395ef4a1669961468dad3ed22c35d209bba948403a5e3e32204b…; `seed`=… |
| `artifacts/reports/english_comparison_20260917/failed_attempt_02/manifest.json` | 2.0 | `base_sha256`=25a4e0551b190890712c79be549ed2381386d72d7c6c0611565e8b538125…; `source_manifest_sha256`=5f9f6097689f83f7e71993d94bd56e0cfeb9e423657beb21ee2e48126d24…; `replay_manifest_sha256`=a53bafc7395ef4a1669961468dad3ed22c35d209bba948403a5e3e32204b…; `seed`=… |
| `artifacts/reports/english_comparison_20260917/failed_attempt_04/manifest.json` | 2.0 | `base_sha256`=25a4e0551b190890712c79be549ed2381386d72d7c6c0611565e8b538125…; `source_manifest_sha256`=5f9f6097689f83f7e71993d94bd56e0cfeb9e423657beb21ee2e48126d24…; `replay_manifest_sha256`=a53bafc7395ef4a1669961468dad3ed22c35d209bba948403a5e3e32204b…; `seed`=… |
| `artifacts/reports/english_comparison_20260917/failed_attempt_03/manifest.json` | 2.0 | `base_sha256`=25a4e0551b190890712c79be549ed2381386d72d7c6c0611565e8b538125…; `source_manifest_sha256`=5f9f6097689f83f7e71993d94bd56e0cfeb9e423657beb21ee2e48126d24…; `replay_manifest_sha256`=a53bafc7395ef4a1669961468dad3ed22c35d209bba948403a5e3e32204b…; `seed`=… |
| `artifacts/reports/english_comparison_20260917/failed_attempt_01/manifest.json` | 2.0 | `base_sha256`=25a4e0551b190890712c79be549ed2381386d72d7c6c0611565e8b538125…; `source_manifest_sha256`=5f9f6097689f83f7e71993d94bd56e0cfeb9e423657beb21ee2e48126d24…; `replay_manifest_sha256`=a53bafc7395ef4a1669961468dad3ed22c35d209bba948403a5e3e32204b…; `seed`=… |
| `artifacts/reports/stage1_direct_translation_20260917/manifest.json` | 2.0 | `base_sha256`=25a4e0551b190890712c79be549ed2381386d72d7c6c0611565e8b538125…; `text_sha256`=1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683f…; `source_manifest_sha256`=5f9f6097689f83f7e71993d94bd56e0cfeb9e423657beb21ee2e48126d24…; `replay_manifest_s… |
| `artifacts/reports/stage1_direct_translation_20260917/failed_attempt_05/manifest.json` | 2.0 | `base_sha256`=25a4e0551b190890712c79be549ed2381386d72d7c6c0611565e8b538125…; `text_sha256`=1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683f…; `source_manifest_sha256`=5f9f6097689f83f7e71993d94bd56e0cfeb9e423657beb21ee2e48126d24…; `replay_manifest_s… |
| `artifacts/reports/stage1_direct_translation_20260917/failed_attempt_04/manifest.json` | 2.0 | `base_sha256`=25a4e0551b190890712c79be549ed2381386d72d7c6c0611565e8b538125…; `text_sha256`=1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683f…; `source_manifest_sha256`=5f9f6097689f83f7e71993d94bd56e0cfeb9e423657beb21ee2e48126d24…; `replay_manifest_s… |
| `artifacts/reports/stage1_direct_translation_20260917/failed_attempt_03/manifest.json` | 2.0 | `base_sha256`=25a4e0551b190890712c79be549ed2381386d72d7c6c0611565e8b538125…; `text_sha256`=1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683f…; `source_manifest_sha256`=5f9f6097689f83f7e71993d94bd56e0cfeb9e423657beb21ee2e48126d24…; `replay_manifest_s… |
| `artifacts/reports/stage1_direct_translation_20260917/failed_attempt_02/manifest.json` | 2.0 | `base_sha256`=25a4e0551b190890712c79be549ed2381386d72d7c6c0611565e8b538125…; `text_sha256`=1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683f…; `source_manifest_sha256`=5f9f6097689f83f7e71993d94bd56e0cfeb9e423657beb21ee2e48126d24…; `replay_manifest_s… |
| `artifacts/reports/stage1_direct_translation_20260917/failed_attempt_01/manifest.json` | 2.0 | `base_sha256`=25a4e0551b190890712c79be549ed2381386d72d7c6c0611565e8b538125…; `text_sha256`=1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683f…; `source_manifest_sha256`=5f9f6097689f83f7e71993d94bd56e0cfeb9e423657beb21ee2e48126d24…; `replay_manifest_s… |
| `artifacts/reports/live_correctness_v17_20260929/defer_probe.json` | 2.0 | nested: `records` |
| `artifacts/reports/have_omission_v17_20261010/round4.json` | 2.0 | `frames`=654 — nested: `results` |
| `artifacts/reports/live_reel_stage1_v17/20260903_105422_017479/history.json` | 2.0 | `format`=slt_live_reel_stage1_v17_session; `version`=1; `prototype_scope`=growing activity-anchored Stage-1 candidates with label-stab…; `started_utc`=2026-09-03T02:54:22.017811+00:00; `source`=camera:0; `mode`=cascade; `score_semantics`=uncalibrated Stage-1 s… |
| `artifacts/reports/have_root_fix_v17_20261010/closed.json` | 2.0 | `frames`=654 — nested: `results` |
| `artifacts/reports/have_root_fix_v17_20261010/closed2.json` | 2.0 | `frames`=654 — nested: `results` |
| `artifacts/reports/have_omission_v17_20261010/ablation.json` | 2.0 | `frames`=654 — nested: `results` |
| `artifacts/reports/live_correctness_v17_20260929/whole_word_probe.json` | 1.9 | nested: `records` |
| `artifacts/reports/live_reel_stage1_v17/20260902_205147_630595/history.json` | 1.9 | `format`=slt_live_reel_stage1_v17_session; `version`=1; `prototype_scope`=growing activity-anchored Stage-1 candidates with label-stab…; `started_utc`=2026-09-02T12:51:47.630845+00:00; `source`=camera:0; `mode`=cascade; `score_semantics`=uncalibrated Stage-1 s… |
| `artifacts/reports/have_root_fix_v17_20261010/correct_shape.json` | 1.9 | `frames`=654 — nested: `results` |
| `artifacts/reports/stage1_window_v17/evaluations/epoch_02.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_v17_seed17111/epoch_02.pth; `checkpoint_sha256`=6ce2fc96b65ca45d6fd1b6af6a6fb4f762056f97f9bb5083c1df6c298b5b…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=2; `latency_incl… |
| `artifacts/reports/o5s5_augmented_v17_20260914/evaluations/epoch_02.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_o5s5_v17_seed17111/epoch_02.p…; `checkpoint_sha256`=3844f9e8e2d8d31136b8eaa692b4413f6c04d8bdce4916c9e5e6d135d877…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=2; `latency_… |
| `artifacts/reports/stage1_window_v17/evaluations/epoch_11.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_v17_seed17111/epoch_11.pth; `checkpoint_sha256`=d4febe82855c0573ea45afaebc578e14ac9b4fb57cb65a2a58c1bb14eaaa…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=11; `latency_inc… |
| `artifacts/reports/stage1_window_v17/evaluations/epoch_12.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_v17_seed17111/epoch_12.pth; `checkpoint_sha256`=181659a544708b73d2ecadd6b071c8668c462388423e934128782da4f6de…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=12; `latency_inc… |
| `artifacts/reports/stage2_v17_transition_adapt_v2/training_result.json` | 1.9 | `format`=slt_stage2_transition_adapt_training_v17; `experiment_manifest`=artifacts/reports/stage2_v17_transition_adapt_v2/manifest.js…; `experiment_manifest_sha256`=d120d9747ed01bc7d7cc0d68d2609a25114c75826484536ee942c278820d…; `frozen_inputs`=artifacts/report… |
| `artifacts/reports/stage1_window_v17/evaluations/epoch_10.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_v17_seed17111/epoch_10.pth; `checkpoint_sha256`=3e3dd99bea147b4236122040521dbbca6f83068d3fabeb2b406d9ac8c7c7…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=10; `latency_inc… |
| `artifacts/reports/stage2_v17_transition_adapt_v1/training_result.json` | 1.9 | `format`=slt_stage2_transition_adapt_training_v17; `experiment_manifest`=artifacts/reports/stage2_v17_transition_adapt_v1/manifest.js…; `experiment_manifest_sha256`=2404aa96c03f5949d6cd3af914f1e97381c279abd980b2e4c5a37b7ce516…; `frozen_inputs`=artifacts/report… |
| `artifacts/reports/stage1_window_v17/evaluations/epoch_09.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_v17_seed17111/epoch_09.pth; `checkpoint_sha256`=d654390d6d4e600c2e92a527ce236f75b13f691cbb6085d875f436f4dfa8…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=9; `latency_incl… |
| `artifacts/reports/stage1_window_v17/evaluations/epoch_06.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_v17_seed17111/epoch_06.pth; `checkpoint_sha256`=7fcfa3523d63208c4b52fd886fc916802be47aa084fdb6939987554a7ea7…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=6; `latency_incl… |
| `artifacts/reports/stage1_window_v17/evaluations/epoch_08.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_v17_seed17111/epoch_08.pth; `checkpoint_sha256`=0103b38795948d081963fcb473c731eceaa2ef329c26b686fffb4369d3ab…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=8; `latency_incl… |
| `artifacts/reports/stage1_window_v17/evaluations/epoch_05.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_v17_seed17111/epoch_05.pth; `checkpoint_sha256`=aee732e26d549d2a7aa6841a6ae175a9069331f9bad9c45775b9f2846723…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=5; `latency_incl… |
| `artifacts/reports/o5s5_augmented_v17_20260914/evaluations/epoch_05.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_o5s5_v17_seed17111/epoch_05.p…; `checkpoint_sha256`=ba133ba6950928c92a1e86d70cb8509a2a133b5180a61a6b24b45086d835…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=5; `latency_… |
| `artifacts/reports/stage1_window_v17/evaluations/epoch_07.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_v17_seed17111/epoch_07.pth; `checkpoint_sha256`=89ce2ef450c8ce00ffc0819f3b3feaf8dbe1e207355ab0d55538dc99ba4f…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=7; `latency_incl… |
| `artifacts/reports/have_root_fix_v17_20261010/odds.json` | 1.9 | `frames`=654 — nested: `results` |
| `artifacts/reports/o5s5_augmented_v17_20260914/evaluations/epoch_08.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_o5s5_v17_seed17111/epoch_08.p…; `checkpoint_sha256`=733aec333ab9c193a0641b3e431411d2f315830f3554787f65ed21068d7b…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=8; `latency_… |
| `artifacts/reports/o5s5_augmented_v17_20260914/evaluations/epoch_11.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_o5s5_v17_seed17111/epoch_11.p…; `checkpoint_sha256`=d5a0b9b24fce100c333befda36f20f651e660dc6ad96c3870e28eef2b901…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=11; `latency… |
| `artifacts/reports/o5s5_augmented_v17_20260914/evaluations/epoch_12.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_o5s5_v17_seed17111/epoch_12.p…; `checkpoint_sha256`=9e51ae27298369fb0ddff6e755ec3d1afc10fa80b069ca65bfd2b04c1b78…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=12; `latency… |
| `artifacts/reports/o5s5_augmented_v17_20260914/evaluations/epoch_06.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_o5s5_v17_seed17111/epoch_06.p…; `checkpoint_sha256`=96220f7d54760d0ce020e51bfdeee700f4fe760d94cb6a74485792aed504…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=6; `latency_… |
| `artifacts/reports/stage1_window_v17/evaluations/epoch_04.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_v17_seed17111/epoch_04.pth; `checkpoint_sha256`=e65f6a2bcb9983a9642951cc11dd5627c822ebfc03e935c01cd13f4b5bec…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=4; `latency_incl… |
| `artifacts/reports/o5s5_augmented_v17_20260914/evaluations/epoch_10.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_o5s5_v17_seed17111/epoch_10.p…; `checkpoint_sha256`=be50d3c65103ac2bbfd1ef036ac0d0582600b25182012f3b639f076962a8…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=10; `latency… |
| `artifacts/reports/stage1_window_v17/evaluations/epoch_03.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_v17_seed17111/epoch_03.pth; `checkpoint_sha256`=3c8b615729d3883a9adb1da5eb518a72ad33f69b98fab5d2a12f9dde3de1…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=3; `latency_incl… |
| `artifacts/reports/o5s5_augmented_v17_20260914/evaluations/epoch_09.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_o5s5_v17_seed17111/epoch_09.p…; `checkpoint_sha256`=82af1e928cbea43d6aaf99ee1d32859438c246099041b49f7346ec12ec8b…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=9; `latency_… |
| `artifacts/reports/o5s5_augmented_v17_20260914/evaluations/epoch_07.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_o5s5_v17_seed17111/epoch_07.p…; `checkpoint_sha256`=1d8432d5472e9e8a1f4ba67f57c704c5ae33f439547b1c3c7996d44b1043…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=7; `latency_… |
| `artifacts/reports/o5s5_augmented_v17_20260914/evaluations/epoch_04.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_o5s5_v17_seed17111/epoch_04.p…; `checkpoint_sha256`=be64faba2b6e024c61ba0784232bb3016e58049f4b662b6c73459546e1b9…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=4; `latency_… |
| `artifacts/reports/o5s5_augmented_v17_20260914/evaluations/epoch_03.json` | 1.9 | `checkpoint`=artifacts/models/stage1_window_o5s5_v17_seed17111/epoch_03.p…; `checkpoint_sha256`=c2fa4f3c53a0f5b790c190b8692dc6ffdfd2bf059667675d3eafaab9396c…; `manifest_sha256`=f1ad5152d2f8126c9bcf365b66559e7b742ccd6e5c350b51241ba5637d37…; `epoch`=3; `latency_… |
| `artifacts/reports/stage2_v17_general_specialist_analysis.json` | 1.8 | nested: `phrase_train`, `phrase_validation`, `context_train`, `context_validation` |
| `artifacts/reports/my_transition_recovery_v17_20261011/balanced_new.json` | 1.8 | nested: `compute_ms`, `records` |
| `artifacts/reports/live_correctness_v17_20260929/rejected_word_rescue.json` | 1.8 | nested: `records` |
| `artifacts/reports/o5s5_augmented_v17_20260914/development_freeze.json` | 1.8 | `format`=stage1_window_development_freeze_v17; `ctc_baseline`=artifacts/reports/stage2_v17_revisable_v1/baseline_evaluatio…; `old_reel_baseline_status`=complete; `candidate_seed`=17111; `confirmation_seed`=17112; `confirmation_requires_first_seed_eligible`=tru… |
| `artifacts/reports/my_transition_recovery_v17_20261011/confirm_new.json` | 1.8 | nested: `compute_ms`, `records` |
| `artifacts/reports/stage1_window_v17/development_freeze.json` | 1.8 | `format`=stage1_window_development_freeze_v17; `ctc_baseline`=artifacts/reports/stage2_v17_revisable_v1/baseline_evaluatio…; `old_reel_baseline_status`=complete; `candidate_seed`=17111; `confirmation_seed`=17112; `confirmation_requires_first_seed_eligible`=tru… |
| `artifacts/reports/my_transition_recovery_v17_20261011/stable_boundary_negative.json` | 1.8 | nested: `records` |
| `artifacts/reports/have_root_fix_v17_20261010/lag.json` | 1.8 | `frames`=654 — nested: `results` |
| `artifacts/generated/xcode_dd_gosplit_devtest/Build/Intermediates.noindex/XCBuildData/d623ffd61dddea8d521785bfdebc5412.xcbuilddata/manifest.json` | 1.8 | nested: `client`, `targets`, `nodes`, `commands` |
| `artifacts/app_sessions/20260922_100618_237589/history.json` | 1.8 | `format`=slt_live_reel_stage1_v17_session; `version`=1; `prototype_scope`=growing activity-anchored Stage-1 candidates with label-stab…; `started_utc`=2026-09-22T02:06:18.237813+00:00; `source`=camera:0; `mode`=cascade; `score_semantics`=uncalibrated Stage-1 s… |
| `artifacts/generated/kaggle_stage1_partaux_w020_kokoab_pull_v2/SLT/data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json` | 1.8 | `created_utc`=2026-08-10T08:09:32.306610+00:00; `purpose`=quality-ranked signer-balanced SemLex train-only supplement; `training_eligible`=false; `split`=train_only; `cap_per_class`=null; `quality_passing_before_cap`=1388; `quality_rejected_or_mismatch_priorit… |
| `artifacts/generated/kaggle_stage1_challengers_v2/tree_probe/data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json` | 1.8 | `created_utc`=2026-08-10T08:09:32.306610+00:00; `purpose`=quality-ranked signer-balanced SemLex train-only supplement; `training_eligible`=false; `split`=train_only; `cap_per_class`=null; `quality_passing_before_cap`=1388; `quality_rejected_or_mismatch_priorit… |
| `artifacts/reports/pose_boundary_transfer_v17_20260922/prepared_manifest.json` | 1.7 | `status`=complete; `training_ready`=false; `training_blocker`=Review bounded-context normalization and supervised start/en…; `source_manifest`=active/v17/temporal_boundary_manifest_20260922.json; `source_sha256`=25b24499633df10cebf69bbe99f49f571c020986a8145f01… |
| `artifacts/reports/my_transition_recovery_v17_20261011/narrow_final_genuine.json` | 1.7 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/preserved_genuine.json` | 1.7 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/joint_genuine.json` | 1.7 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/merge_genuine.json` | 1.7 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/context_final_genuine.json` | 1.7 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/jitter_genuine.json` | 1.7 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/ambiguous_genuine.json` | 1.7 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/confidence_genuine.json` | 1.7 | nested: `compute_ms`, `records` |
| `artifacts/reports/stage2_v17_live_matched_v1/frozen_inputs.json` | 1.7 | nested: `contract`, `manifests`, `rows`, `counts` |
| `artifacts/reports/my_transition_recovery_v17_20261011/calibrated_genuine.json` | 1.7 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/stable_boundary_screen.json` | 1.7 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/rank_swap_genuine.json` | 1.6 | nested: `compute_ms`, `records` |
| `artifacts/reports/canonical_recognition_comparison_v17_20261007/phone_results.json` | 1.6 | nested: `summary`, `runs` |
| `artifacts/reports/my_transition_recovery_v17_20261011/distinct_screen.json` | 1.6 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/contact_screen.json` | 1.6 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/veto40_new.json` | 1.6 | nested: `compute_ms`, `records` |
| `artifacts/reports/clean_lineage_recovery_v17_20261011/trace_spelling.json.trace.json` | 1.6 | array, 323 items of objects keyed `start` |
| `artifacts/reports/clean_lineage_recovery_v17_20261011/trace_always.json.trace.json` | 1.6 | array, 323 items of objects keyed `start` |
| `artifacts/reports/clean_lineage_recovery_v17_20261011/trace_replay.json.trace.json` | 1.6 | array, 327 items of objects keyed `start` |
| `artifacts/reports/my_transition_recovery_v17_20261011/paired_new.json` | 1.6 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/hybrid_new.json` | 1.6 | nested: `compute_ms`, `records` |
| `artifacts/reports/clean_lineage_recovery_v17_20261011/letter_recipe.json` | 1.6 | `scope`=letter-head-only; frozen existing clean recognizer; `phrase_recipe_sha256`=cb9a91b0fc1845d901dfa0fa0a7bca435a9a053c3c6b62f4f28c93dcf047…; `recognizer_sha256`=b3d3134f40f3f2c31fa57a827a1137651a96c5654c9e7803ead8dee7ef6a…; `train_videos`=179; `validation… |
| `artifacts/reports/clean_lineage_recovery_v17_20261011/adapted_bundle/letter_recipe.json` | 1.6 | `scope`=letter-head-only; frozen existing clean recognizer; `phrase_recipe_sha256`=cb9a91b0fc1845d901dfa0fa0a7bca435a9a053c3c6b62f4f28c93dcf047…; `recognizer_sha256`=969b090323c5fc4b199de4edf5e2d08f365c63fd1cc4998e517afed59caf…; `train_videos`=179; `validation… |
| `artifacts/reports/clean_lineage_recovery_v17_20261011/adaptation_recipe.json` | 1.5 | `scope`=frozen-encoder fusion-head continuation on approved native p…; `epoch_zero_eligible_for_selection`=true; `no_synthetic_fragment_or_reject_views`=true; `manifest_sha256`=cb9a91b0fc1845d901dfa0fa0a7bca435a9a053c3c6b62f4f28c93dcf047…; `trainer_sha256`=1a8… |
| `artifacts/reports/my_like_regression_v17_20261011/dump.json.inputs.json` | 1.5 | array, 4 items of objects keyed `boxes`, `embeddings`, `end`, `landmarks`, `pad`, `start`, `valid` |
| `artifacts/generated/xcode_dd_gosplit_release/Build/Intermediates.noindex/XCBuildData/dd31f3653b10029f3d1e1f24813b960f.xcbuilddata/manifest.json` | 1.5 | nested: `client`, `targets`, `nodes`, `commands` |
| `artifacts/generated/xcode_dd_gosplit_release/Build/Intermediates.noindex/XCBuildData/135e320aad2c1dc58ee58874c200c3d1.xcbuilddata/manifest.json` | 1.5 | nested: `client`, `targets`, `nodes`, `commands` |
| `artifacts/reports/good_thankyou_variant_audit_v17_20261008/two_handed_good_inputs/semlex_train_two_handed_good.json` | 1.5 | `created_utc`=2026-08-10T08:09:32.306610+00:00; `purpose`=quality-ranked signer-balanced SemLex train-only supplement; `training_eligible`=false; `split`=train_only; `cap_per_class`=null; `quality_passing_before_cap`=1388; `quality_rejected_or_mismatch_priorit… |
| `artifacts/generated/xcode_dd_gosplit_release/Build/Intermediates.noindex/XCBuildData/bd32c94b1aa106488a8b5f7e20ed8806.xcbuilddata/manifest.json` | 1.5 | nested: `client`, `targets`, `nodes`, `commands` |
| `artifacts/generated/xcode_dd_gosplit_release/Build/Intermediates.noindex/XCBuildData/6ce3fd16d09be02c54f36e76b6b3a05c.xcbuilddata/manifest.json` | 1.5 | nested: `client`, `targets`, `nodes`, `commands` |
| `artifacts/generated/xcode_dd_minspan_release/Build/Intermediates.noindex/XCBuildData/14cee22f90c6e974246f8c390890322f.xcbuilddata/manifest.json` | 1.5 | nested: `client`, `targets`, `nodes`, `commands` |
| `artifacts/reports/supplement_finalization_v17_20260922/asllrp_segmented.json` | 1.4 | `format`=finalized_supervised_supplement_v17; `source`=asllrp_segmented; `signer_policy`=preserve existing roles; overlap does not disqualify — nested: `counts`, `records`, `evidence_sha256`, `exclusions` |
| `artifacts/reports/boundary_tcn_sweep_v17_20260922/round3_power.json` | 1.4 | `round`=2; `objective`=output formulation and supervision coverage at a fixed trunk…; `gap_max_seconds`=0.4; `hand_geometry`=true — nested: `trunk`, `seeds`, `reel`, `results` |
| `artifacts/reports/signwriting_gloss_bank_v17_v5b/report.json` | 1.4 | `format`=slt_avatar_gloss_bank_v17; `symbol_motion`=true; `manifest_sha256`=3a665bda8d2b916c504406be815e601eeb55badfe62afcec42c7869885ea…; `lexicon_sha256`=1cdb86846dc0f3cba595864fb00ec0a2ccc5068caf13f59b8a69e97d2f6d…; `dictionary_sha256`=6583d313fc71ad278af88… |
| `artifacts/reports/my_transition_recovery_v17_20261011/gated_swap_new.json` | 1.4 | nested: `compute_ms`, `records` |
| `artifacts/reports/signwriting_gloss_bank_v17_v4/report.json` | 1.4 | `format`=slt_avatar_gloss_bank_v17; `symbol_motion`=true; `manifest_sha256`=3a665bda8d2b916c504406be815e601eeb55badfe62afcec42c7869885ea…; `lexicon_sha256`=1cdb86846dc0f3cba595864fb00ec0a2ccc5068caf13f59b8a69e97d2f6d…; `dictionary_sha256`=6583d313fc71ad278af88… |
| `artifacts/reports/signwriting_gloss_bank_v17_v3/report.json` | 1.4 | `format`=slt_avatar_gloss_bank_v17; `symbol_motion`=true; `manifest_sha256`=3a665bda8d2b916c504406be815e601eeb55badfe62afcec42c7869885ea…; `lexicon_sha256`=1cdb86846dc0f3cba595864fb00ec0a2ccc5068caf13f59b8a69e97d2f6d…; `dictionary_sha256`=6583d313fc71ad278af88… |
| `artifacts/reports/supplement_finalization_v17_20260922/citizen.json` | 1.4 | `format`=finalized_supervised_supplement_v17; `source`=citizen; `signer_policy`=preserve existing roles; overlap does not disqualify — nested: `counts`, `records`, `evidence_sha256`, `exclusions` |
| `artifacts/reports/atlas_canonical_source_20261013/before_sources/active/v17/temporal_boundary_manifest_20260922.json` | 1.4 | `format`=slt_temporal_boundary_v17; `training_entrypoint`=scripts/train_temporal_boundary_v17.py; `training_ready`=true; `scope`=boundary edges only; no lexical targets, background or phras…; `supervision`=start/end positives; negatives only inside accepted in… |
| `artifacts/reports/my_transition_recovery_v17_20261011/gated_pair_new.json` | 1.4 | nested: `compute_ms`, `records` |
| `artifacts/reports/asl_temporal_boundary_v17_20260922/recipe_before_final_review.json` | 1.4 | `format`=slt_temporal_boundary_v17; `training_entrypoint`=scripts/train_temporal_boundary_v17.py; `training_ready`=true; `scope`=boundary edges only; no lexical targets, background or phras…; `supervision`=start/end positives; negatives only inside accepted in… |
| `artifacts/reports/asl_temporal_boundary_v17_20260922/recipe_before_review.json` | 1.4 | `format`=slt_temporal_boundary_v17; `training_entrypoint`=scripts/train_temporal_boundary_v17.py; `training_ready`=true; `scope`=boundary edges only; no lexical targets, background or phras…; `supervision`=start/end positives; negatives only inside accepted in… |
| `artifacts/reports/my_transition_recovery_v17_20261011/bounded_new.json` | 1.4 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/positive_screen.json` | 1.4 | nested: `compute_ms`, `records` |
| `artifacts/reports/have_root_fix_v17_20261010/ordered.json` | 1.4 | `frames`=654 — nested: `results` |
| `artifacts/reports/signwriting_gloss_bank_v17_v2/report.json` | 1.3 | `format`=slt_avatar_gloss_bank_v17; `symbol_motion`=true; `manifest_sha256`=3a665bda8d2b916c504406be815e601eeb55badfe62afcec42c7869885ea…; `lexicon_sha256`=1cdb86846dc0f3cba595864fb00ec0a2ccc5068caf13f59b8a69e97d2f6d…; `dictionary_sha256`=6583d313fc71ad278af88… |
| `artifacts/reports/signwriting_gloss_bank_v17_v1/report.json` | 1.3 | `format`=slt_avatar_gloss_bank_v17; `symbol_motion`=true; `manifest_sha256`=3a665bda8d2b916c504406be815e601eeb55badfe62afcec42c7869885ea…; `lexicon_sha256`=1cdb86846dc0f3cba595864fb00ec0a2ccc5068caf13f59b8a69e97d2f6d…; `dictionary_sha256`=6583d313fc71ad278af88… |
| `artifacts/reports/have_root_fix_v17_20261010/closed3.json` | 1.3 | `frames`=654 — nested: `results` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__EXCITED__06119477404825924-EXCITED.json` | 1.3 | nested: `frames` |
| `artifacts/reports/pose_boundary_transfer_v17_20260922/prepared_manifest_serial.json` | 1.3 | `status`=preparing; `training_ready`=false; `training_blocker`=Review bounded-context normalization and supervised start/en…; `source_manifest`=active/v17/temporal_boundary_manifest_20260922.json; `source_sha256`=25b24499633df10cebf69bbe99f49f571c020986a8145f0… |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__SAD__04551864995837951-SAD.json` | 1.3 | nested: `frames` |
| `artifacts/reports/my_transition_recovery_v17_20261011/successor_identity_screen.json` | 1.3 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/confirm_genuine.json` | 1.3 | nested: `compute_ms`, `records` |
| `artifacts/reports/paper_page_check_update_20261013/page_check_before.json` | 1.3 | `tabId`=t.8u4bn2h16og9; `title`=Page Check (Oct 6) Revision; `revisionId`=AHj4eMQsngo-q9XW0rvovSrUopc2OZ5kH1lRWVgnxXfqy0OKF2RyC9oTKz2t… — nested: `body`, `inlineObjects` |
| `artifacts/reports/signwriting_cwasa_comparison_v17_v1/need_bend_probe/cdp_capture.json` | 1.3 | `ready`=complete; `timerFrameFallback`=true; `record`={"events":[{"type":"status","msg":"Animgen by Client"},{"typ…; `body`=NEED — SignWriting to SiGML pilot Unverified notation conver… |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__TALK__873026373390853-TALK.json` | 1.2 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__YEAR__17164656138899548-YEAR.json` | 1.2 | nested: `frames` |
| `artifacts/reports/my_transition_recovery_v17_20261011/veto40_genuine.json` | 1.2 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__COLD__39072948576699607-COLD.json` | 1.2 | nested: `frames` |
| `artifacts/reports/my_transition_recovery_v17_20261011/hybrid_genuine.json` | 1.2 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/contact_negative.json` | 1.2 | nested: `records` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__WATER__03174577769237463-WATER.json` | 1.2 | nested: `frames` |
| `artifacts/reports/my_transition_recovery_v17_20261011/global_localA_genuine.json` | 1.2 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__LISTEN__1493429103533752-LISTEN.json` | 1.2 | nested: `frames` |
| `artifacts/reports/my_transition_recovery_v17_20261011/distinct_negative.json` | 1.2 | nested: `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/bounded_genuine.json` | 1.2 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/gated_pair_genuine.json` | 1.2 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/gated_swap_genuine.json` | 1.2 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/decision_genuine.json` | 1.2 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__HOME__049094629596487804-HOME.json` | 1.2 | nested: `frames` |
| `artifacts/reports/my_transition_recovery_v17_20261011/paired_genuine.json` | 1.2 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/balanced_genuine.json` | 1.2 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__MORNING__5178361322749898-MORNING.json` | 1.2 | nested: `frames` |
| `artifacts/reports/stage2_v17_transition_diagnosis_v1/training_detection.json` | 1.2 | nested: `rows`, `models` |
| `artifacts/reports/my_transition_recovery_v17_20261011/global_current_genuine.json` | 1.2 | nested: `compute_ms`, `records` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__YOUR__7464906167030079-YOUR.json` | 1.2 | nested: `frames` |
| `artifacts/reports/my_transition_recovery_v17_20261011/negative_no_boundary.json` | 1.2 | nested: `records` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__CHILD__08822684142689163-CHILD.json` | 1.2 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__YESTERDAY__004997545233198508-YESTERDAY.json` | 1.2 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__FEEL__9861411419156798-FEEL.json` | 1.2 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__MAKE__16526106983233912-MAKE.json` | 1.1 | nested: `frames` |
| `artifacts/reports/stage2_v17_transition_repair_v3/validation.json` | 1.1 | `format`=slt_stage2_other_preservation_validation_v17; `selected_checkpoint`=/Users/frnzlo/Documents/machine_learning/SLT/artifacts/model…; `citizen_test_accessed`=false; `semlex_test_accessed`=false; `local_test_accessed`=false; `disclosure`=Repeated developm… |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__LEARN__9349267580836926-LEARN.json` | 1.1 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__SLEEP__5103688507862052-SLEEP.json` | 1.1 | nested: `frames` |
| `artifacts/reports/live_reel_stage1_v17/20260903_113617_134428/history.json` | 1.1 | `format`=slt_live_reel_stage1_v17_session; `version`=1; `prototype_scope`=growing activity-anchored Stage-1 candidates with label-stab…; `started_utc`=2026-09-03T03:36:17.134693+00:00; `source`=camera:0; `mode`=cascade; `score_semantics`=uncalibrated Stage-1 s… |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__HAPPY__3698632660671215-HAPPY.json` | 1.1 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__MORE__9000660794636397-MORE.json` | 1.1 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__ASK__0773952609683628-ASK.json` | 1.1 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__MAN__7254166313904942-MAN.json` | 1.1 | nested: `frames` |
| `artifacts/reports/my_transition_recovery_v17_20261011/triple_cadence.json` | 1.1 | nested: `records` |
| `artifacts/reports/my_transition_recovery_v17_20261011/triple_merge.json` | 1.1 | nested: `records` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__EAT__009883456113471967-EAT.json` | 1.1 | nested: `frames` |
| `artifacts/reports/live_reel_continuous_v17/20260921_105203_986380/history.json` | 1.1 | `format`=slt_live_reel_stage1_v17_session; `version`=1; `prototype_scope`=growing activity-anchored Stage-1 candidates with label-stab…; `started_utc`=2026-09-21T02:52:03.986571+00:00; `source`=camera:0; `mode`=cascade; `score_semantics`=uncalibrated Stage-1 s… |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__SICK__45273738498753846-SICK.json` | 1.1 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__TIME__6677797392644389-TIME.json` | 1.1 | nested: `frames` |
| `artifacts/app_sessions/20260922_081431_659584/history.json` | 1.1 | `format`=slt_live_reel_stage1_v17_session; `version`=1; `prototype_scope`=growing activity-anchored Stage-1 candidates with label-stab…; `started_utc`=2026-09-22T00:14:31.659831+00:00; `source`=camera:0; `mode`=cascade; `score_semantics`=uncalibrated Stage-1 s… |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__WANT__8782648080433364-WANT 2.json` | 1.1 | nested: `frames` |
| `artifacts/reports/citizen_variant_local_filter_v17_20261008/commit_replay_decoder_variants/results.json` | 1.1 | nested: `installed`, `no_letters_in_word_decoder`, `words_only` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__IMPORTANT__3510284242642503-IMPORTANT.json` | 1.1 | nested: `frames` |
| `artifacts/reports/live_reel_emission_stage1_v17/20260903_083154_858313/history.json` | 1.1 | `format`=slt_live_reel_stage1_v17_session; `version`=1; `prototype_scope`=growing activity-anchored Stage-1 candidates with label-stab…; `started_utc`=2026-09-03T00:31:54.859162+00:00; `source`=camera:0; `mode`=cascade; `score_semantics`=uncalibrated Stage-1 s… |
| `artifacts/reports/live_reel_emission_stage1_v17/20260903_074154_998790/history.json` | 1.1 | `format`=slt_live_reel_stage1_v17_session; `version`=1; `prototype_scope`=growing activity-anchored Stage-1 candidates with label-stab…; `started_utc`=2026-09-02T23:41:54.999141+00:00; `source`=camera:0; `mode`=cascade; `score_semantics`=uncalibrated Stage-1 s… |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__TELL__030859863529683285-TELL.json` | 1.1 | nested: `frames` |
| `artifacts/reports/my_transition_recovery_v17_20261011/joint_phase.json` | 1.1 | nested: `records` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__NEED__30879418056454777-NEED.json` | 1.1 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/tune__69037c74a20a790763ad.json` | 1.1 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__NAME__9031073980623976-NAME.json` | 1.1 | nested: `frames` |
| `artifacts/reports/slt_stage3_dataset_final.csv` | 1.1 | 15843 rows — columns: `gloss`, `text` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__ANSWER__17262273091142677-ANSWER.json` | 1.1 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/tune__da05efb9120efb89ab29.json` | 1.1 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__MAYBE__03356585888825525-MAYBE.json` | 1.0 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/tune__d7e301b2b67fa4546722.json` | 1.0 | nested: `frames` |
| `artifacts/reports/atlas_canonical_source_20261013/before_sources/active/v17/approved_phrase_manifest_20260921_v2.json` | 1.0 | `format`=approved_phrase_data_v17; `version`=2; `training_ready`=false; `scope`=Phrase admission only; no certification of isolated, RGB, bl…; `policy`=Exclude entire ambiguous/clipped/overlapping sequences; neve…; `salvage_policy`=Gap-free complete establishe… |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__WORK__4328643376522927-WORK.json` | 1.0 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__NOW__8203203237125376-NOW.json` | 1.0 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__HOSPITAL__5851960225980641-HOSPITAL.json` | 1.0 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__THEY__38731487247930807-THEY.json` | 1.0 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__EASY__38489622716415495-EASY.json` | 1.0 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/tune__fa25a7e6648e35cf24e0.json` | 1.0 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__LANGUAGE__8054077229465761-LANGUAGE.json` | 1.0 | nested: `frames` |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__SMALL__8815573731027433-SMALL.json` | 1.0 | nested: `frames` |
| `artifacts/app_sessions/20260922_082406_979339/history.json` | 1.0 | `format`=slt_live_reel_stage1_v17_session; `version`=1; `prototype_scope`=growing activity-anchored Stage-1 candidates with label-stab…; `started_utc`=2026-09-22T00:24:06.979596+00:00; `source`=camera:0; `mode`=cascade; `score_semantics`=uncalibrated Stage-1 s… |
| `artifacts/reports/my_have_joint_experiment_v17_20261011/validation_traces/citizen_val__ANGRY__9167479595954919-ANGRY.json` | 1.0 | nested: `frames` |

