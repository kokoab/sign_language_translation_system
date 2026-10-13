# ATLAS software test case register — October 13, 2026

47 selected cases; 47 passed, 0 failed, 0 skipped in their final recorded executions. Initial native failure and correction retained in logs.

| ID | Group | Framework | Test method | Final outcome | Evidence |
|---|---|---|---|---|---|
| PY-01 | WB-01 | Python unittest | test_normalization_keeps_missing_nodes_exactly_zero | Passed | python.log |
| PY-02 | WB-01 | Python unittest | test_resampling_preserves_binary_presence | Passed | python.log |
| PY-03 | WB-01 | Python unittest | test_isotropic_coordinates_match_portrait_and_landscape | Passed | python.log |
| PY-04 | WB-01 | Python unittest | test_gap_interpolation_fills_every_joint_without_partial_zero_bug | Passed | python.log |
| PY-05 | WB-01 | Python unittest | test_rotation_and_unmirror_are_exactly_reversible | Passed | python.log |
| PY-06 | WB-01 | Python unittest | test_image_limit_preserves_aspect_and_never_upscales | Passed | python.log |
| PY-07 | WB-01 | Python unittest | test_result_round_trip_enforces_schema | Passed | python.log |
| PY-08 | WB-03 | Python unittest | test_partwise_global_model_is_finite_and_mobile_sized | Passed | python.log |
| PY-09 | WB-03 | Python unittest | test_partwise_temporal_projection_is_feature_isolated | Passed | python.log |
| PY-10 | WB-03 | Python unittest | test_small_unified_model_has_four_runtime_inputs | Passed | python.log |
| PY-11 | WB-03 | Python unittest | test_right_hand_and_union_keep_their_slots_and_rest_skips_encoder | Passed | python.log |
| PY-12 | WB-04 | Python unittest | test_stream_plus_flush_equals_clip | Passed | python.log |
| PY-13 | WB-04 | Python unittest | test_two_signs_separated_by_rest | Passed | python.log |
| PY-14 | WB-04 | Python unittest | test_repeated_movement_collapses | Passed | python.log |
| PY-15 | WB-04 | Python unittest | test_low_confidence_is_not_emitted | Passed | python.log |
| PY-16 | WB-04 | Python unittest | test_contact_requires_new_evidence_and_rejects_a_lowering_tail | Passed | python.log |
| PY-17 | WB-04 | Python unittest | test_completed_contact_survives_angry_exit_and_big_crossing_span | Passed | python.log |
| PY-18 | WB-04 | Python unittest | test_early_big_cannot_jump_a_pending_have | Passed | python.log |
| PY-19 | WB-04 | Python unittest | test_whole_sign_span_needs_max_inner_o | Passed | python.log |
| PY-20 | WB-04 | Python unittest | test_inner_floor_prefers_the_whole_sign | Passed | python.log |
| PY-21 | WB-04 | Python unittest | test_adjacent_repeats_hold_reset_and_no_fabricated_eof | Passed | python.log |
| PY-22 | WB-04 | Python unittest | test_repeated_start_without_end_does_not_create_a_completed_sign | Passed | python.log |
| PY-23 | WB-04 | Python unittest | test_stream_probabilities_match_complete_prefix_without_future_leakage | Passed | python.log |
| PY-24 | WB-06 | Python unittest | test_model_receives_how_you_together_across_long_pause | Passed | python.log |
| PY-25 | WB-06 | Python unittest | test_torch_contract_uses_model_even_for_old_reviewed_phrase | Passed | python.log |
| PY-26 | WB-06 | Python unittest | test_coreml_contract_uses_model_even_for_old_reviewed_phrase | Passed | python.log |
| PY-27 | WB-06 | Python unittest | test_coreml_overlong_input_falls_back_without_silent_truncation | Passed | python.log |
| PY-28 | WB-06 | Python unittest | test_slots_are_restored_exactly | Passed | python.log |
| PY-29 | WB-06 | Python unittest | test_dropped_slot_falls_back_to_literal | Passed | python.log |
| SW-01 | WB-01 | XCTest | testLandmarkCaptureRotation | Passed | swift.log |
| SW-02 | AI-04 | XCTest | testAllPracticeReferencesPlayAndLoop | Passed | swift.log |
| SW-03 | AI-03 | XCTest | testCameraLayoutsLandmarkPreferenceAndReferenceOverlay | Passed | swift.log |
| SW-04 | WB-02 | XCTest | testLatestFrameDropsBacklogAndInvalidatesStoppedWork | Passed | swift.log |
| SW-05 | WB-05 | XCTest | testFinishPoseRightOpenLeftFistAboveShoulders | Passed | swift.log |
| SW-06 | WB-04 | XCTest | testWordModeBridgesRestDipInsideSign | Passed | swift.log |
| SW-07 | WB-04 | XCTest | testTiredRescueMatchesPython | Passed | swift.log |
| SW-08 | AI-02 | XCTest | testCameraToolbarOmitsRemovedControls | Passed | swift_retest.log |
| SW-09 | AI-02 | XCTest | testLiveAndPracticeSupportPortraitAndLandscapeLayout | Passed | swift_retest.log |
| FL-01 | AI-01 | Flutter Test | ATLAS shell shows the splash, then Home with its four entry points | Passed | flutter.jsonl |
| FL-02 | AI-01 | Flutter Test | Home opens Practice setup, Glosses and History, and back returns Home | Passed | flutter.jsonl |
| FL-03 | AI-01 | Flutter Test | Home opens the manual; a section leads to the next and back returns Home | Passed | flutter.jsonl |
| FL-04 | AI-02 | Flutter Test | every manual section lays out on a phone | Passed | flutter.jsonl |
| FL-05 | AI-05 | Flutter Test | session detail lists recognized signs and the timeline | Passed | flutter.jsonl |
| FL-06 | WB-07 | Flutter Test | history rows are described like the desktop shell | Passed | flutter.jsonl |
| FL-07 | WB-07 | Flutter Test | native inference result contract parses single and phrase output | Passed | flutter.jsonl |
| KT-01 | WB-04 | JUnit | configuredBoundaryDipKeepsWholeGestureCandidate | Passed | kotlin_results.xml |
| KT-02 | WB-04 | JUnit | delayedClosureDoesNotRewindWordEndpoints | Passed | kotlin_results.xml |
