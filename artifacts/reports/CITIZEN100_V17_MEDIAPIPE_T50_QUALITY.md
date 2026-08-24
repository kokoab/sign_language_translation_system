# Citizen100 v17 Landmark Availability Audit

- Feature root: `data/local/citizen100_v17/landmarks_mediapipe_t50`
- Archives measured: 1852
- Clips below 25% hand-active output frames: 1
- Values below are min / p10 / median / p90 / max.
- This audit measures missingness and confidence. Visual overlay review is
  still required to judge geometric placement or semantic correctness.

## Clip-level metrics

| Metric | Min / p10 / median / p90 / max |
| --- | --- |
| `hand_active_output_frames` | 0.1562 / 0.7500 / 0.8750 / 0.8750 / 1.0000 |
| `left_active_output_frames` | 0.0000 / 0.0000 / 0.5938 / 0.8750 / 1.0000 |
| `right_active_output_frames` | 0.0000 / 0.1562 / 0.8125 / 0.8750 / 0.9688 |
| `both_hands_active_output_frames` | 0.0000 / 0.0000 / 0.0938 / 0.8125 / 0.9062 |
| `left_joint_completeness_when_active` | 0.0000 / 0.0000 / 1.0000 / 1.0000 / 1.0000 |
| `right_joint_completeness_when_active` | 0.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000 |
| `observed_point_confidence_mean` | 0.5529 / 0.7175 / 0.7833 / 0.8411 / 0.9410 |
| `detected_hand_source_frames_before_trim` | 0.0893 / 0.2667 / 0.4138 / 0.5455 / 1.0000 |
| `detected_hand_source_frames_after_trim` | 0.1724 / 0.7333 / 0.8598 / 0.9070 / 1.0000 |

## Per-node presence

Presence counts include explicit bounded interpolation, but never Kalman
extrapolation. A missing point remains zero and contributes no motion feature.

| Node | Presence fraction |
| --- | ---: |
| `left_hand_wrist` | 0.4544 |
| `left_hand_thumb_cmc` | 0.4544 |
| `left_hand_thumb_mp` | 0.4544 |
| `left_hand_thumb_ip` | 0.4544 |
| `left_hand_thumb_tip` | 0.4544 |
| `left_hand_index_mcp` | 0.4544 |
| `left_hand_index_pip` | 0.4544 |
| `left_hand_index_dip` | 0.4544 |
| `left_hand_index_tip` | 0.4544 |
| `left_hand_middle_mcp` | 0.4544 |
| `left_hand_middle_pip` | 0.4544 |
| `left_hand_middle_dip` | 0.4544 |
| `left_hand_middle_tip` | 0.4544 |
| `left_hand_ring_mcp` | 0.4544 |
| `left_hand_ring_pip` | 0.4544 |
| `left_hand_ring_dip` | 0.4544 |
| `left_hand_ring_tip` | 0.4544 |
| `left_hand_little_mcp` | 0.4544 |
| `left_hand_little_pip` | 0.4544 |
| `left_hand_little_dip` | 0.4544 |
| `left_hand_little_tip` | 0.4544 |
| `right_hand_wrist` | 0.7067 |
| `right_hand_thumb_cmc` | 0.7067 |
| `right_hand_thumb_mp` | 0.7067 |
| `right_hand_thumb_ip` | 0.7067 |
| `right_hand_thumb_tip` | 0.7067 |
| `right_hand_index_mcp` | 0.7067 |
| `right_hand_index_pip` | 0.7067 |
| `right_hand_index_dip` | 0.7067 |
| `right_hand_index_tip` | 0.7067 |
| `right_hand_middle_mcp` | 0.7067 |
| `right_hand_middle_pip` | 0.7067 |
| `right_hand_middle_dip` | 0.7067 |
| `right_hand_middle_tip` | 0.7067 |
| `right_hand_ring_mcp` | 0.7067 |
| `right_hand_ring_pip` | 0.7067 |
| `right_hand_ring_dip` | 0.7067 |
| `right_hand_ring_tip` | 0.7067 |
| `right_hand_little_mcp` | 0.7067 |
| `right_hand_little_pip` | 0.7067 |
| `right_hand_little_dip` | 0.7067 |
| `right_hand_little_tip` | 0.7067 |
| `left_pupil` | 0.7795 |
| `right_pupil` | 0.7795 |
| `left_brow_start` | 0.7795 |
| `left_brow_end` | 0.7795 |
| `right_brow_start` | 0.7795 |
| `right_brow_end` | 0.7795 |
| `nose_tip` | 0.7795 |
| `mouth_left` | 0.7795 |
| `mouth_right` | 0.7795 |
| `upper_lip` | 0.7795 |
| `lower_lip` | 0.7795 |
| `jaw_left` | 0.7795 |
| `chin` | 0.7795 |
| `jaw_right` | 0.7795 |
| `nose_bridge` | 0.7795 |
| `left_shoulder` | 0.5986 |
| `right_shoulder` | 0.5890 |
| `left_elbow` | 0.4161 |
| `right_elbow` | 0.3768 |

## Weakest classes by median hand-active output frames

| Class | Median | Minimum | Clips |
| --- | ---: | ---: | ---: |
| ASK | 0.8125 | 0.3438 | 18 |
| COME | 0.8125 | 0.5625 | 18 |
| GIVE | 0.8125 | 0.5625 | 19 |
| HE | 0.8281 | 0.5312 | 18 |
| FIND | 0.8438 | 0.5938 | 18 |
| MAN | 0.8438 | 0.4688 | 19 |
| WHERE | 0.8438 | 0.5312 | 18 |
| BAD | 0.8594 | 0.5312 | 16 |
| ANGRY | 0.8750 | 0.8125 | 17 |
| ANSWER | 0.8750 | 0.3750 | 19 |
| BIG | 0.8750 | 0.4688 | 19 |
| CHILD | 0.8750 | 0.5000 | 21 |
| COLD | 0.8750 | 0.4688 | 19 |
| DAY | 0.8750 | 0.5000 | 19 |
| DIFFERENT | 0.8750 | 0.5938 | 19 |

## Lowest-coverage clips

| Feature | Hand-active frames | Left completeness | Right completeness |
| --- | ---: | ---: | ---: |
| `train/I/0035172227152260316-ME.v17.npz` | 0.1562 | 0.0000 | 1.0000 |
| `train/MAKE/12398889502337007-seedMAKE.v17.npz` | 0.2500 | 1.0000 | 1.0000 |
| `val/HOME/049094629596487804-HOME.v17.npz` | 0.3125 | 1.0000 | 1.0000 |
| `train/ASK/9643840474305965-seedASK.v17.npz` | 0.3438 | 1.0000 | 1.0000 |
| `train/WE/2803881528625629-seedWE.v17.npz` | 0.3438 | 1.0000 | 1.0000 |
| `train/WEEK/5369002649936094-seedWEEK.v17.npz` | 0.3438 | 1.0000 | 1.0000 |
| `train/ANSWER/8669225425641554-seedANSWER.v17.npz` | 0.3750 | 1.0000 | 1.0000 |
| `train/HELP/09953879192572268-seedHELP.v17.npz` | 0.3750 | 1.0000 | 1.0000 |
| `train/I/6421087005073287-ME.v17.npz` | 0.3750 | 1.0000 | 1.0000 |
| `train/SICK/7809273250779392-seedSICK.v17.npz` | 0.3750 | 1.0000 | 1.0000 |
| `val/SAD/04551864995837951-SAD.v17.npz` | 0.3750 | 1.0000 | 1.0000 |
| `train/HAVE/3064521963954243-seedHAVE.v17.npz` | 0.4062 | 1.0000 | 1.0000 |
| `train/STOP/6679488064816475-STOP.v17.npz` | 0.4062 | 1.0000 | 1.0000 |
| `train/HELLO/8841964348021558-HELLO.v17.npz` | 0.4375 | 1.0000 | 1.0000 |
| `train/HUNGRY/3031010038877062-HUNGRY.v17.npz` | 0.4375 | 1.0000 | 1.0000 |
| `train/SAD/19468908305966748-seedSAD.v17.npz` | 0.4375 | 1.0000 | 1.0000 |
| `train/WOMAN/0638032411056142-WOMAN.v17.npz` | 0.4375 | 1.0000 | 1.0000 |
| `train/BIG/02604937104585603-BIG.v17.npz` | 0.4688 | 1.0000 | 1.0000 |
| `train/HAVE/41458666491716034-HAVE.v17.npz` | 0.4688 | 1.0000 | 1.0000 |
| `train/HUNGRY/48783810292836627-HUNGRY.v17.npz` | 0.4688 | 0.0000 | 1.0000 |

Full clip measurements: `artifacts/reports/citizen100_v17_mediapipe_t50_quality.csv`
