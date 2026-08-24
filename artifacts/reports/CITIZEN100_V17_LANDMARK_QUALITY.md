# Citizen100 v17 Landmark Availability Audit

- Feature root: `data/local/citizen100_v17/landmarks`
- Archives measured: 3101
- Clips below 25% hand-active output frames: 0
- Values below are min / p10 / median / p90 / max.
- This audit measures missingness and confidence. Visual overlay review is
  still required to judge geometric placement or semantic correctness.

## Clip-level metrics

| Metric | Min / p10 / median / p90 / max |
| --- | --- |
| `hand_active_output_frames` | 0.2812 / 0.8125 / 0.8750 / 0.9375 / 1.0000 |
| `left_active_output_frames` | 0.0000 / 0.0000 / 0.7188 / 0.8750 / 1.0000 |
| `right_active_output_frames` | 0.0000 / 0.0000 / 0.8750 / 0.8750 / 1.0000 |
| `both_hands_active_output_frames` | 0.0000 / 0.0000 / 0.0000 / 0.8125 / 0.9375 |
| `left_joint_completeness_when_active` | 0.0000 / 0.0000 / 0.9167 / 0.9872 / 1.0000 |
| `right_joint_completeness_when_active` | 0.0000 / 0.0000 / 0.9559 / 0.9932 / 1.0000 |
| `observed_point_confidence_mean` | 0.3904 / 0.5164 / 0.5889 / 0.6521 / 0.7458 |
| `detected_hand_source_frames_before_trim` | 0.0417 / 0.3261 / 0.4667 / 0.6207 / 1.0000 |
| `detected_hand_source_frames_after_trim` | 0.2976 / 0.7826 / 0.8750 / 0.9130 / 1.0000 |

## Per-node presence

Presence counts include explicit bounded interpolation, but never Kalman
extrapolation. A missing point remains zero and contributes no motion feature.

| Node | Presence fraction |
| --- | ---: |
| `left_hand_wrist` | 0.4561 |
| `left_hand_thumb_cmc` | 0.4623 |
| `left_hand_thumb_mp` | 0.4674 |
| `left_hand_thumb_ip` | 0.4643 |
| `left_hand_thumb_tip` | 0.4556 |
| `left_hand_index_mcp` | 0.4750 |
| `left_hand_index_pip` | 0.4741 |
| `left_hand_index_dip` | 0.4695 |
| `left_hand_index_tip` | 0.4653 |
| `left_hand_middle_mcp` | 0.4733 |
| `left_hand_middle_pip` | 0.4701 |
| `left_hand_middle_dip` | 0.4660 |
| `left_hand_middle_tip` | 0.4653 |
| `left_hand_ring_mcp` | 0.4633 |
| `left_hand_ring_pip` | 0.4592 |
| `left_hand_ring_dip` | 0.4639 |
| `left_hand_ring_tip` | 0.4566 |
| `left_hand_little_mcp` | 0.4473 |
| `left_hand_little_pip` | 0.4519 |
| `left_hand_little_dip` | 0.4526 |
| `left_hand_little_tip` | 0.4485 |
| `right_hand_wrist` | 0.6694 |
| `right_hand_thumb_cmc` | 0.6773 |
| `right_hand_thumb_mp` | 0.6848 |
| `right_hand_thumb_ip` | 0.6802 |
| `right_hand_thumb_tip` | 0.6699 |
| `right_hand_index_mcp` | 0.6933 |
| `right_hand_index_pip` | 0.6932 |
| `right_hand_index_dip` | 0.6884 |
| `right_hand_index_tip` | 0.6845 |
| `right_hand_middle_mcp` | 0.6930 |
| `right_hand_middle_pip` | 0.6909 |
| `right_hand_middle_dip` | 0.6880 |
| `right_hand_middle_tip` | 0.6843 |
| `right_hand_ring_mcp` | 0.6831 |
| `right_hand_ring_pip` | 0.6800 |
| `right_hand_ring_dip` | 0.6849 |
| `right_hand_ring_tip` | 0.6751 |
| `right_hand_little_mcp` | 0.6639 |
| `right_hand_little_pip` | 0.6700 |
| `right_hand_little_dip` | 0.6695 |
| `right_hand_little_tip` | 0.6630 |
| `left_pupil` | 0.7892 |
| `right_pupil` | 0.7892 |
| `left_brow_start` | 0.7892 |
| `left_brow_end` | 0.7892 |
| `right_brow_start` | 0.7892 |
| `right_brow_end` | 0.7892 |
| `nose_tip` | 0.7892 |
| `mouth_left` | 0.7892 |
| `mouth_right` | 0.7892 |
| `upper_lip` | 0.7892 |
| `lower_lip` | 0.7892 |
| `jaw_left` | 0.7892 |
| `chin` | 0.7892 |
| `jaw_right` | 0.7892 |
| `nose_bridge` | 0.7892 |
| `left_shoulder` | 0.5805 |
| `right_shoulder` | 0.5740 |
| `left_elbow` | 0.3996 |
| `right_elbow` | 0.3570 |

## Weakest classes by median hand-active output frames

| Class | Median | Minimum | Clips |
| --- | ---: | ---: | ---: |
| ANGRY | 0.8750 | 0.8125 | 32 |
| ANSWER | 0.8750 | 0.7812 | 31 |
| ASK | 0.8750 | 0.4375 | 31 |
| BAD | 0.8750 | 0.4688 | 30 |
| BIG | 0.8750 | 0.5625 | 31 |
| CHILD | 0.8750 | 0.5625 | 31 |
| COLD | 0.8750 | 0.4062 | 31 |
| COME | 0.8750 | 0.6250 | 30 |
| DAY | 0.8750 | 0.6250 | 29 |
| DIFFERENT | 0.8750 | 0.5000 | 31 |
| DOCTOR | 0.8750 | 0.6875 | 30 |
| DRINK | 0.8750 | 0.7500 | 31 |
| EASY | 0.8750 | 0.7812 | 31 |
| EAT | 0.8750 | 0.5000 | 39 |
| EXCITED | 0.8750 | 0.7812 | 30 |

## Lowest-coverage clips

| Feature | Hand-active frames | Left completeness | Right completeness |
| --- | ---: | ---: | ---: |
| `val/SLEEP/5103688507862052-SLEEP.v17.npz` | 0.2812 | 0.0000 | 0.9206 |
| `val/LISTEN/1493429103533752-LISTEN.v17.npz` | 0.3125 | 0.0000 | 0.9905 |
| `val/HOME/049094629596487804-HOME.v17.npz` | 0.3750 | 0.0000 | 0.8889 |
| `test/HUNGRY/5801976577083523-HUNGRY.v17.npz` | 0.4062 | 0.0000 | 0.9121 |
| `train/KNOW/6589842343221302-seedKNOW.v17.npz` | 0.4062 | 0.5714 | 0.9683 |
| `train/TIME/9956589338184496-seedTIME.v17.npz` | 0.4062 | 0.5794 | 0.9683 |
| `train/TRY/21482100381740965-seedTRY.v17.npz` | 0.4062 | 0.9304 | 0.9365 |
| `val/COLD/39072948576699607-COLD.v17.npz` | 0.4062 | 0.9881 | 0.8828 |
| `val/TALK/873026373390853-TALK.v17.npz` | 0.4062 | 0.0000 | 0.9267 |
| `train/HELLO/8841964348021558-HELLO.v17.npz` | 0.4375 | 1.0000 | 1.0000 |
| `train/SLEEP/24375109516725524-SLEEP.v17.npz` | 0.4375 | 0.0000 | 0.9252 |
| `val/ASK/0773952609683628-ASK.v17.npz` | 0.4375 | 0.3810 | 0.9451 |
| `val/SAD/04551864995837951-SAD.v17.npz` | 0.4375 | 0.9206 | 0.8791 |
| `val/WATER/03174577769237463-WATER.v17.npz` | 0.4375 | 0.3810 | 0.9158 |
| `train/BAD/6558607659620332-BAD.v17.npz` | 0.4688 | 0.0000 | 0.9365 |
| `val/YOUR/7464906167030079-YOUR.v17.npz` | 0.4688 | 0.0000 | 0.8603 |
| `train/DIFFERENT/5558132949950849-DIFFERENT.v17.npz` | 0.5000 | 0.9873 | 0.9345 |
| `train/WHY/009620790877519658-seedWHY.v17.npz` | 0.5000 | 0.0000 | 0.9315 |
| `train/YOU/9739021665350334-YOU.v17.npz` | 0.5000 | 0.0000 | 0.8839 |
| `val/EAT/009883456113471967-EAT.v17.npz` | 0.5000 | 0.0000 | 0.9315 |

Full clip measurements: `artifacts/reports/citizen100_v17_landmark_quality.csv`
