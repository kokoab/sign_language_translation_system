# Original mobile preview provenance audit — 2026-10-13

The original catalog contains 77 local previews: **50 current train**, **7 validation**, and **20 absent**. Source paths and SHA256 membership agree; all admitted labels agree. This is a membership audit, not a recognition score. No official test video was accessed.

BIG and SIGN were original deep-clean training inputs removed for differing Citizen variants. COME and CHILD previews were never admitted in examined deep-clean metadata; their local classes are also explicitly variant-excluded. ASK was restored in v2. The preview quality pool itself declares `training_eligible=false`.

| Label | Membership | Original source video |
|---|---|---|
| YOU | train | `data/raw_videos/ASL VIDEOS/YOU/YOU_12.mp4` |
| MY | train | `data/raw_videos/ASL VIDEOS/MY/MY_1.mp4` |
| YOUR | train | `data/raw_videos/ASL VIDEOS/YOUR/6759a789.mp4` |
| OUR | train | `data/raw_videos/ASL VIDEOS/OUR/3dc7518e.mp4` |
| WHERE | train | `data/raw_videos/ASL VIDEOS/WHERE/WHERE_19.mp4` |
| WHEN | absent | `data/raw_videos/ASL VIDEOS/WHEN/8773c424.mp4` |
| WHO | train | `data/raw_videos/ASL VIDEOS/WHO/WHO_23.mp4` |
| WHY | train | `data/raw_videos/ASL VIDEOS/WHY/WHY_2.mp4` |
| HELLO | train | `data/raw_videos/ASL VIDEOS/HELLO/3f54b470.mp4` |
| PLEASE | train | `data/raw_videos/ASL VIDEOS/PLEASE/83e982ca.mp4` |
| THANKYOU | train | `data/raw_videos/ASL VIDEOS/THANKYOU/a9c96653.mp4` |
| SORRY | train | `data/raw_videos/ASL VIDEOS/SORRY/c01ace66.mp4` |
| YES | absent | `data/raw_videos/ASL VIDEOS/YES/YES_2.mp4` |
| NO | absent | `data/raw_videos/ASL VIDEOS/NO/NO_7.mp4` |
| MAYBE | train | `data/raw_videos/ASL VIDEOS/MAYBE/30135481.mp4` |
| HELP | train | `data/raw_videos/ASL VIDEOS/HELP/276970e3.mp4` |
| NEED | absent | `data/raw_videos/ASL VIDEOS/NEED/NEED_1.mp4` |
| LIKE | train | `data/raw_videos/ASL VIDEOS/LIKE/LIKE_3.mp4` |
| LOVE | train | `data/raw_videos/ASL VIDEOS/LOVE/LOVE_8.mp4` |
| KNOW | absent | `data/raw_videos/ASL VIDEOS/KNOW/dc9dc051.mp4` |
| UNDERSTAND | train | `data/raw_videos/ASL VIDEOS/UNDERSTAND/UNDERSTAND_3.mp4` |
| THINK | train | `data/raw_videos/ASL VIDEOS/THINK/THINK_1.mp4` |
| FEEL | train | `data/raw_videos/ASL VIDEOS/FEEL/FEEL_1.mp4` |
| GIVE | train | `data/raw_videos/ASL VIDEOS/GIVE/GIVE_10.mp4` |
| TAKE | absent | `data/raw_videos/ASL VIDEOS/TAKE/TAKE_1.mp4` |
| COME | absent | `data/raw_videos/ASL VIDEOS/COME/COME_21.mp4` |
| GO | train | `data/raw_videos/ASL VIDEOS/GO/GO_16.mp4` |
| STOP | train | `data/raw_videos/ASL VIDEOS/STOP/7c56ae76.mp4` |
| WAIT | train | `data/raw_videos/ASL VIDEOS/WAIT/WAIT_25.mp4` |
| TRY | absent | `data/raw_videos/ASL VIDEOS/TRY/99aa59c2.mp4` |
| USE | absent | `data/raw_videos/ASL VIDEOS/USE/fc41ed03.mp4` |
| MAKE | train | `data/raw_videos/ASL VIDEOS/MAKE/e880ee91.mp4` |
| SEE | train | `data/raw_videos/ASL VIDEOS/SEE/SEE_1.mp4` |
| TELL | absent | `data/raw_videos/ASL VIDEOS/TELL/TELL_3.mp4` |
| ASK | train | `data/raw_videos/ASL VIDEOS/ASK/ASK_7.mp4` |
| ANSWER | train | `data/raw_videos/ASL VIDEOS/ANSWER/ANSWER_12.mp4` |
| LEARN | train | `data/raw_videos/ASL VIDEOS/LEARN/e6634440.mp4` |
| WORK | validation | `data/raw_videos/ASL VIDEOS/WORK/59d6f5f4.mp4` |
| READ | train | `data/raw_videos/ASL VIDEOS/READ/5f3588d6.mp4` |
| WRITE | train | `data/raw_videos/ASL VIDEOS/WRITE/dc3b4e86.mp4` |
| WATER | absent | `data/raw_videos/ASL VIDEOS/WATER/WATER_17.mp4` |
| SLEEP | validation | `data/raw_videos/ASL VIDEOS/SLEEP/SLEEP_1.mp4` |
| GOOD | train | `data/raw_videos/ASL VIDEOS/GOOD/GOOD_14.mp4` |
| BAD | validation | `data/raw_videos/ASL VIDEOS/BAD/BAD_3.mp4` |
| HAPPY | train | `data/raw_videos/ASL VIDEOS/HAPPY/HAPPY_15.mp4` |
| SAD | train | `data/raw_videos/ASL VIDEOS/SAD/SAD_22.mp4` |
| ANGRY | absent | `data/raw_videos/ASL VIDEOS/ANGRY/ANGRY_10.mp4` |
| EXCITED | train | `data/raw_videos/ASL VIDEOS/EXCITED/EXCITED_25.mp4` |
| TIRED | absent | `data/raw_videos/ASL VIDEOS/TIRED/TIRED_24.mp4` |
| HOT | train | `data/raw_videos/ASL VIDEOS/HOT/HOT_1.mp4` |
| COLD | train | `data/raw_videos/ASL VIDEOS/COLD/COLD_5.mp4` |
| BIG | absent | `data/raw_videos/ASL VIDEOS/BIG/ecb0b1fa.mp4` |
| SMALL | train | `data/raw_videos/ASL VIDEOS/SMALL/SMALL_25.mp4` |
| MORE | train | `data/raw_videos/ASL VIDEOS/MORE/2e3262ee.mp4` |
| LESS | train | `data/raw_videos/ASL VIDEOS/LESS/6db34606.mp4` |
| DIFFERENT | train | `data/raw_videos/ASL VIDEOS/DIFFERENT/DIFFERENT_14.mp4` |
| EASY | train | `data/raw_videos/ASL VIDEOS/EASY/6d9163ca.mp4` |
| IMPORTANT | train | `data/raw_videos/ASL VIDEOS/IMPORTANT/IMPORTANT_2.mp4` |
| READY | absent | `data/raw_videos/ASL VIDEOS/READY/READY_6.mp4` |
| NOW | absent | `data/raw_videos/ASL VIDEOS/NOW/451467eb.mp4` |
| TOMORROW | absent | `data/raw_videos/ASL VIDEOS/TOMORROW/TOMORROW_13.mp4` |
| YESTERDAY | validation | `data/raw_videos/ASL VIDEOS/YESTERDAY/YESTERDAY_15.mp4` |
| MORNING | validation | `data/raw_videos/ASL VIDEOS/MORNING/17d784c5.mp4` |
| TIME | train | `data/raw_videos/ASL VIDEOS/TIME/95ab1c0c.mp4` |
| DAY | train | `data/raw_videos/ASL VIDEOS/DAY/abe26bfb.mp4` |
| WEEK | train | `data/raw_videos/ASL VIDEOS/WEEK/WEEK_16.mp4` |
| YEAR | absent | `data/raw_videos/ASL VIDEOS/YEAR/41719d22.mp4` |
| SCHOOL | train | `data/raw_videos/ASL VIDEOS/SCHOOL/74c7254b.mp4` |
| FAMILY | train | `data/raw_videos/ASL VIDEOS/FAMILY/cfc5ba04.mp4` |
| FRIEND | train | `data/raw_videos/ASL VIDEOS/FRIEND/26b51918.mp4` |
| MOTHER | train | `data/raw_videos/ASL VIDEOS/MOTHER/b0644df4.mp4` |
| FATHER | validation | `data/raw_videos/ASL VIDEOS/FATHER/15c4318a.mp4` |
| CHILD | absent | `data/raw_videos/ASL VIDEOS/CHILD/0cf98949.mp4` |
| MAN | train | `data/raw_videos/ASL VIDEOS/MAN/9bd0f908.mp4` |
| NAME | validation | `data/raw_videos/ASL VIDEOS/NAME/315d535c.mp4` |
| SIGN | absent | `data/raw_videos/ASL VIDEOS/SIGN/9dc6d501.mp4` |
| LANGUAGE | train | `data/raw_videos/ASL VIDEOS/LANGUAGE/cd2b6376.mp4` |

Other absent previews have no established exclusion reason in the examined pre/post deep-clean manifests, quarantine or extraction-rejection metadata. Absence alone does not prove an incorrect sign.

Evidence:

- `data/local/web_variant_supplement_v17_20261008/local_train_citizen_variant_v2.json`
- `data/local/web_variant_supplement_v17_20261008/local_val_citizen_variant_v2.json`
- `ATLAS/mobile_app/slt_mobile_app/assets/gloss_examples/catalog.json`
- `ATLAS/mobile_app/slt_mobile_app/ios/Runner/LiveReel/gloss_examples/catalog.json`
- `data/local/local_deep_clean_v17/train_final_manifest.json`
- `data/local/local_deep_clean_v17/val_final_manifest.json`
- `data/local/local_citizen100_quality_audit_q82_cap14_exact/candidate_selection.json`
- `artifacts/reports/citizen_variant_local_filter_v17_20261008/build_manifests.py`
- `artifacts/reports/citizen_variant_local_filter_v17_20261008/run_experiment_v2.py`

Rerun: `venv/bin/python artifacts/reports/mobile_preview_audit_v17_20261013/audit_provenance.py`
