# Data sufficiency audit — 2026-09-13

Better public data has now been acquired, exact-mapped, and extracted with Apple Vision
without accounts or access requests. The next bounded development experiment is data-ready;
full 100-class continuous coverage and phone-generalization readiness remain open.

| Required evidence | Verified state |
|---|---|
| Raw continuous ASL with timed hand labels | O5S5: six paired narratives/EAFs, 3,959 hand events; RWTH: 201 sentence videos, 888 tokens |
| Exact visual variants of all 100 classes | O5S5 exact Signbank-ID join admits 256 occurrences across 53 classes; combined ASLLRP+O5S5 training covers 73/100 contextual classes |
| Diverse signer-disjoint continuous training and selection | O5S5 supplies five training signers and one whole validation signer (LG); RWTH remains training-only auxiliary data |
| Verified transitions, repetitions, holds and OOV/background | O5S5 supplies natural transitions and masks other annotated signs; only existing verified ASLLRP gaps supply background |
| Independent portrait-iPhone selection recordings | Not acquired; studio/Zoom recordings do not establish this |
| Full, correct temporal targets | All 256 exact O5S5 targets fit source timing and contain Apple Vision hands; O5S5 gaps remain excluded because annotation completeness is not proven |

## MediaPipe to Apple Vision

Approximate joint mapping is possible for a separate experiment. It cannot recover
Apple Vision observations: Épée stores image-normalized XYZ, whereas our contract
uses body-relative coordinates, a scale-based depth proxy, presence and detector
confidence. Detector failures/confidence cannot be reconstructed from XYZ; face
landmark definitions also differ. A learned adapter needs paired raw-video outputs
and independent validation. No adapter or forged Apple fingerprint was created.

## New source checks

- Better no-account acquisition is complete at
  `artifacts/reports/open_asl_alternatives_20260913/README.md`. O5S5 contributes six
  frontal 540p-to-4K narratives, 21.73 minutes, six signers, 2,601 right-hand and
  1,358 left-hand tier events under CC BY-NC-SA 4.0. The exact Signbank-ID manifest
  admits 256 deduplicated positives across 53 classes. Real Apple Vision extraction
  produced 26,079 observations and detected hands in 256/256 targets. Ready combined
  supervision is in `artifacts/reports/o5s5_citizen100_v17/combined_supervision.json`.
  RWTH-BOSTON-104 remains low-resolution auxiliary sequence data.
- SoMe ASL is excluded from training after user visual review found it predominantly
  one-handed and unsuitable. Preserve the downloaded files; do not re-admit them.

- Official MoLo provider lists 26 MP4s. Matching all 16 public EAFs yields the
  acquired systems video and a 2.14 GB interview. Interview tier inspection found
  only one hand gloss across four EAFs, despite extensive English translations.
  Stopped its transfer; partial file retained, not counted as acquired. Evidence:
  `molo_current_video_listing.json`, `molo_interview_annotation_audit.json`.
- How2Sign official download page provides RGB plus English translations, without
  a timed-gloss download. Frontal training clips alone are 31 GB; free disk was
  9.7 GiB. No bulk download. https://how2sign.github.io/
- NCSLGR expanded XML remains behind the official DAI login per prior audit;
  existing public subset is already local. No access bypass attempted.
- Full RIT Homework requires institutional Databrary authorization. CUNY needs a
  publisher request. No outgoing contact has been authorized.
- ASLLRP documentation excludes DawnSignPress data from downloadable material;
  advertised additional signers cannot all be counted as available training data.
  https://www.bu.edu/asllrp/about-datasets.pdf
- Daily Moth Figshare EAF downloaded. It marks 32 first-person-reference instances,
  not all signed words. Raw sample downloaded (75,053,731 bytes), publisher MD5 and local SHA256 recorded; it does not fill full
  gloss supervision. Evidence: `daily_moth_source.json` and downloaded EAF.

No training or protected-test access occurred. The next bounded training attempt has
enough new exact two-handed contextual data to run. Full continuous coverage for the
remaining 27 classes and independent portrait-iPhone evidence remain broader gates.

## Access requests prepared, not sent

`ACCESS_REQUESTS.md` contains three concrete inquiries for the CUNY/RIT corpora,
How2Sign human ELAN annotations and Épée raw video. `access_request_vocabulary.csv`
preserves all 100 exact raw-gloss/ASL-LEX pairs from the frozen manifest. The
How2Sign CVPR supplement describes human ELAN collection, so availability needs
publisher confirmation rather than assuming the data were never annotated:
https://openaccess.thecvf.com/content/CVPR2021/supplemental/Duarte_How2Sign_A_Large-Scale_CVPR_2021_supplemental.pdf

Sending is unnecessary for the next bounded experiment. Those requests remain unsent
options only if broader 100-class coverage is pursued later.
