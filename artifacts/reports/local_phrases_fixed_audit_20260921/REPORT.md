# Corrected local phrase audit

User confirms every retained clip was checked against its folder phrase.

685 videos are unchanged original content;95 originals were removed, none relabeled or edited.

Retained cached local clips: 232 train / 200 validation. Removed from this supervised subset: 55.

Features and target arrays are bitwise identical for retained clips; source paths and review provenance are updated. Signer roles are preserved and disjoint. ASLLRP and NCSLGR archives are copied unchanged.

Fixed videos outside the admitted cache: 253; see `cache_verification.json` for per-phrase counts. These include phrases outside the existing locked-label recipe and prior cache-admission exclusions; no new labels or unverified aliases are added.
