# O5S5 exact Citizen-100 supervision

The six public, frontal O5S5 narratives provide 21.73 minutes from six signers. Exact ASL-LEX-to-Signbank matching admits 256 deduplicated occurrences across 53 frozen Citizen classes; all 256 contain Apple Vision hand detections. These are genuine Apple Vision observations extracted from source pixels, not converted MediaPipe coordinates.

`LG` is reserved as a whole validation signer. O5S5 adds 1005 positive training windows across 49 classes and 318 validation windows across 27 classes. Combined with the existing ASLLRP supervision, training has 6819 contextual windows across 73/100 classes plus 494 verified background windows. O5S5 itself contributes zero background windows because ID-gloss completeness is not established.

The source is public under CC BY-NC-SA 4.0; no account or access request is required.

- `exact_occurrences.csv`: authoritative exact positive-event manifest
- `apple_vision_supervision.json`: O5S5 positive-only Apple Vision supervision
- `combined_supervision.json`: ready-to-load ASLLRP + O5S5 supervision
- `audit.json`, `apple_vision_audit.json`, `loader_audit.json`: provenance, coverage, and loader checks
- `sources.json`: source video/EAF hashes and deduplicated ID-gloss intervals
