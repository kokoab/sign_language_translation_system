# Boundary-aware fixed-window experiment

The model keeps exactly 100 visible glosses and learns a separate internal endpoint phase: KNOWN, UNKNOWN, or TRANSITION. Training and replay use the same 32-frame 0.27s/0.53s normalization. Incomplete O5S5 rows supplied known sign cores only; negative phases came only from fully annotated ASLLRP.

## Results

| Measure | Result |
| --- | ---: |
| Validation phase accuracy | 50.23% |
| Validation known-core gloss accuracy | 55.53% (708/1275) |
| Citizen isolated retention | 93.39% (353/378) |
| SemLex isolated retention | 84.76% (829/978) |
| Complete-ASLLRP online WER | 106.82% |
| Empty online transcripts | 55/237 |

Every admitted training sample was used exactly once per epoch. No replacement sampler, 1.07-second target, CTC loss, or incomplete-row transition label was used. Full confusion counts, transcripts, and edit operations are in `evaluation.json`. Citizen test remained sealed and no runtime was promoted automatically.

## Decision

Reject this checkpoint for runtime use. Phase accuracy is only 50.23%, and online WER is 106.82% with 86 deletions, 110 substitutions, and 133 insertions over 308 reference glosses. Isolated retention also fell from the starting checkpoint's 95.24%/85.28% Citizen/SemLex accuracy to 93.39%/84.76%.

Correction after direct data inspection: the prior claim that clean supervision and matched windows removed the known pipeline mismatches was too strong. Matching tensor shapes and normalization did not validate target visibility, source crop completeness, physical transition semantics, or train/inference sampling. All known context inputs end before their sign annotations end; two FEEL windows contain no raw target observations, and crop-completeness flags were lost downstream. These results reject this checkpoint but do not isolate an architectural failure. See [data audit and annotated video examples](../boundary_data_audit_20260920/REPORT.md). Original checkpoint and metrics are unchanged.
