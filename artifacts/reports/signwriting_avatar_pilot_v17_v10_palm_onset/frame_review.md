# Blue-avatar frame review

Reviewed every frame of the v7, v8, v9 and v10 YOU/NEED avatar videos, alongside the source footage. Numbered contact sheets and frame_motion.csv are saved with the renders.

| Observed defect | Evidence | Correction |
| --- | --- | --- |
| Hand rises toward face, then drops before signing | v7 YOU12–18; NEED11–15 | Missing wrist coordinates are interpolated before smoothing instead of mapped from zero |
| Extra upward movement after signing | v7 YOU37–43 rises17.6cm; NEED39–42 ends22.3cm above frame36 | Same missing-wrist fix; generated clips use a single release |
| YOU rolls/redirects its hand around the sign | v8 YOU21–23 and37–43 | v10 uses existing notation-driven core instead of copying detector orientation noise |
| NEED palm spreads/twists during preparation | v9 NEED4–8; rotated-palm regression reproduced6.2cm excess separation | Interpolate finger flexion in palm-local coordinates and rotate the palm as one unit |

V10 reviewed all52YOU and53NEED frames. One approach, encoded movement, one release. Both inactive hands remain stationary. YOUcore travels6cm forward with no extra orientation change. NEEDcore keeps wrist position fixed and contains two flex strokes with one intervening return. Core pose arrays are bit-identical to v6_calibrated; the S100/S106 handshape rules were not changed. All25rig/mesh tests pass.

Opening and release are inferred animation, not captured source motion. Comparison timing is retimed to class-median duration. Maximum per-frame wrist displacement during the release is10.6cm for YOU and9.7cm for NEED; these are measured values, not evidence of natural timing. Human assessment of the complete motion remains pending. No continuous-join quality or training eligibility is claimed.
