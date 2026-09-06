# V17 rigged-avatar review prototype

This prototype retargets the existing grounded `HELLO HOW YOU` phrase onto a
triangulated human avatar with a fixed metric skeleton. It exercises a one-hand to
two-hand transition and a two-hand to one-hand transition.

Review [the MP4](hello_how_you.rigged_avatar.mp4), the
[six-frame contact sheet](hello_how_you.contact_sheet.png), and the
[machine-readable report](hello_how_you.report.json).

## Measured invariants

- The source's maximum 2D hand-tree scale ratio is 2.46735x.
- Maximum avatar upper-arm length error is 0.0000404%.
- Maximum avatar forearm length error is 0.0000337%.
- Maximum avatar hand-bone length error is 0.0003104%.
- The 160 rendered hand/frame states comprise 111 observed, 7 imputed-active,
  and 42 intentional-rest states.

The resting hand is an explicit relaxed pose beside the hip with low-amplitude sway.
It is not copied from a detector placeholder near the stomach. State changes ease over
seven frames so a hand does not snap between rest and signing.

## Repeat

```bash
venv/bin/python scripts/render_rigged_avatar_v17.py \
  artifacts/reports/stage2_v17_grounded_text_input_demo_v2/hello_how_you.grounded_text_to_sign_v17.npz
venv/bin/python -m unittest test.test_avatar_rig_v17 -v
```

## Contract and limits

The v17 third spatial channel is a log-scale proxy. The renderer uses it only as a
bounded front/back ordering cue and does not call it metric depth or 3D reconstruction.
The mesh and software renderer are original project code; no third-party avatar asset
is bundled. This is a shaded, rigged human review proxy, not a photoreal person. Facial
grammar is not yet retargeted. ASL-fluent review is pending, so these outputs remain
`synthetic_native_review_only` and are not eligible as training, validation, or test
data.
