# Genuine local phrase landmark reference

This review contains one complete, genuine train-role performance from each of the
nine local phrase families. No signs were concatenated and no transition or motion was
generated.

- [Watch the side-by-side MP4](genuine_local_phrase_reference.mp4)
- [Open the nine-phrase contact sheet](contact_sheet.png)
- [Inspect the machine-readable report](report.json)

The left panel shows the detector observations exactly as extracted. The right panel
uses the same real coordinates and motion, filling missing anatomy only for drawing.
It never adds a hand that does not participate in the performance and bridges only
short detector gaps for a participating hand.
The raw `.npz` files keep `animation_rig_xyz` separate from
`observation_xyz`, `observation_presence`, and `observation_confidence`; they contain no
ambiguous `landmarks` training tensor.

Observed-node presence ranges from 49.8% to 77.5% across these selected references. That
sparsity is detector behavior, not missing ground truth to be replaced with all-one
presence. The completed rig is reference/render material only and is explicitly
ineligible for recognition training, validation, or testing.

This establishes the safe v3 rendering baseline. It does not validate the rejected
isolated-sign composition approach, generate a new phrase combination, or establish
native linguistic naturalness.
