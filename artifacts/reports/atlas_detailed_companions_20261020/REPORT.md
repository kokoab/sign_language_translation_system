# ATLAS detailed FigJam companions — 2026-10-20

Added eight detailed companions (25 subdiagrams) to the [existing board](https://www.figma.com/board/QD1x6X7ts34SVBSpvkRhYS), beside the matching overview panels. The original eight sections are unchanged, including their positions, text, nested cards and connectors; exact before/after section XML comparisons passed for all eight.

## Navigation

- [R: R · Detailed runtime](https://www.figma.com/board/QD1x6X7ts34SVBSpvkRhYS?node-id=48-1195) — 4 subdiagrams
- [H: H · Detailed HELLO example](https://www.figma.com/board/QD1x6X7ts34SVBSpvkRhYS?node-id=48-1204) — 3 subdiagrams
- [I: I · Detailed isolated training](https://www.figma.com/board/QD1x6X7ts34SVBSpvkRhYS?node-id=48-1213) — 3 subdiagrams
- [P: P · Detailed phrase adaptation](https://www.figma.com/board/QD1x6X7ts34SVBSpvkRhYS?node-id=48-1222) — 3 subdiagrams
- [C: C · Detailed interval adaptation](https://www.figma.com/board/QD1x6X7ts34SVBSpvkRhYS?node-id=48-1231) — 3 subdiagrams
- [B: B · Detailed boundary training](https://www.figma.com/board/QD1x6X7ts34SVBSpvkRhYS?node-id=48-1240) — 3 subdiagrams
- [E: E · Detailed paper T5 training](https://www.figma.com/board/QD1x6X7ts34SVBSpvkRhYS?node-id=48-1249) — 3 subdiagrams
- [D: D · Detailed deployment](https://www.figma.com/board/QD1x6X7ts34SVBSpvkRhYS?node-id=48-1258) — 3 subdiagrams

## Content

Each companion traces input origin, calculations/operations, shapes, outputs and destinations. Runtime detail separates extraction, landmark processing, image-feature fusion, timing/search and paper English. Training detail separates preparation, objective, gradient/update and selection. HELLO remains a separate illustrative example.

The source supplements distinguish stage-specific settings from paper-reported results. Phrase adaptation freezes encoders; the selected primary interval run trains encoders at a smaller learning rate; a later guarded refinement remains a separate phase. Frozen components still compute features. Teacher targets supervise timing rather than ASL labels. The T5 recipe is matched to the paper's reported receipt; its final-step retention differs from recognizer selection.

Toy interpolation/confidence examples are marked illustrative. Actual bundled token IDs are conditional on the stated example input, without claiming a measured generated greeting. No generalization claim is inferred from conversion comparisons.

## Verification

- All eight original overview subtrees preserved exactly.
- New section text scanned for excluded subsystem/alternate-path terms: no matches.
- All 25 detailed subdiagrams have in-bounds children; no companion section collisions.
- Screenshots inspected for runtime extraction, landmark processing, fusion, decoding, HELLO, interval loss, boundary features, T5 training, isolated/phrase training and deployment.
- Fixed a clipped boundary-feature label; full feature breakdown remains in its explanatory text.
- Source-backed notes and stable node IDs are saved in board_content.json.
- No source/model edit, training, export, mobile build, benchmark or protected evaluation was run.

Next safe action: open the runtime companion, zoom into one subdiagram at a time, and follow its referenced overview step IDs.
