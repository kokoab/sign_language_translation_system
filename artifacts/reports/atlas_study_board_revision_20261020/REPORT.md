# ATLAS study-board revision — 2026-10-20

[Open the general pipeline](https://www.figma.com/board/QD1x6X7ts34SVBSpvkRhYS?node-id=19-633).

The user requested a simpler board with explicit input provenance, a separate HELLO example, current continuous-signing implementation only and submitted-paper T5 English only. The replacement removes the prior English-implementation comparison and scoped-out subsystems from the board.

## Eight separate diagrams

- R: general runtime, seven steps.
- H: illustrative HELLO walkthrough, mapped one-to-one to R.
- I: isolated-sign recognition training and isolated multimodal lineage.
- P: frozen-encoder phrase-segment head adaptation.
- C: current candidate-interval adaptation, primary retention and scoped refinement.
- B: teacher/student boundary training.
- E: paper T5 training.
- D: checkpoint export and mobile-runtime connection.

Each process has its own step prefix. 54 explanatory cards identify input origins, processing, outputs and downstream consumers. Reading the main flow does not require tracing long cross-panel arrows. General and HELLO sections are retained separately.

## Evidence and distinctions

Paper text: methods/prototype pp.75–82; base selection pp.93–94; English validation loss0.2515 at optimizer step4,539. Code-only configurations are labeled as supplements.

Verified phrase phase freezes encoders, selected primary interval phase does not, and later scoped transition refinement freezes encoders. Interval trainer uses classification, tiny-fragment flat targets, replay retention and anchoring, with stage-specific masks. The timing student uses aligned teacher O/B/I distributions and soft-target cross-entropy; held-out KL selects its checkpoint.

Matched T5 receipt: stage3_multisentence_bakeoff_v17_20260929/tiny/recipe.json records Adafactor, batch32, peak LR0.0003,100-step warm-up and linear decay; targeted training_result history confirms step4,539/loss0.2515. The final step was kept; validation loss was reported, not used to choose a best epoch. This differs from recognition checkpoint selection.

## Validation

Rendered all eight sections and inspected flow, text fit and note grouping. Structural check reports no child overflow; content scan found no excluded terms in the board sections. The HELLO path is explicitly illustrative and has no fabricated model confidence, duration or measured English output.

Companion board_content.json preserves content, generated Mermaid, node IDs and coordinates for session continuity. Previous artifact remains historical; this revision supersedes its board description. No app/model/training-gate changes or new evaluation.

