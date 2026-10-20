# ATLAS camera-to-output FigJam study board

Board: [ATLAS camera to English](https://www.figma.com/board/QD1x6X7ts34SVBSpvkRhYS)

Verified against submitted-paper text and canonical source on 2026-10-20. Scope is the current iPhone LIVE path, with a separately labeled submitted-paper T5 English path. Shared overview stages were merged at the user's request, using simple names with technical terms in parentheses.

## Contents

- Shared overview, branching only at English generation.
- 01 Capture/extraction: camera scheduling, orientation, Vision cadence, signer assignment, landmarks, crops and cached embeddings.
- 02 Timing/preparation: 450-feature boundary inputs, 64-position window, lookahead, candidates, normalization, interpolation and fixed tensor contracts.
- 03 Landmark recognition: masked motion, selected distances, four regional branches, four global blocks and pooling.
- 04 Appearance/fusion: view mixing, temporal branch, fusion mathematics, fixed batch eight and primary/adapted outputs.
- 05 Segmental decoding: score formula, dynamic programming, commitment, repeats, repairs and optional spelling.
- 06A Paper T5 path; 06B current constrained LiveCaptions path.
- 07 Finish, speech, queues, storage and reset.

46 explanatory cards distinguish paper evidence from code supplements. The JSON companion records content, source references and section IDs; diagramSources retains generated Mermaid, including the first decoder draft, which was subsequently replaced on the board.

## Verification and limits

Read current entry point LiveReelViewController rather than assuming older ReelCamera settings. Inspected selected recognizer checkpoint metadata (epoch4) without running inference. Confirmed current LIVE loads caption models and does not load T5. T5 retained export uses length64 and d_model256. No new latency/accuracy claim or phone benchmark was produced.

Visual QA found large connector bounds after reparenting generated diagrams; rebuilt attached connectors. A floating-point routing experiment was removed after it produced incorrect canvas coordinates. The final board uses native attached connectors. Final screenshots and node-bound checks verify layout. No application/model/data/training-gate changes.

