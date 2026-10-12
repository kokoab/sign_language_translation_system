# Apple / MediaPipe interval-adaptation diagnostic — 2026-10-13

Fresh inference on each extractor's cached Citizen validation inputs; 378 clips each.
Historical before/after checkpoint pair from the October4 comparison, not the newer
96.83% chain. No training, protected-test access, model selection or deployment change.

| Pipeline | Before correct | After correct | Lost | Gained | Changed predictions |
|---|---:|---:|---:|---:|---:|
| Apple Vision |363|360|4|1|5|
| MediaPipe |357|357|3|3|6|

MediaPipe's unchanged accuracy hides offsetting errors and corrections. Both changed
all259landmark tensors,108hand tensors,14fusion-head tensors. Apple errors newly include
WHO→YOU,TALK→WATER,and twoHAPPY→PLEASE; HEAR was corrected. MediaPipe newly confuses
NEED→HE,KNOW→HELLO,HAPPY→PLEASE; correctsTHANKYOU,TALK,ANSWER. See result JSONs for IDs.

Checkpoint swaps (not separately trained ablations): Apple updated-head-only361,
updated-encoders-only361 vs363parent; both360. MediaPipe updated-head-only356,
updated-encoders-only357 vs357parent; both357. The MediaPipe encoder-only swap reproduces
all three newly wrong examples. Changes in learned representations and classification
both matter; not a conversion artifact. Narrow phrase supervision is consistent with
some confusions toward phrase glosses, but is not isolated as the sole cause.

Selection rule: minimum tuning WER subject to isolated floors, strict improvement only.
Apple selectedepoch4:95.24%Citizen,27.4336%historicalWER. Epoch6 has95.77%Citizen and
identicalWER, local96.4088%local vs96.3053%;SemLex89.5706% vs89.6728%.
Strict '<' keeps epoch4: no isolated-accuracy tie-break. This explains part of the
reported Apple loss. Epoch6 is a historical observation, not a newly promoted checkpoint.
MediaPipe epoch6 Citizen94.71% had worseWER24.78 vs24.34, so was not selected.
These historical WERs are diagnostic evidence, not replacements for the paper's five-stage story.

MediaPipe local isolated accuracy also falls:2824→2815 correct (full paired denominator2896).
Its unchangedCitizen94.44% should never imply adaptation is harmless across domains.
Apple/MP selector alpha differs(.7/1.0); this is not a pure extractor-only causal trial.

Screenshot component update: first re-evaluate existing standalone landmark, standalone
hand, and fused checkpoints on one pinned validation membership. No retraining is
inherently needed. Embedded branches after joint adaptation are not automatically
independently trained standalone baselines. Provisional10–30min for cached evaluation;
new full matched training is separate scope. Historical cached hand training6.03min,
local hand adaptation10.07min; interval runs20.59minApple/15.73minMP. These are recorded
training-loop times, excluding extraction and setup, not guarantees for a new recipe.
