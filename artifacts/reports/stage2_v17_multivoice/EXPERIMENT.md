# Stage 2 63-voice transition-preserving adaptation

**Run date:** 2026-08-22 (Asia/Manila)  
**Status:** selected development candidate; independent signer/capture confirmation
is still required

## Result

The selected candidate improves all three available development-validation gates over
the previously retained context-adapted Stage 2 model.

| Gate | Previous retained candidate | 63-voice candidate |
| --- | ---: | ---: |
| ASLLRP genuine phrases | 12/24 edits, 50.0000% WER | **11/24, 45.8333% WER** |
| Local phrases | 11/259 edits, 4.2471% WER | **7/259, 2.7027% WER** |
| JONATHAN contextual signs | 46/254 edits, 18.1102% WER | **44/254, 17.3228% WER** |

The base CTC checkpoint selected seed 4702 epoch 12 after 142.07 seconds. Its SHA-256
is `6a72ca836247fe8717e0b4a7f930b11b291aa2889b49cc839a9fbc6d86d6cf8e`.
The loadable model after applying the already fitted train-only HOME/WHERE context
residual at its previously selected weight 1.5 is:

`artifacts/models/stage2_v17_multivoice_transfer_context_adapted_v3/model.pth`

SHA-256: `f2ea9d99796e71b7355657a5bbcf791bdfedddec07adb54a053d9eba4292164b`.
A cold reload reproduced every metric.

## Voice inventory

The frozen pool contains 3,978 train-only trajectories and 67 dataset-local signer
identities:

- 1,475 ASL Citizen official-training clips from 32 signer IDs;
- 1,388 exact-variant, quality-gated SemLex official-training clips from 32 signer IDs;
- 1,115 contextual ASLLRP training segments from BENJAMIN_JAMES_BAHAN, CORY, and
  RACHEL.

Four identities have only one class. The synthesis eligibility rule requires at least
two distinct signs, leaving **63 usable style voices**: 29 Citizen, 31 SemLex, and 3
ASLLRP. These are dataset-local identities; cross-dataset identity equivalence is not
claimed.

The pool is
`data/local/stage2_v17_synthetic/train_only_multivoice_pool_v3.npz`, SHA-256
`ee079873023b782bbc64c9fe4c64b32f4be2cb6d95494fa3fe446443e18e6653`.
SemLex frozen encoding took 29.43 seconds on MPS and peaked at 103,579,648 driver
bytes.

## Transition-preserving style transfer

Directly joining isolated Citizen/SemLex signs was tested first and rejected. It kept
the ASLLRP phrase result at 45.8333% but worsened local phrases to 3.4749%. A lower-rate
second-stage adaptation correctly selected epoch 0 for all seeds, proving that the
direct isolated-bridge data added no validated value.

The selected design instead separates **transition content** from **voice style**:

1. every two- or three-sign transition trajectory comes from exactly one genuine
   continuous ASLLRP training signer;
2. 60 additional official-training Citizen/SemLex voices transfer only their observed
   sign-duration distributions and neutral endpoint context;
3. the core sign and boundary trajectories are never replaced by invented
   Citizen/SemLex transitions;
4. the 18,000-row plan contains 6,000 native ASLLRP voice sequences, 6,000 transferred
   style sequences balanced at 100 per additional voice, and 6,000 full-vocabulary
   Citizen replay sequences; and
5. JONATHAN and all Citizen, SemLex, and local test material remain excluded.

The plan is `active/v17/stage2_multivoice_transfer_plan_v17.json`, SHA-256
`8c24e8268c38d840a8b10a9a59caf5d9dfecd607bac69f0dc9887d7f9cb34dc4`.

This is a recognition-oriented latent-style transfer experiment. The result supports
the usefulness of multi-signer timing/style diversity, but recognition WER does not
prove that rendered motion is perceptually natural, that facial grammar is correct,
or that a particular unseen person's identity has been cloned.

## Data and evaluation integrity

- Citizen official test: not accessed.
- SemLex official test: not accessed.
- Local sealed test: not accessed.
- 2M-Flores `devtest`: not accessed.
- Already-consumed RIT external test: not accessed.
- JONATHAN was used only by the pre-existing development-validation gates and never
  as a synthesis/style source.

The ASLLRP phrase set remains only 12 clips and 24 tokens. The one-token improvement
is real for this development set but too small for a generalization claim. The next
accuracy claim requires a new signer/capture evaluation set; visual naturalness
requires rendered pose/video and fluent-signer assessment.

Twelve focused Stage 2/model/data tests, Python compilation, a full 18,000-row plan
audit, six JSON parses, saved-artifact cold reload, and `git diff --check` pass. The
audit verified all 63 declared style voices, exact pool-label/source alignment, and
the absence of JONATHAN from synthesis inputs.
