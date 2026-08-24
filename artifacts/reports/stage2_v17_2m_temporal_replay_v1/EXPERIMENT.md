# Stage 2 genuine-sentence temporal pretraining and replay ablation

**Run date:** 2026-08-24 (Asia/Manila)  
**Decision:** temporal initialization retained as research evidence; not promoted

## Design

The rejected 448-token 2M-Flores auxiliary CTC transfer was replaced with a
label-free objective over genuine sentence motion. A frozen selected Stage-2 encoder
provided clean token trajectories while a student reconstructed randomly masked
contiguous spans and learned token-level contrastive alignment. The 100-sign CTC head
was frozen throughout pretraining. Only 2M-Flores `dev` rows were used; the reserved
`devtest` split and every project test split remained untouched.

The pretraining split is deterministic: 127 source sentences for training and 28 for
training-only validation. Bounded contiguous crops use no more than eight windows,
matching the compact deployment graph. The selected epoch was 13 after 80.79 seconds
on capped MPS. The checkpoint is
`artifacts/models/stage2_v17_2m_flores_temporal_pretrain_v1/temporal_pretrained.pth`,
SHA-256 `ba543b1fd9fa9dd5827b5c67b5f4b0b4a52748187d45513b17a87d64274519ab`.

## Controlled replay sweep

Each screen used seed 12701, the same selector-distillation teacher, 1,500 samples per
epoch, and identical real-data sub-ratios. The only changed variable was total
synthetic sampling mass.

| Synthetic mass | ASLLRP phrase edits | Local phrase edits | JONATHAN contextual edits |
| ---: | ---: | ---: | ---: |
| 0% | 12/24 | 8/259 | 41/254 |
| 10% | **12/24** | **7/259** | **41/254** |
| 20% | 12/24 | 8/259 | 41/254 |
| 30% | 12/24 | 8/259 | 41/254 |

The 10% setting was confirmed with seeds 12702 and 12703. Seed 12702 reproduced
12/7/41; seed 12703 reproduced 12/8/41. A label-checked selector audit then scanned
all 39,350 replay rows: the research selector owned 714, but only 85 were exact
against their training target. Oversampling only those exact rows by 64x produced
12/9/41 and was rejected.

The temporal alternative therefore improves the compact student's contextual result
from 43 to 41 edits, but regresses the genuine ASLLRP phrase gate from 11 to 12 edits.
It also remains behind the two-head research selector at 9/24 ASLLRP and 6/259 local
edits. It fails the no-regression promotion contract and does not replace either
retained model.

## Exact packaged artifacts

The distillation packager previously saved a bare CTC head even though validation ran
with the HOME/WHERE residual. That mismatch is fixed. Both compact artifacts below
contain the exact evaluated context-adapted graph and cold-reload their reported
metrics:

- Retained compact model: `artifacts/models/stage2_v17_compact_context_student_v1/model.pth`,
  SHA-256 `623f9b56141643704b3562a8d2fdcebe44269985b2f618eb8f0a471e857a2cf5`,
  with 11/24 ASLLRP, 7/259 local, and 43/254 contextual edits.
- Temporal research alternative: `artifacts/models/stage2_v17_temporal_context_student_v1/model.pth`,
  SHA-256 `a2568f6d38416a41a5f9b547224c50740874bd046cfa268c2f4a58166c88c4e6`,
  with 12/24 ASLLRP, 7/259 local, and 41/254 contextual edits.

The selected accuracy-research model remains
`artifacts/models/stage2_v17_general_ctc_selector_v1/model.pth`, SHA-256
`0782d052f0500164a2433ebfee86dcce7413c6bcffca03fae379871ece86dc3d`,
with 9/24 ASLLRP, 6/259 local, and 43/254 contextual edits. It is not a single Core ML
graph.

## Readiness decision

Stage 3 translation work may begin as an independently evaluated gloss-to-English
module, using reference gloss sequences and keeping recognition errors separate.
End-to-end Stage 3 accuracy must not be claimed from the current 12-phrase ASLLRP
development set.

The compact Stage-2 artifact is now correctly packageable for Core ML export, but
mobile deployment is not complete: v17 Stage-2 Core ML conversion, numerical parity,
app integration, and iPhone measurement are still required. General continuous-sign
promotion also remains blocked by weak and very small genuine-phrase evidence, not by
missing 2M-Flores acquisition or a software failure.

No Citizen, SemLex, local, 2M-Flores `devtest`, or already-consumed RIT test split was
accessed.
