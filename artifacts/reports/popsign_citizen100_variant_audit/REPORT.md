# PopSign/Citizen100 exact-variant triage

**Status:** model-assisted preview audit only; no PopSign class is approved for training.

PopSign previews are downsampled and speed-normalized. Frozen-model agreement can
prioritize review but cannot establish that a PopSign gloss is the same lexical
variant as the pinned Citizen raw gloss and ASL-LEX code.

- Preview clips: 129
- Overlap classes: 43
- Clip top-1 label agreement: 33.33%
- Clip top-5 label agreement: 65.12%
- Triage counts: `{"ambiguous_manual_review_required": 23, "high_risk_manual_review_required": 11, "model_consistent_manual_review_required": 9}`

| Canonical | Citizen raw | ASL-LEX | PopSign | Top-1 | Top-5 | Triage |
| --- | --- | --- | --- | ---: | ---: | --- |
| BAD | BAD | B_02_082 | bad | 2/3 | 2/3 | ambiguous_manual_review_required |
| CHILD | CHILD | F_03_101 | child | 0/3 | 1/3 | high_risk_manual_review_required |
| DRINK | DRINK1 | B_02_012 | drink | 1/3 | 1/3 | high_risk_manual_review_required |
| FATHER | FATHER | B_01_077 | dad | 3/3 | 3/3 | model_consistent_manual_review_required |
| FIND | FIND | B_03_076 | find | 2/3 | 3/3 | model_consistent_manual_review_required |
| GIVE | GIVE | F_01_023 | give | 2/3 | 3/3 | model_consistent_manual_review_required |
| GO | GO | C_03_056 | go | 0/3 | 0/3 | high_risk_manual_review_required |
| GOODBYE | BYE | E_01_058 | bye | 1/3 | 2/3 | ambiguous_manual_review_required |
| HAPPY | HAPPY | C_03_078 | happy | 1/3 | 2/3 | ambiguous_manual_review_required |
| HAVE | HAVE | B_03_057 | have | 0/3 | 2/3 | ambiguous_manual_review_required |
| HEAR | HEAR2 | J_02_006 | hear | 0/3 | 1/3 | high_risk_manual_review_required |
| HELLO | HELLO | D_02_055 | hello | 2/3 | 2/3 | ambiguous_manual_review_required |
| HOME | HOME | B_03_063 | home | 2/3 | 3/3 | model_consistent_manual_review_required |
| HOT | HOT | F_02_093 | hot | 2/3 | 3/3 | model_consistent_manual_review_required |
| HUNGRY | HUNGRY | C_01_010 | hungry | 1/3 | 3/3 | ambiguous_manual_review_required |
| LIKE | LIKE | F_03_063 | like | 1/3 | 3/3 | ambiguous_manual_review_required |
| LISTEN | LISTEN | K_01_117 | listen | 0/3 | 0/3 | high_risk_manual_review_required |
| MAKE | MAKE | C_01_032 | make | 0/3 | 1/3 | high_risk_manual_review_required |
| MAN | MAN | C_01_040 | man | 2/3 | 2/3 | ambiguous_manual_review_required |
| MORNING | MORNING | C_02_012 | morning | 0/3 | 2/3 | ambiguous_manual_review_required |
| MOTHER | MOTHER | B_02_008 | mom | 0/3 | 2/3 | ambiguous_manual_review_required |
| NIGHT | NIGHT1 | A_01_003 | night | 1/3 | 3/3 | ambiguous_manual_review_required |
| NO | NO | C_03_041 | no | 1/3 | 2/3 | ambiguous_manual_review_required |
| NOW | NOW | C_03_062 | now | 0/3 | 2/3 | ambiguous_manual_review_required |
| PLEASE | PLEASE | B_02_007 | please | 2/3 | 3/3 | model_consistent_manual_review_required |
| READ | READ | A_01_022 | read | 0/3 | 0/3 | high_risk_manual_review_required |
| SAD | SAD | B_02_053 | sad | 0/3 | 0/3 | high_risk_manual_review_required |
| SAME | SAME1 | B_02_009 | same | 2/3 | 3/3 | model_consistent_manual_review_required |
| SEE | SEE | C_02_030 | see | 2/3 | 2/3 | ambiguous_manual_review_required |
| SICK | SICK | B_01_072 | sick | 0/3 | 3/3 | ambiguous_manual_review_required |
| SLEEP | SLEEP | B_03_037 | sleep | 2/3 | 3/3 | model_consistent_manual_review_required |
| TALK | TALK1 | C_01_008 | talk | 3/3 | 3/3 | model_consistent_manual_review_required |
| THANKYOU | THANKYOU | H_02_053 | thankyou | 1/3 | 2/3 | ambiguous_manual_review_required |
| THINK | THINK | C_03_053 | think | 0/3 | 2/3 | ambiguous_manual_review_required |
| TIME | TIME | B_02_080 | time | 0/3 | 1/3 | high_risk_manual_review_required |
| TOMORROW | TOMORROW | F_02_040 | tomorrow | 2/3 | 2/3 | ambiguous_manual_review_required |
| WAIT | WAIT | B_02_016 | wait | 0/3 | 1/3 | high_risk_manual_review_required |
| WATER | WATER | A_02_031 | water | 1/3 | 2/3 | ambiguous_manual_review_required |
| WHERE | WHERE | B_02_035 | where | 1/3 | 2/3 | ambiguous_manual_review_required |
| WHO | WHO | C_01_041 | who | 1/3 | 3/3 | ambiguous_manual_review_required |
| WHY | WHY | D_01_067 | why | 2/3 | 2/3 | ambiguous_manual_review_required |
| YES | YES | G_03_074 | yes | 0/3 | 2/3 | ambiguous_manual_review_required |
| YESTERDAY | YESTERDAY | D_01_024 | yesterday | 0/3 | 0/3 | high_risk_manual_review_required |
