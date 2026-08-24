# RIT/Citizen100 exact-variant triage

**Status:** model-assisted audit only; no RIT class is automatically approved.

RIT and Citizen use different lexical identifiers. Frozen-model agreement
prioritizes exact ASLLRP-to-ASL-LEX review but cannot prove variant identity.

- Candidate clips: 292
- Candidate classes: 60
- Clip top-1 label agreement: 38.36%
- Clip top-5 label agreement: 60.62%
- Triage counts: `{"ambiguous_manual_variant_review_required": 22, "high_risk_manual_variant_review_required": 22, "model_consistent_manual_variant_review_required": 16}`

| Canonical | Citizen raw | ASL-LEX | Tier | Clips/signers | Top-1 | Top-5 | Triage |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| ANGRY | ANGRY | B_01_070 | pinned_raw_gloss_exact | 1/1 | 1/1 | 1/1 | ambiguous_manual_variant_review_required |
| ANSWER | ANSWER | E_03_065 | pinned_raw_gloss_exact | 4/4 | 3/4 | 3/4 | ambiguous_manual_variant_review_required |
| ASK | ASK | B_02_023 | pinned_raw_gloss_exact | 5/5 | 0/5 | 1/5 | high_risk_manual_variant_review_required |
| BAD | BAD | B_02_082 | pinned_raw_gloss_exact | 6/6 | 1/6 | 2/6 | high_risk_manual_variant_review_required |
| BIG | BIG | F_02_054 | pinned_raw_gloss_exact | 7/5 | 4/7 | 6/7 | model_consistent_manual_variant_review_required |
| COLD | COLD | C_02_068 | pinned_raw_gloss_exact | 5/5 | 1/5 | 2/5 | high_risk_manual_variant_review_required |
| COME | COME | C_03_074 | pinned_raw_gloss_exact | 6/6 | 0/6 | 0/6 | high_risk_manual_variant_review_required |
| DAY | DAY | C_01_044 | pinned_raw_gloss_exact | 6/5 | 5/6 | 6/6 | model_consistent_manual_variant_review_required |
| DOCTOR | DOCTOR1 | A_03_020 | canonical_label_only | 7/7 | 0/7 | 4/7 | ambiguous_manual_variant_review_required |
| DRINK | DRINK1 | B_02_012 | canonical_label_only | 5/5 | 0/5 | 1/5 | high_risk_manual_variant_review_required |
| FAMILY | FAMILY | C_01_025 | pinned_raw_gloss_exact | 7/7 | 7/7 | 7/7 | model_consistent_manual_variant_review_required |
| FRIEND | FRIEND | D_03_010 | pinned_raw_gloss_exact | 12/9 | 12/12 | 12/12 | model_consistent_manual_variant_review_required |
| GIVE | GIVE | F_01_023 | pinned_raw_gloss_exact | 2/2 | 1/2 | 2/2 | model_consistent_manual_variant_review_required |
| GO | GO | C_03_056 | pinned_raw_gloss_exact | 4/4 | 0/4 | 3/4 | ambiguous_manual_variant_review_required |
| HAVE | HAVE | B_03_057 | pinned_raw_gloss_exact | 8/8 | 2/8 | 5/8 | ambiguous_manual_variant_review_required |
| HEAR | HEAR2 | J_02_006 | canonical_label_only | 1/1 | 0/1 | 0/1 | high_risk_manual_variant_review_required |
| HELLO | HELLO | D_02_055 | pinned_raw_gloss_exact | 7/7 | 0/7 | 1/7 | high_risk_manual_variant_review_required |
| HELP | HELP | D_01_042 | pinned_raw_gloss_exact | 5/5 | 1/5 | 2/5 | high_risk_manual_variant_review_required |
| HOME | HOME | B_03_063 | pinned_raw_gloss_exact | 7/7 | 1/7 | 1/7 | high_risk_manual_variant_review_required |
| HOSPITAL | HOSPITAL1 | B_02_026 | canonical_label_only | 9/9 | 5/9 | 8/9 | model_consistent_manual_variant_review_required |
| HOT | HOT | F_02_093 | pinned_raw_gloss_exact | 7/7 | 1/7 | 1/7 | high_risk_manual_variant_review_required |
| IMPORTANT | IMPORTANT | B_01_081 | pinned_raw_gloss_exact | 8/8 | 4/8 | 4/8 | ambiguous_manual_variant_review_required |
| LANGUAGE | LANGUAGE | A_01_067 | pinned_raw_gloss_exact | 4/4 | 2/4 | 3/4 | ambiguous_manual_variant_review_required |
| LEARN | LEARN | B_01_042 | pinned_raw_gloss_exact | 7/7 | 4/7 | 6/7 | model_consistent_manual_variant_review_required |
| LESS | LESS | E_02_043 | pinned_raw_gloss_exact | 7/7 | 4/7 | 6/7 | model_consistent_manual_variant_review_required |
| LIKE | LIKE | F_03_063 | pinned_raw_gloss_exact | 8/7 | 2/8 | 3/8 | high_risk_manual_variant_review_required |
| LOVE | LOVE | G_01_068 | pinned_raw_gloss_exact | 10/7 | 6/10 | 7/10 | ambiguous_manual_variant_review_required |
| MAKE | MAKE | C_01_032 | pinned_raw_gloss_exact | 7/7 | 5/7 | 6/7 | model_consistent_manual_variant_review_required |
| MAN | MAN | C_01_040 | pinned_raw_gloss_exact | 10/9 | 1/10 | 6/10 | ambiguous_manual_variant_review_required |
| MORNING | MORNING | C_02_012 | pinned_raw_gloss_exact | 5/5 | 4/5 | 5/5 | model_consistent_manual_variant_review_required |
| NAME | NAME | D_01_021 | pinned_raw_gloss_exact | 3/3 | 2/3 | 3/3 | model_consistent_manual_variant_review_required |
| NIGHT | NIGHT1 | A_01_003 | canonical_label_only | 11/9 | 5/11 | 10/11 | ambiguous_manual_variant_review_required |
| NOW | NOW | C_03_062 | pinned_raw_gloss_exact | 5/5 | 4/5 | 5/5 | model_consistent_manual_variant_review_required |
| READ | READ | A_01_022 | pinned_raw_gloss_exact | 3/3 | 1/3 | 1/3 | high_risk_manual_variant_review_required |
| READY | READY | F_03_042 | pinned_raw_gloss_exact | 4/4 | 1/4 | 2/4 | ambiguous_manual_variant_review_required |
| SAD | SAD | B_02_053 | pinned_raw_gloss_exact | 5/5 | 5/5 | 5/5 | model_consistent_manual_variant_review_required |
| SAME | SAME1 | B_02_009 | canonical_label_only | 1/1 | 0/1 | 0/1 | high_risk_manual_variant_review_required |
| SEE | SEE | C_02_030 | pinned_raw_gloss_exact | 1/1 | 0/1 | 1/1 | ambiguous_manual_variant_review_required |
| SICK | SICK | B_01_072 | pinned_raw_gloss_exact | 1/1 | 0/1 | 1/1 | ambiguous_manual_variant_review_required |
| SLEEP | SLEEP | B_03_037 | pinned_raw_gloss_exact | 1/1 | 0/1 | 0/1 | high_risk_manual_variant_review_required |
| STOP | STOP | D_01_010 | pinned_raw_gloss_exact | 2/2 | 0/2 | 2/2 | ambiguous_manual_variant_review_required |
| TAKE | TAKE | G_01_093 | pinned_raw_gloss_exact | 2/2 | 0/2 | 1/2 | ambiguous_manual_variant_review_required |
| TALK | TALK1 | C_01_008 | canonical_label_only | 1/1 | 0/1 | 1/1 | ambiguous_manual_variant_review_required |
| THINK | THINK | C_03_053 | pinned_raw_gloss_exact | 7/7 | 2/7 | 7/7 | ambiguous_manual_variant_review_required |
| TIRED | TIRED | D_02_050 | pinned_raw_gloss_exact | 5/5 | 3/5 | 5/5 | model_consistent_manual_variant_review_required |
| TOMORROW | TOMORROW | F_02_040 | pinned_raw_gloss_exact | 7/6 | 1/7 | 1/7 | high_risk_manual_variant_review_required |
| TRY | TRY | B_02_034 | pinned_raw_gloss_exact | 4/3 | 2/4 | 2/4 | ambiguous_manual_variant_review_required |
| UNDERSTAND | UNDERSTAND | C_01_006 | pinned_raw_gloss_exact | 9/7 | 0/9 | 2/9 | high_risk_manual_variant_review_required |
| WAIT | WAIT | B_02_016 | pinned_raw_gloss_exact | 3/3 | 1/3 | 2/3 | ambiguous_manual_variant_review_required |
| WANT | WANT1 | E_01_025 | canonical_label_only | 4/3 | 1/4 | 2/4 | ambiguous_manual_variant_review_required |
| WATER | WATER | A_02_031 | pinned_raw_gloss_exact | 1/1 | 0/1 | 0/1 | high_risk_manual_variant_review_required |
| WHEN | WHEN | C_03_086 | pinned_raw_gloss_exact | 2/2 | 2/2 | 2/2 | model_consistent_manual_variant_review_required |
| WHERE | WHERE | B_02_035 | pinned_raw_gloss_exact | 4/4 | 0/4 | 0/4 | high_risk_manual_variant_review_required |
| WHO | WHO | C_01_041 | pinned_raw_gloss_exact | 3/3 | 0/3 | 0/3 | high_risk_manual_variant_review_required |
| WHY | WHY | D_01_067 | pinned_raw_gloss_exact | 1/1 | 0/1 | 0/1 | high_risk_manual_variant_review_required |
| WOMAN | WOMAN1 | C_02_028 | canonical_label_only | 4/4 | 0/4 | 3/4 | ambiguous_manual_variant_review_required |
| WORK | WORK | B_03_059 | pinned_raw_gloss_exact | 1/1 | 1/1 | 1/1 | ambiguous_manual_variant_review_required |
| WRITE | WRITE | D_01_051 | pinned_raw_gloss_exact | 1/1 | 0/1 | 0/1 | high_risk_manual_variant_review_required |
| YEAR | YEAR | B_02_015 | pinned_raw_gloss_exact | 3/3 | 3/3 | 3/3 | model_consistent_manual_variant_review_required |
| YESTERDAY | YESTERDAY | D_01_024 | pinned_raw_gloss_exact | 6/6 | 1/6 | 1/6 | high_risk_manual_variant_review_required |
