# Local Citizen100 model-assisted triage

**Status:** audit only; no local class is automatically approved.

The local corpus lacks signer and exact lexical-variant IDs. Frozen-model
agreement is only a mismatch screen and is not accuracy evidence.

- Candidate clips/classes: 356/89
- Clip top-1/top-5 agreement: 51.69% / 81.18%
- Triage counts: `{"ambiguous_manual_variant_review_required": 27, "high_risk_manual_variant_review_required": 8, "model_consistent_manual_variant_review_required": 54}`

| Class | Raw gloss | ASL-LEX | Tier | Top-1 | Top-5 | Triage |
| --- | --- | --- | --- | ---: | ---: | --- |
| ANGRY | ANGRY | B_01_070 | canonical_and_pinned_raw_text_equal | 1/4 | 4/4 | ambiguous_manual_variant_review_required |
| ANSWER | ANSWER | E_03_065 | canonical_and_pinned_raw_text_equal | 2/4 | 3/4 | model_consistent_manual_variant_review_required |
| ASK | ASK | B_02_023 | canonical_and_pinned_raw_text_equal | 2/4 | 3/4 | model_consistent_manual_variant_review_required |
| BAD | BAD | B_02_082 | canonical_and_pinned_raw_text_equal | 3/4 | 4/4 | model_consistent_manual_variant_review_required |
| BIG | BIG | F_02_054 | canonical_and_pinned_raw_text_equal | 1/4 | 3/4 | ambiguous_manual_variant_review_required |
| CHILD | CHILD | F_03_101 | canonical_and_pinned_raw_text_equal | 3/4 | 3/4 | model_consistent_manual_variant_review_required |
| COLD | COLD | C_02_068 | canonical_and_pinned_raw_text_equal | 3/4 | 4/4 | model_consistent_manual_variant_review_required |
| COME | COME | C_03_074 | canonical_and_pinned_raw_text_equal | 0/4 | 1/4 | high_risk_manual_variant_review_required |
| DAY | DAY | C_01_044 | canonical_and_pinned_raw_text_equal | 3/4 | 4/4 | model_consistent_manual_variant_review_required |
| DIFFERENT | DIFFERENT | E_03_095 | canonical_and_pinned_raw_text_equal | 4/4 | 4/4 | model_consistent_manual_variant_review_required |
| DOCTOR | DOCTOR1 | A_03_020 | canonical_only_variant_review_required | 3/4 | 3/4 | model_consistent_manual_variant_review_required |
| DRINK | DRINK1 | B_02_012 | canonical_only_variant_review_required | 3/4 | 3/4 | model_consistent_manual_variant_review_required |
| EASY | EASY | D_01_070 | canonical_and_pinned_raw_text_equal | 3/4 | 4/4 | model_consistent_manual_variant_review_required |
| EAT | EAT1 | B_02_002 | canonical_only_variant_review_required | 2/4 | 3/4 | model_consistent_manual_variant_review_required |
| EXCITED | EXCITED | H_01_102 | canonical_and_pinned_raw_text_equal | 4/4 | 4/4 | model_consistent_manual_variant_review_required |
| FAMILY | FAMILY | C_01_025 | canonical_and_pinned_raw_text_equal | 4/4 | 4/4 | model_consistent_manual_variant_review_required |
| FATHER | FATHER | B_01_077 | canonical_and_pinned_raw_text_equal | 2/4 | 4/4 | model_consistent_manual_variant_review_required |
| FEEL | FEEL | C_02_001 | canonical_and_pinned_raw_text_equal | 3/4 | 3/4 | model_consistent_manual_variant_review_required |
| FRIEND | FRIEND | D_03_010 | canonical_and_pinned_raw_text_equal | 0/4 | 3/4 | ambiguous_manual_variant_review_required |
| GIVE | GIVE | F_01_023 | canonical_and_pinned_raw_text_equal | 0/4 | 3/4 | ambiguous_manual_variant_review_required |
| GO | GO | C_03_056 | canonical_and_pinned_raw_text_equal | 0/4 | 4/4 | ambiguous_manual_variant_review_required |
| GOOD | GOOD | B_01_052 | canonical_and_pinned_raw_text_equal | 1/4 | 3/4 | ambiguous_manual_variant_review_required |
| GOODBYE | BYE | E_01_058 | canonical_only_variant_review_required | 0/4 | 0/4 | high_risk_manual_variant_review_required |
| HAPPY | HAPPY | C_03_078 | canonical_and_pinned_raw_text_equal | 0/4 | 3/4 | ambiguous_manual_variant_review_required |
| HEAR | HEAR2 | J_02_006 | canonical_only_variant_review_required | 0/4 | 3/4 | ambiguous_manual_variant_review_required |
| HELLO | HELLO | D_02_055 | canonical_and_pinned_raw_text_equal | 1/4 | 4/4 | ambiguous_manual_variant_review_required |
| HELP | HELP | D_01_042 | canonical_and_pinned_raw_text_equal | 1/4 | 3/4 | ambiguous_manual_variant_review_required |
| HOSPITAL | HOSPITAL1 | B_02_026 | canonical_only_variant_review_required | 2/4 | 4/4 | model_consistent_manual_variant_review_required |
| HOT | HOT | F_02_093 | canonical_and_pinned_raw_text_equal | 2/4 | 2/4 | ambiguous_manual_variant_review_required |
| HOW | HOW1 | D_02_082 | canonical_only_variant_review_required | 0/4 | 3/4 | ambiguous_manual_variant_review_required |
| IMPORTANT | IMPORTANT | B_01_081 | canonical_and_pinned_raw_text_equal | 4/4 | 4/4 | model_consistent_manual_variant_review_required |
| KNOW | KNOW | C_01_048 | canonical_and_pinned_raw_text_equal | 3/4 | 4/4 | model_consistent_manual_variant_review_required |
| LANGUAGE | LANGUAGE | A_01_067 | canonical_and_pinned_raw_text_equal | 2/4 | 4/4 | model_consistent_manual_variant_review_required |
| LEARN | LEARN | B_01_042 | canonical_and_pinned_raw_text_equal | 3/4 | 4/4 | model_consistent_manual_variant_review_required |
| LESS | LESS | E_02_043 | canonical_and_pinned_raw_text_equal | 2/4 | 2/4 | ambiguous_manual_variant_review_required |
| LIKE | LIKE | F_03_063 | canonical_and_pinned_raw_text_equal | 1/4 | 4/4 | ambiguous_manual_variant_review_required |
| LOVE | LOVE | G_01_068 | canonical_and_pinned_raw_text_equal | 4/4 | 4/4 | model_consistent_manual_variant_review_required |
| MAKE | MAKE | C_01_032 | canonical_and_pinned_raw_text_equal | 3/4 | 3/4 | model_consistent_manual_variant_review_required |
| MAN | MAN | C_01_040 | canonical_and_pinned_raw_text_equal | 3/4 | 3/4 | model_consistent_manual_variant_review_required |
| MAYBE | MAYBE | B_03_022 | canonical_and_pinned_raw_text_equal | 3/4 | 3/4 | model_consistent_manual_variant_review_required |
| MORE | MORE | B_03_032 | canonical_and_pinned_raw_text_equal | 3/4 | 4/4 | model_consistent_manual_variant_review_required |
| MORNING | MORNING | C_02_012 | canonical_and_pinned_raw_text_equal | 4/4 | 4/4 | model_consistent_manual_variant_review_required |
| MOTHER | MOTHER | B_02_008 | canonical_and_pinned_raw_text_equal | 1/4 | 4/4 | ambiguous_manual_variant_review_required |
| MY | MY | C_01_060 | canonical_and_pinned_raw_text_equal | 2/4 | 3/4 | model_consistent_manual_variant_review_required |
| NAME | NAME | D_01_021 | canonical_and_pinned_raw_text_equal | 4/4 | 4/4 | model_consistent_manual_variant_review_required |
| NEED | NEED | C_02_034 | canonical_and_pinned_raw_text_equal | 1/4 | 3/4 | ambiguous_manual_variant_review_required |
| NIGHT | NIGHT1 | A_01_003 | canonical_only_variant_review_required | 4/4 | 4/4 | model_consistent_manual_variant_review_required |
| NO | NO | C_03_041 | canonical_and_pinned_raw_text_equal | 2/4 | 4/4 | model_consistent_manual_variant_review_required |
| NOW | NOW | C_03_062 | canonical_and_pinned_raw_text_equal | 3/4 | 4/4 | model_consistent_manual_variant_review_required |
| OUR | OUR | G_02_067 | canonical_and_pinned_raw_text_equal | 4/4 | 4/4 | model_consistent_manual_variant_review_required |
| PLEASE | PLEASE | B_02_007 | canonical_and_pinned_raw_text_equal | 4/4 | 4/4 | model_consistent_manual_variant_review_required |
| READ | READ | A_01_022 | canonical_and_pinned_raw_text_equal | 1/4 | 3/4 | ambiguous_manual_variant_review_required |
| READY | READY | F_03_042 | canonical_and_pinned_raw_text_equal | 2/4 | 4/4 | model_consistent_manual_variant_review_required |
| SAD | SAD | B_02_053 | canonical_and_pinned_raw_text_equal | 3/4 | 4/4 | model_consistent_manual_variant_review_required |
| SCHOOL | SCHOOL | C_03_089 | canonical_and_pinned_raw_text_equal | 4/4 | 4/4 | model_consistent_manual_variant_review_required |
| SEE | SEE | C_02_030 | canonical_and_pinned_raw_text_equal | 4/4 | 4/4 | model_consistent_manual_variant_review_required |
| SIGN | SIGN | E_03_050 | canonical_and_pinned_raw_text_equal | 0/4 | 0/4 | high_risk_manual_variant_review_required |
| SLEEP | SLEEP | B_03_037 | canonical_and_pinned_raw_text_equal | 2/4 | 4/4 | model_consistent_manual_variant_review_required |
| SMALL | SMALL | D_01_030 | canonical_and_pinned_raw_text_equal | 2/4 | 3/4 | model_consistent_manual_variant_review_required |
| SORRY | SORRY | C_01_020 | canonical_and_pinned_raw_text_equal | 3/4 | 4/4 | model_consistent_manual_variant_review_required |
| STOP | STOP | D_01_010 | canonical_and_pinned_raw_text_equal | 1/4 | 3/4 | ambiguous_manual_variant_review_required |
| TAKE | TAKE | G_01_093 | canonical_and_pinned_raw_text_equal | 0/4 | 1/4 | high_risk_manual_variant_review_required |
| TELL | TELL | C_02_013 | canonical_and_pinned_raw_text_equal | 4/4 | 4/4 | model_consistent_manual_variant_review_required |
| THANKYOU | THANKYOU | H_02_053 | canonical_and_pinned_raw_text_equal | 3/4 | 4/4 | model_consistent_manual_variant_review_required |
| THEY | THEY1 | F_02_103 | canonical_only_variant_review_required | 2/4 | 2/4 | ambiguous_manual_variant_review_required |
| THINK | THINK | C_03_053 | canonical_and_pinned_raw_text_equal | 3/4 | 4/4 | model_consistent_manual_variant_review_required |
| TIME | TIME | B_02_080 | canonical_and_pinned_raw_text_equal | 3/4 | 4/4 | model_consistent_manual_variant_review_required |
| TIRED | TIRED | D_02_050 | canonical_and_pinned_raw_text_equal | 3/4 | 4/4 | model_consistent_manual_variant_review_required |
| TOMORROW | TOMORROW | F_02_040 | canonical_and_pinned_raw_text_equal | 4/4 | 4/4 | model_consistent_manual_variant_review_required |
| TRY | TRY | B_02_034 | canonical_and_pinned_raw_text_equal | 3/4 | 4/4 | model_consistent_manual_variant_review_required |
| UNDERSTAND | UNDERSTAND | C_01_006 | canonical_and_pinned_raw_text_equal | 0/4 | 1/4 | high_risk_manual_variant_review_required |
| USE | USE | D_03_043 | canonical_and_pinned_raw_text_equal | 3/4 | 3/4 | model_consistent_manual_variant_review_required |
| WAIT | WAIT | B_02_016 | canonical_and_pinned_raw_text_equal | 1/4 | 3/4 | ambiguous_manual_variant_review_required |
| WANT | WANT1 | E_01_025 | canonical_only_variant_review_required | 0/4 | 1/4 | high_risk_manual_variant_review_required |
| WATER | WATER | A_02_031 | canonical_and_pinned_raw_text_equal | 3/4 | 3/4 | model_consistent_manual_variant_review_required |
| WEEK | WEEK | B_02_079 | canonical_and_pinned_raw_text_equal | 1/4 | 4/4 | ambiguous_manual_variant_review_required |
| WHAT | WHAT1 | D_02_094 | canonical_only_variant_review_required | 0/4 | 0/4 | high_risk_manual_variant_review_required |
| WHEN | WHEN | C_03_086 | canonical_and_pinned_raw_text_equal | 1/4 | 4/4 | ambiguous_manual_variant_review_required |
| WHERE | WHERE | B_02_035 | canonical_and_pinned_raw_text_equal | 1/4 | 3/4 | ambiguous_manual_variant_review_required |
| WHO | WHO | C_01_041 | canonical_and_pinned_raw_text_equal | 2/4 | 4/4 | model_consistent_manual_variant_review_required |
| WHY | WHY | D_01_067 | canonical_and_pinned_raw_text_equal | 1/4 | 3/4 | ambiguous_manual_variant_review_required |
| WOMAN | WOMAN1 | C_02_028 | canonical_only_variant_review_required | 0/4 | 2/4 | ambiguous_manual_variant_review_required |
| WORK | WORK | B_03_059 | canonical_and_pinned_raw_text_equal | 2/4 | 4/4 | model_consistent_manual_variant_review_required |
| WRITE | WRITE | D_01_051 | canonical_and_pinned_raw_text_equal | 0/4 | 2/4 | ambiguous_manual_variant_review_required |
| YEAR | YEAR | B_02_015 | canonical_and_pinned_raw_text_equal | 3/4 | 4/4 | model_consistent_manual_variant_review_required |
| YES | YES | G_03_074 | canonical_and_pinned_raw_text_equal | 0/4 | 3/4 | ambiguous_manual_variant_review_required |
| YESTERDAY | YESTERDAY | D_01_024 | canonical_and_pinned_raw_text_equal | 3/4 | 3/4 | model_consistent_manual_variant_review_required |
| YOU | YOU | D_02_065 | canonical_and_pinned_raw_text_equal | 3/4 | 4/4 | model_consistent_manual_variant_review_required |
| YOUR | YOUR | F_01_052 | canonical_and_pinned_raw_text_equal | 0/4 | 1/4 | high_risk_manual_variant_review_required |
