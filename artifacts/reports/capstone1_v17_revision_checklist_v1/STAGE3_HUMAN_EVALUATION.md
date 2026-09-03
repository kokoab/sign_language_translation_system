# English-output human evaluation form

This form complements automatic exact-match and chrF++ results. It does not contain
fabricated human scores. Recruit three raters if possible and have them work
independently.

## Instructions for raters

For each gloss sequence, rate the generated English sentence from 1 to 5 on:

| Dimension | 1 | 3 | 5 |
| --- | --- | --- | --- |
| Semantic adequacy | Meaning is mostly wrong or missing | Main idea is present but some meaning is wrong/unclear | Meaning is fully preserved |
| Grammaticality | Very difficult or ungrammatical | Understandable with noticeable errors | Fully grammatical English |
| Completeness/faithfulness | Important content added, omitted, or changed | Minor addition/omission | No unjustified addition, omission, or change |
| Overall acceptability | Unusable | Usable with editing | Natural and ready to use |

Do not score punctuation or harmless article choices harshly unless they change the
meaning. Record a brief issue code where relevant: `ADD` (added meaning), `OMIT`,
`TENSE`, `PRONOUN`, `ARTICLE`, `WORD`, `ORDER`, or `OTHER`.

## Blinded rating sheet

The 26 long examples are the complete controlled five-or-more-gloss held-out set. Four
shorter examples are included to avoid evaluating only one sentence template.

| # | Gloss sequence | Generated English | Adequacy 1–5 | Grammar 1–5 | Faithfulness 1–5 | Overall 1–5 | Issue/comment |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | TOMORROW MORNING I WANT GO HOME READ NAME | Tomorrow morning, I want to go home and read the name. | | | | | |
| 2 | TOMORROW MORNING I WANT GO HOME SEE FRIEND | Tomorrow morning, I want to go home and see my friend. | | | | | |
| 3 | TOMORROW MORNING I WANT GO SCHOOL READ SIGN | Tomorrow morning, I want to go to school and read the sign. | | | | | |
| 4 | TOMORROW MORNING I WANT GO SCHOOL READ NAME | Tomorrow morning, I want to go to school and read the name. | | | | | |
| 5 | TOMORROW MORNING I WANT GO SCHOOL LEARN LANGUAGE | Tomorrow morning, I want to go to school and learn the language. | | | | | |
| 6 | TOMORROW MORNING I WANT GO HOSPITAL READ NAME | Tomorrow morning, I want to go to the hospital and read the name. | | | | | |
| 7 | TOMORROW MORNING I WANT GO HOSPITAL LEARN SIGN | Tomorrow morning, I want to go to the hospital and learn the sign. | | | | | |
| 8 | TOMORROW MORNING I WANT GO HOSPITAL FIND DOCTOR | Tomorrow morning, I want to go to the hospital and find the doctor. | | | | | |
| 9 | TOMORROW MORNING YOU WANT GO HOME LEARN SIGN | Tomorrow morning, you want to go home and learn the sign. | | | | | |
| 10 | TOMORROW MORNING YOU WANT GO HOME SEE FRIEND | Tomorrow morning, you want to go home and see my friend. | | | | | |
| 11 | TOMORROW MORNING YOU WANT GO SCHOOL LEARN LANGUAGE | Tomorrow morning, you want to go to school and learn the language. | | | | | |
| 12 | TOMORROW MORNING YOU WANT GO SCHOOL LEARN SIGN | Tomorrow morning, you want to go to school and learn the sign. | | | | | |
| 13 | TOMORROW MORNING WE WANT GO SCHOOL LEARN SIGN | Tomorrow morning, we want to go to school and learn the sign. | | | | | |
| 14 | TOMORROW MORNING WE WANT GO SCHOOL FIND FRIEND | Tomorrow morning, we want to go to school and find my friend. | | | | | |
| 15 | TOMORROW MORNING WE WANT GO SCHOOL SEE DOCTOR | Tomorrow morning, we want to go to school and see the doctor. | | | | | |
| 16 | TOMORROW MORNING WE WANT GO HOSPITAL LEARN SIGN | Tomorrow morning, we want to go to the hospital and learn the sign. | | | | | |
| 17 | YESTERDAY I GO SCHOOL LEARN LANGUAGE COME HOME | Yesterday, I went to school, learned the language, and came home. | | | | | |
| 18 | YESTERDAY I GO SCHOOL FIND TIME COME HOME | Yesterday, I went to school, found the time, and came home. | | | | | |
| 19 | YESTERDAY I GO HOSPITAL FIND TIME COME HOME | Yesterday, I went to the hospital, found the time, and came home. | | | | | |
| 20 | YESTERDAY YOU GO SCHOOL FIND TIME COME HOME | Yesterday, you went to school, found the time, and came home. | | | | | |
| 21 | YESTERDAY WE GO SCHOOL SEE DOCTOR COME HOME | Yesterday, we went to school, saw the doctor, and came home. | | | | | |
| 22 | YESTERDAY WE GO HOSPITAL FIND TIME COME HOME | Yesterday, we went to the hospital, found the time, and came home. | | | | | |
| 23 | NOW YOU FEEL SICK YOU NEED HELP | You feel sick now and need help. | | | | | |
| 24 | MY MOTHER FEEL SICK NEED DOCTOR NOW | My mother feels sick and needs a doctor now. | | | | | |
| 25 | YESTERDAY I GO SCHOOL READ SIGN COME HOME NOW I FEEL TIRED | Yesterday, I went to school, read the sign, and came home. Now, I feel tired. | | | | | |
| 26 | YESTERDAY I GO SCHOOL LEARN LANGUAGE COME HOME NOW I FEEL TIRED | Yesterday, I went to school, learned the language, and came home. Now, I feel tired. | | | | | |
| 27 | SEE YOU TOMORROW | See you tomorrow. | | | | | |
| 28 | YOU NEED WHAT | What do you need? | | | | | |
| 29 | NOW STUDENT FEEL SICK | Now, the student is feeling sick. | | | | | |
| 30 | MORNING I FEEL SICK | In the morning, I am feeling sick. | | | | | |

## Reporting template

Report, for each dimension, the mean, standard deviation, and number of ratings. Also
report per-rater means and an inter-rater agreement statistic. Krippendorff's alpha is
appropriate for ordinal ratings and tolerates missing values; weighted Cohen's kappa
can be used only for two raters.

| Dimension | Mean | SD | Agreement | N ratings |
| --- | ---: | ---: | ---: | ---: |
| Semantic adequacy | | | | |
| Grammaticality | | | | |
| Completeness/faithfulness | | | | |
| Overall acceptability | | | | |

Suggested success rule, declared before rating: mean semantic adequacy and overall
acceptability of at least 4.0/5, with no example receiving a majority adequacy score of
1 or 2. Keep the individual ratings and comments as appendix evidence.
