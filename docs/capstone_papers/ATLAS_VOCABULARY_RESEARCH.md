# ATLAS: rationale for the selected 100 signs

Research note — 2026-09-30. For discussion before manuscript revision.

## Main finding

The strongest rationale combines **a bounded recognition scope** with **useful
vocabulary content**. The current 100-sign selection contains WHAT, WHERE, WHEN,
WHO, WHY, and HOW. These give the vocabulary labels for asking about things, places,
times, people, reasons, and manner. They strengthen the communication rationale;
they do not establish 100 as an optimal vocabulary size.

The 100 signs are ATLAS's selected subset. They should not be described as an
official, universally defined “Citizen-100” benchmark.

## Why the question signs matter

Boise State University's *Let's Chat! American Sign Language* places question-sign
practice in Level 1, Activity 2, including all six signs and WHICH. This is direct
educational support for treating information-seeking as a basic communication
function. It supports inclusion of the question signs, rather than validating the
entire ATLAS vocabulary as a complete beginner curriculum.
[University teaching resource](https://boisestate.pressbooks.pub/pathwaysasl/front-matter/introduction-page/).

| Sign label | Information sought | Present in ATLAS |
| --- | --- | --- |
| WHAT | A thing or activity | Yes |
| WHERE | A place | Yes |
| WHEN | A time | Yes |
| WHO | A person | Yes |
| WHY | A reason | Yes |
| HOW | Manner or method | Yes |

These are six question signs, not the whole ASL question system. ASL question
construction also uses facial and other grammatical information; a vocabulary
membership check does not demonstrate sentence understanding. Bill Vicars's ASL
teaching materials explain WH-question facial grammar and additional forms such as
WHICH and HOW-MANY. [ASL University, Lesson 15](https://www.lifeprint.com/asl101/lessons/lesson15.htm).

## Comparison with two published 100-class subsets

This table is a direct audit of class labels, not a recognition-accuracy comparison.
“Absent” means absent from that 100-class subset; the larger source vocabulary may
contain the sign. English gloss matches do not establish identical sign variants.

| Question label | ATLAS selected 100 | WLASL-100 | MS-ASL100 |
| --- | --- | --- | --- |
| WHAT | Present | Present | Present |
| WHERE | Present | Absent | Present |
| WHEN | Present | Absent | Present |
| WHO | Present | Present | Present |
| WHY | Present | Absent | Absent |
| HOW | Present | Present | Present |
| **Coverage of these six labels** | **6/6** | **3/6** | **5/6** |

Reproduction: read the ATLAS manifest's `classes[].canonical_label`; take the first
100 glosses from the authors' WLASL v0.3 metadata, following their top-K subset
instructions; take indices 0–99 from the official MS-ASL class list, following its
README's `label < 100` rule. Only metadata was inspected; no videos were acquired.

- [ATLAS manifest](../../active/v17/citizen100_manifest.json).
- [WLASL authors' repository and subset instructions](https://github.com/dxli94/WLASL),
  [v0.3 class metadata](https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json).
  One-based positions: WHAT 31, WHERE 255, WHEN 347, WHO 8, WHY 136, HOW 87.
- [Official MS-ASL metadata download](https://www.microsoft.com/en-gb/download/details.aspx?id=100121).
  Read `MS-ASL/MSASL_classes.json` and `MS-ASL/README.md` inside the archive.
  One-based positions: WHAT 16, WHERE 31, WHEN 84, WHO 70, WHY 144, HOW 60.
  [Vaezi Joze and Koller (2019)](https://www.microsoft.com/applied-sciences/uploads/publications/3/ms-asl.pdf)
  describes the nested subsets and their construction.

The defensible statement is: **Neither of the two examined 100-class benchmark
subsets includes all six question labels, whereas ATLAS's selection does.**
Two comparisons do not establish what “most datasets” include. This difference
supports the selection's information-seeking coverage, not a claim of overall
superiority or better translation accuracy.

## Additional evidence for everyday relevance

The existing [local vocabulary audit](../../artifacts/reports/CITIZEN100_VOCABULARY_JUSTIFICATION.md)
matched all 100 exact lexical codes to published ASL frequency norms. It reports:

| Measure | Selected 100 | Reference lexicon |
| --- | ---: | ---: |
| Mean subjective frequency, scale 1–7 | 5.81 | 4.13 |
| Signs above the reference median | 95 of 100 | — |
| Signs in the reference top quartile | 87 of 100 | — |

These are deaf signers' ratings of how frequently signs are used, not measured
percentages of conversations covered. The values are the project's existing audit
results; the publications supply the norms, not an endorsement of ATLAS's selection.
The norms span [Caselli et al. (2017)](https://doi.org/10.3758/s13428-016-0742-0)
and [Sevcikova Sehyr et al. (2021)](https://doi.org/10.1093/deafed/enaa038).
Retain both when documenting the rating provenance.

## Suggested paper wording

> ATLAS uses a 100-sign prototype vocabulary containing foundational signs,
> including WHAT, WHERE, WHEN, WHO, WHY, and HOW. These signs provide vocabulary
> for basic information-seeking and are included in introductory ASL learning
> activities (Boise State University, n.d.). The selected vocabulary gives the
> prototype a defined recognition scope and includes signs for asking about people,
> things, places, times, reasons, and manner.

For the recommendations section: future vocabulary expansion can address additional
communication situations identified with Deaf users, supported by signer-diverse
recordings and signer-disjoint evaluation.

This paragraph keeps descriptive wording for the project's data. Named benchmark
comparisons belong in this research note or a cited comparison section if approved.

## Short panel answer

“Our prototype has a fixed vocabulary of 100 signs, including foundational signs
such as what, where, when, who, why, and how. Introductory ASL materials teach these
question signs early. Their inclusion gives the selected vocabulary a clear
information-seeking purpose within the prototype’s 100-sign scope.”

Author direction, clarified after this research: emphasize the contribution of the
selected signs, state scope neutrally, and put expansion in recommendations. Avoid
repeated deficit framing. Recommended term:
**a 100-sign prototype vocabulary containing foundational signs**. Use **question
signs** in ordinary explanations and **interrogative signs** where useful.
