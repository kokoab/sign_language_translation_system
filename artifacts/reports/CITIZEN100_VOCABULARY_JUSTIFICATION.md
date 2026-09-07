# Citizen-100 vocabulary: evidence-based justification (ASL)

**Status:** `evidence_complete_pending_asl_review`
**Scope:** justifies the 100 canonical classes in `active/v17/citizen100_manifest.json`
**Evidence policy:** ASL-specific sources, published 2020 or later. Non-ASL corpora appear
only in §5, explicitly labelled as secondary cross-linguistic support.
**Reproduce:** `venv/bin/python scripts/audit_citizen100_asllex_frequency.py`
**Companion:** `artifacts/reports/IOS100_VOCABULARY_PROPOSAL.md`, `active/v17/citizen100_seed.json`

The original rationale was "commonly used in conversation," asserted without citation.
This document replaces that assertion with a measurement.

---

## 1. Headline claim

**The Citizen-100 vocabulary sits 1.43 standard deviations above the mean of the ASL
lexicon on published deaf-signer frequency ratings, with a median percentile rank of
94.6.** This is not a literature analogy — it is computed directly on our own 100 classes
against the ASL-LEX 2.0 norms, and it is reproducible from the repo.

## 2. Primary evidence: ASL-LEX 2.0 frequency norms

**Source.** Sehyr, Caselli, Cohen-Goldberg & Emmorey (2021), *The ASL-LEX 2.0 Project*,
*Journal of Deaf Studies and Deaf Education* 26(2), 263–277. It provides subjective
frequency ratings from 25–31 deaf signers for each of 2,723 ASL signs, on a 1–7 scale,
alongside phonological complexity, neighborhood density, lexical class, sign type and
sign duration.

**Why this join is valid.** Every class in our manifest already carries a
`citizen_asl_lex_code` (e.g. `B_01_068` for I). ASL Citizen draws its vocabulary from
ASL-LEX, so the code is an exact join key. **All 100 of our codes resolve against
ASL-LEX 2.0 with zero misses**, so this is a complete census of our vocabulary, not a
sample.

### 2.1 Where the 100 sit

| | n | mean | median | SD | range |
| --- | ---: | ---: | ---: | ---: | ---: |
| All ASL-LEX 2.0 | 2,723 | 4.13 | 4.25 | 1.17 | 1.00 – 6.96 |
| **Citizen-100** | 100 | **5.81** | 5.93 | 0.73 | 3.47 – 6.96 |

- Our mean is **+1.43 SD** above the lexicon mean.
- **70 of 100** fall in the top decile of ASL-LEX by rated frequency.
- **87 of 100** fall in the top quartile; **95 of 100** above the median; **100 of 100**
  above the 25th percentile. There is no low-frequency tail in this vocabulary.
- Mann–Whitney U (Citizen-100 > remaining 2,623 signs): U = 237,668, **p = 1.2 × 10⁻⁴³**.

### 2.2 Coverage of the high-frequency lexicon

Read the other direction — how much of ASL's most frequent vocabulary do we capture?

| ASL-LEX rank band | captured by Citizen-100 |
| --- | ---: |
| top 100 | 40 |
| top 200 | 61 |
| top 300 | 74 |
| top 500 | 84 |
| top 1,000 | 93 |

The 60 top-100 signs we do **not** cover are informative and mostly defensible: concrete
nouns (BOOK, MONEY, CAR, PAPER, BATHROOM), multi-sign discourse phrases (OH-I-SEE,
LET-ME-SEE, WE-WILL-SEE, I-LOVE-YOU), and community/identity terms (DEAF, PARENTS,
TEACH). Concrete nouns are context-dependent and expected to be excluded from a general
conversational core; **BATHROOM and DEAF are the two omissions hardest to defend** and
should be raised at ASL review.

## 3. What the norms say about the *shape* of the list

Measured against the rest of ASL-LEX, our 100 are not a random high-frequency draw —
they have a distinct and explicable profile.

### 3.1 Lexical class: verb- and function-weighted, noun-light

| Lexical class | Citizen-100 | rest of ASL-LEX |
| --- | ---: | ---: |
| Verb | 34% | 33.2% |
| Noun | 23% | **47.9%** |
| Adjective | 19% | 10.7% |
| Minor (function/grammatical) | **19%** | 3.8% |
| Adverb | 5% | 1.7% |

Nouns are halved; grammatical "Minor" items are **5× over-represented**. This is the
expected signature of a conversational core rather than a topic vocabulary, and it is now
an ASL-internal measurement rather than an argument by analogy.

### 3.2 Findings with direct engineering consequences

These fall out of the same join and matter for the recogniser, not just the write-up.

| ASL-LEX property | Citizen-100 | rest | p |
| --- | ---: | ---: | ---: |
| Sign duration (ms) | **614** | 861 | 9.5 × 10⁻¹⁷ |
| Neighborhood density 2.0 | **6.30** | 4.95 | 6.6 × 10⁻⁵ |
| Phonological complexity | **1.66** | 1.94 | 0.0096 |
| Iconicity | 3.36 | 3.38 | 0.76 (n.s.) |

1. **Our signs are ~29% shorter than the ASL-LEX average (614 ms vs 861 ms).** At 30 fps
   that is ≈18 frames of citation-form articulation against a 32-frame clip window. High-
   frequency vocabulary is *fast* vocabulary, which tightens the temporal-resolution
   budget for Stage 1 and the streaming CTC front end.
2. **Our signs have significantly more phonological neighbours (6.30 vs 4.95).** Choosing
   frequent signs buys usefulness at the cost of a denser minimal-pair space — an
   intrinsic confusability floor that no amount of extra data removes. This is the
   evidence-backed reason to expect residual confusion within the 100 and to keep the
   phonology-aware work in `active/v17/citizen100_phonology.json`.
3. **Phonological complexity is lower**, replicating Sehyr et al.'s reported
   frequency↔simplicity correlation within our subset.
4. **Iconicity shows no difference**, so Sehyr et al.'s frequency↔iconicity correlation
   does *not* reproduce here. Reported because it is a null result against expectation.

Separately, **57% of our 100 are one-handed** versus 38.5% of ASL-LEX. One-handed signs
carry less redundant articulatory evidence, which compounds the neighborhood-density
finding above.

## 4. Actionable defect found by this audit

Three classes are pinned to the **lower-frequency** ASL-LEX variant when a
higher-rated variant exists at equivalent ASL Citizen signer coverage. **ASL Citizen's
`WHAT1`/`WHAT2` numbering does not match ASL-LEX's `what_1`/`what_2` numbering** — the
ASL-LEX code is the only reliable key, and by that key our picks are:

| Class | Our code | Rating | Alternative | Rating | Signers train/val/test (ours → alt) |
| --- | --- | ---: | --- | ---: | --- |
| WHAT | `D_02_094` | **3.81** | `D_02_084` | **6.41** | 13/3/11 → 12/3/11 |
| HEAR | `J_02_006` | 3.53 | `E_01_097` | 5.03 | 13/3/11 → 13/3/11 |
| DOCTOR | `A_03_020` | 4.13 | `K_03_015` | 5.00 | 13/4/11 → 13/3/11 |

WHAT is the serious one: a 2.6-point rating gap on a core question sign, moving it from
the 38th to roughly the 97th percentile, at the cost of one training signer. THEY has a
higher-rated alternative too (`E_01_044`, 5.11 vs 4.89) but at materially worse coverage
(10/3/7), so our pick stands.

This is not strictly a bug — the manifest's stated rule was *"prefer the eligible exact
variant with the strongest balanced split coverage"*, i.e. selection by data availability,
not by frequency. But the rule produced a low-frequency variant for a core sign, and the
tradeoff should be an explicit decision. **Requires ASL-fluent review before any change**;
do not swap codes on the strength of a rating alone.

## 5. Secondary: cross-linguistic support for the number 100

*Not ASL, and not load-bearing.* Included only because no ASL conversational corpus of
comparable size exists, and because it is the only direct evidence that **100** is the
right cutoff rather than 50 or 300.

Corpus studies of other signed languages show token frequency is steeply concentrated:
the top 100 signs account for **52.8%** of 63,436 Auslan tokens (Johnston 2012) and
**57.2%** of 24,864 BSL conversational tokens (Cormier, Fenlon, Rentelis & Schembri 2011;
Fenlon et al. 2014); for NZSL, 116 signs cover ~50% of 100,000 tokens (McKee & Kennedy
2006). All four also find first-person pointing to be the most or second-most frequent
sign, supporting the 8 pronoun classes in `people_and_reference`.

These are pre-2020 and non-ASL. They are cited as a plausibility argument for the vocabulary
*size*; every claim about vocabulary *content* in §2–§4 rests on ASL data only.

## 6. Limitations

1. **Ratings, not token counts.** ASL-LEX frequency is subjective rating by deaf signers,
   not corpus-counted tokens. The BSL work found familiarity ratings correlate poorly with
   corpus frequency, so this measure may overstate the case. It is nonetheless the largest
   and most recent ASL-specific frequency resource that exists, and Sehyr et al. report it
   is stable across native and non-native raters.
2. **No ASL conversational frequency corpus.** ASLLRP (Neidle et al. 2022) offers 41,830
   lexical sign examples across ~6,000 entries and is the most promising route to a
   token-counted ASL check; it is elicited/narrative rather than conversational.
3. **Register.** ASL-LEX durations and ratings reflect citation-form isolated signing. The
   614 ms figure is a *lower bound on tightness* — coarticulated conversational signing
   will be faster still.
4. **Setting-critical classes are not frequency-justified.** HOSPITAL (57th pctile),
   DOCTOR (46th) and LESS (56th) are the weakest members on this evidence. They were
   originally included on deaf-healthcare-access grounds; that literature is deliberately
   out of scope here, so **those three currently have no justification in this document.**
5. **BATHROOM and DEAF** are high-frequency ASL signs absent from the 100 (§2.2).

## 7. References

1. Sehyr, Z. S., Caselli, N., Cohen-Goldberg, A. M. & Emmorey, K. (2021). The ASL-LEX 2.0
   Project: a database of lexical and phonological properties for 2,723 signs in American
   Sign Language. *JDSDE* 26(2), 263–277. https://academic.oup.com/jdsde/article/26/2/263/6142509
2. Desai, A., Berger, L., Minakov, F. et al. (2023). ASL Citizen: a community-sourced
   dataset for advancing isolated sign language recognition. *NeurIPS Datasets &
   Benchmarks.* https://arxiv.org/abs/2304.05934
3. Neidle, C. & Ballard, C. (2022). ASL video corpora & Sign Bank: resources available
   through the American Sign Language Linguistic Research Project (ASLLRP).
   https://arxiv.org/abs/2201.07899
4. Atwell, K., Bragg, D. & Alikhani, M. (2024). Studying and mitigating biases in sign
   language understanding models. https://arxiv.org/abs/2410.05206
   *(uses ASL Citizen with ASL-LEX lexical features to analyse performance disparities;
   cited for approach only — specific feature/accuracy directions not verified here.)*

**Secondary, cross-linguistic (§5 only):**

5. Johnston, T. (2012). Lexical frequency in signed languages. *JDSDE* 17(2), 163–193.
   https://academic.oup.com/jdsde/article/17/2/163/581884
6. Cormier, K., Fenlon, J., Rentelis, R. & Schembri, A. (2011). Lexical frequency in
   British Sign Language conversation. *Proc. LDLT3.*
   https://bslcorpusproject.org/wp-content/uploads/lexical-frequency-in-british-sign-language-ldlt3.pdf
7. Fenlon, J., Schembri, A., Rentelis, R., Vinson, D. & Cormier, K. (2014). Using
   conversational data to determine lexical frequency in BSL. *Lingua* 143, 187–202.
8. McKee, D. & Kennedy, G. (2006). The distribution of signs in New Zealand Sign Language.
   *Sign Language Studies* 6(4), 372–390.
