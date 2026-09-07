# Citizen-100 vocabulary: source list

Companion to `artifacts/reports/CITIZEN100_VOCABULARY_JUSTIFICATION.md`.
Every entry below states **what it contributes**, **how it was verified**, and
**what it does not support**.

**Verification key:**
`[V-full]` opened and read the primary text · `[V-meta]` publisher page confirmed
(title/authors/DOI) · `[V-2nd]` figures taken from another paper citing it, primary
text not accessible.

---

## 0. Direct answer: is it one study?

**No — two, and they are jointly required.** The `SignFrequency(M)` column we measured
against was built in two rating rounds, and the ASL-LEX project's own download page
instructs users to cite **both** papers. Splitting our 100 classes by ASL-LEX batch:

| Rating round | ASL-LEX batches | Signs rated | **Of our 100** |
| --- | --- | ---: | ---: |
| Caselli et al. **2017** (ASL-LEX 1.0) | A–D | 993 | **75** |
| Sehyr et al. **2021** (ASL-LEX 2.0) | E–K | ~1,735 new | **25** |

**This breaks the "2020 and newer" rule, and it cannot be fixed.** The frequency ratings
for three-quarters of our vocabulary were collected for the 2017 paper. The 2021 paper is
the current release of the database and the primary citation, but citing it alone for our
100 would misattribute where 75% of the underlying data came from. Recommended framing:
*"ASL-LEX 2.0 (Sehyr et al., 2021), incorporating ratings from Caselli et al. (2017)."*

---

## 1. Primary ASL evidence — the frequency norms

### 1.1 ASL-LEX 2.0 `[V-meta]` — current release, primary citation

> Sevcikova Sehyr, Z., Caselli, N., Cohen-Goldberg, A. M., & Emmorey, K. (2021).
> The ASL-LEX 2.0 Project: A database of lexical and phonological properties for 2,723
> signs in American Sign Language. *Journal of Deaf Studies and Deaf Education*, 26(2),
> 263–277.

- **DOI:** https://doi.org/10.1093/deafed/enaa038
- **Publisher:** https://academic.oup.com/jdsde/article/26/2/263/6142509
- **Open copy:** https://digitalcommons.chapman.edu/comm_science_articles/43/
- **Data:** https://osf.io/zpha4/ · project site https://asl-lex.org/
- **Licence:** CC BY-NC 4.0 (compatible with this project's noncommercial use)
- **Local copy:** `data/local/dataset_metadata/asllex2_official/signdata.csv` (2,723 rows)

**Contributes:** the `SignFrequency(M)` ratings for all 2,723 signs — the measurement
surface for our entire argument — plus `Phonological Complexity`,
`Neighborhood Density 2.0`, `SignDuration(ms)`, `LexicalClass` and `SignType.2.0`, which
produced the §3 engineering findings. Covers **25** of our 100 classes' ratings.

**Method:** 1,806 new signs rated (1,735 usable after excluding 14 catch items and 57
later-excluded items); **25–35 deaf participants per sign**; 129 deaf adults across both
rounds.

**Does not support:** anything about running conversation. These are ratings, not counts.

### 1.2 ASL-LEX 1.0 `[V-full]` — origin of 75% of our ratings

> Caselli, N. K., Sevcikova Sehyr, Z., Cohen-Goldberg, A. M., & Emmorey, K. (2017).
> ASL-LEX: A lexical database of American Sign Language. *Behavior Research Methods*,
> 49(2), 784–801.

- **DOI:** https://doi.org/10.3758/s13428-016-0742-0
- **Free full text:** https://slla.lab.uconn.edu/wp-content/uploads/sites/1793/2017/06/Caselli_et_al_2017.pdf

**Contributes:** the frequency ratings for **75** of our 100 classes.

**Method, verified from the paper itself** — this is the sentence to quote if anyone asks
what "frequent" means here:

> "Each video clip was presented individually with the rating scale below the clip and
> participants rated the video on a 7-point scale based on **how often they felt the sign
> appears in everyday conversation** (1 = very infrequently, 7 = very frequently)."

- **69 deaf adults** total (45 female, mean age 34); **25–31 deaf signers rated each sign**.
- 22 further participants excluded for incomplete surveys, degenerate scale use (SD ≤ 1),
  or ASL acquisition after age 6.
- Native vs early signers correlated **r = .94**, means did not differ (Kruskal-Wallis
  χ²(1,69) = .80, p = .37) — ratings are stable across acquisition background.
- "Do not know the sign" was checked for only **1.5%** of responses.

**Why this matters for us:** the rating prompt literally asks about *everyday
conversation*. Our claim is "commonly used in conversation." The instrument and the claim
are the same construct — that is the tightest part of the justification.

**Verified against our own data:** our 100 classes were rated by a mean of **27.4** signers
each (min 23, max 35), of whom a mean of **15.3** were native signers.

---

## 2. Independent ASL corroboration

### 2.1 Mayberry et al. (2014) `[V-full]` — second, independent ASL ratings study

> Mayberry, R. I., Hall, M. L., & Zvaigzne, M. (2014). Subjective frequency ratings for
> 432 ASL signs. *Behavior Research Methods*, 46(2), 526–539.

- **Free full text:** https://pmc.ncbi.nlm.nih.gov/articles/PMC3923849/

**Contributes:** an independent replication that ASL frequency ratings are stable. **59
deaf signers**, 432 signs, 7-point scale; ratings did not differ by age of ASL exposure
(native / early / late). Of the highest-rated signs it names, **6 of 9 are in our 100**
(TIME, HOME, WORK, UNDERSTAND, HAPPY, EAT).

**Caveat:** pre-2020, and ratings again — not counts. Caselli et al. (2017) explicitly
replicate its native-vs-early finding, so the two are not fully independent in method.

### 2.2 Morford & MacFarlane (2003) `[V-2nd]` — the only ASL token-count study

> Morford, J. P., & MacFarlane, J. (2003). Frequency characteristics of American Sign
> Language. *Sign Language Studies*, 3(2), 213–225.

- **Publisher:** https://muse.jhu.edu/article/37891 (paywalled; not read directly)

**Contributes:** the only actual counted ASL corpus — **4,111 sign tokens from 27 deaf
signers**. Core lexicon 73.2% of tokens, pointing signs 13.8%, rising to 17.3% in casual
conversation. Non-first-person pronoun was the most frequent sign.

**Caveat:** 2003, and **4,111 tokens is roughly 1/15th of the Auslan corpus** — too small
to yield a stable top-100 list. Caselli et al. (2017) describe it and Mayberry et al.
(2014) as the only two pre-existing ASL lexical resources, which is why ASL-LEX exists.

---

## 3. Dataset provenance

### 3.1 ASL Citizen `[V-meta]`
> Desai, A., Berger, L., Minakov, F., et al. (2023). ASL Citizen: A community-sourced
> dataset for advancing isolated sign language recognition. *NeurIPS Datasets & Benchmarks.*
- https://arxiv.org/abs/2304.05934 · https://www.microsoft.com/en-us/research/project/asl-citizen/

83,399 videos, 2,731 signs, 52 signers. **Its vocabulary is drawn from ASL-LEX**, which is
what makes the `citizen_asl_lex_code` join in our manifest exact.

### 3.2 ASLLRP `[V-meta]`
> Neidle, C., & Ballard, C. (2022). ASL video corpora & Sign Bank: Resources available
> through the American Sign Language Linguistic Research Project.
- https://arxiv.org/abs/2201.07899

~6,000 entries, 41,830 lexical sign examples. **The best available route to a token-counted
ASL check** on our 100 — currently unused, listed as the obvious next step.

### 3.3 Atwell, Bragg & Alikhani (2024) `[V-meta]`
> Studying and mitigating biases in sign language understanding models.
- https://arxiv.org/abs/2410.05206

Uses ASL Citizen with ASL-LEX lexical features to study recognition performance
disparities. **Cited for approach only** — we retrieved the abstract, not the results, so
no specific feature→accuracy direction is claimed anywhere in our report.

---

## 4. Cross-linguistic support — vocabulary SIZE only

These justify **100 as the cutoff**. They are not ASL and support no claim about which
signs we chose.

| Source | Verified | Corpus | Key figure |
| --- | --- | --- | --- |
| Johnston (2012), *JDSDE* 17(2), 163–193 — Auslan | `[V-2nd]` | 63,436 tokens | **top 100 signs = 52.8%** of tokens |
| Cormier, Fenlon, Rentelis & Schembri (2011), *LDLT3* — BSL | `[V-full]` | 24,864 tokens, 50 signers, conversation only | **top 100 = 57.2%**; top 10 = 27.9% |
| Fenlon, Schembri, Rentelis, Vinson & Cormier (2014), *Lingua* 143, 187–202 — BSL | `[V-2nd]` | same corpus | journal version; text type shifts distribution |
| McKee & Kennedy (2006), *Sign Language Studies* 6(4), 372–390 — NZSL | `[V-2nd]` | 100,000 tokens | 116 signs ≈ 50%; 665 ≈ 80% |

- Johnston: https://academic.oup.com/jdsde/article/17/2/163/581884 (paywalled)
- Cormier et al. **(open, and the one actually read):**
  https://bslcorpusproject.org/wp-content/uploads/lexical-frequency-in-british-sign-language-ldlt3.pdf
- Fenlon et al.: https://doi.org/10.1016/j.lingua.2014.02.003
- McKee & Kennedy: https://muse.jhu.edu/article/202261 (paywalled)

**Honesty note:** the Johnston and McKee & Kennedy figures above were taken from the
open-access Cormier et al. paper, which reports them directly. We have not read those two
primary texts. If either figure is load-bearing in a submission, get the originals.

**One finding from this group that argues *against* us:** Cormier et al. and Johnston both
report that subjective familiarity ratings correlate poorly with counted corpus frequency
— of the 100 BSL signs rated most familiar, only 8 appear in the BSL corpus top 100. Our
entire ASL argument rests on ratings. This is the strongest available objection and is
recorded as limitation 1 in the justification document.

---

## 5. Reproducing our own numbers

`scripts/audit_citizen100_asllex_frequency.py` joins
`active/v17/citizen100_manifest.json` to the ASL-LEX 2.0 CSV and regenerates every
statistic in §2 of the justification document.

```
venv/bin/python scripts/audit_citizen100_asllex_frequency.py
```
