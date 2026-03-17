# Stage 3 Dataset Generator — Peer Review
#### `src/generate_stage3_data.py` — Full Script Audit

---

## 1. Linguistic Mapping

### Critical — `HAVE` in `gen_topic_comment` produces semantic nonsense
`conjugate("HAVE", "NOW", ...)` hits the E-rule (`HAVE` ends in `E`) → `ing = "having"` → outputs **"As for the book, he is having it."** ❌
`HAVE` is possessive, not progressive. It must be handled outside `conjugate()`, same as it is in `gen_number`.
**Fix:** Hardcode `"has"/"have"` for `HAVE` in `gen_topic_comment` before calling `conjugate()`.

### Significant — `gen_wh_question` uses `p.plural()` on mass nouns
`things` includes `"MONEY"`. `p.plural("money")` = `"moneys"` → outputs **"How many moneys do I have?"** ❌
Mass nouns (`MONEY`, `WATER`, `FOOD`, `COFFEE`, `MEDICINE`, `DATA`) require `"How much"`, not `"How many"`.
**Fix:** Add a `mass_nouns` set and branch to `"How much"` before calling `p.plural()`. Same check already exists in `generate_idiom_variations` (line 466) — apply the same logic to `gen_wh_question`.

### Minor — ASL perspective shift in `gen_wh_question` `what_name`
`"NAME WHAT I"` → `"What is your name?"` — `I` maps to `"your"`. This is a valid ASL-to-English perspective convention but it is the **only** example of a first/second-person perspective shift in the entire dataset. T5 will never generalize this; it will just memorize it. Not harmful, just low-signal.

### Minor — `adj_people` overlaps with `feelings`
`adj_people` contains `"BUSY"`, `"FREE"`, `"STRONG"`, `"WEAK"` which also appear in `feelings`. A person described as `"BUSY"` via `gen_adjective` generates `"Yesterday, the teacher saw the busy student."` which is grammatically fine. However `"STRONG"` and `"WEAK"` in `adj_people` will produce `"He sees the strong man"` — acceptable, but the same words also fire in `gen_feeling` as emotional states. The model sees both and the signal is coherent only if the T5 context window is long enough to disambiguate — for a short gloss this is fine.

---

## 2. Data Distribution & Deduplication

### Critical — Negation is almost entirely absent
The entire 50,000-row dataset contains **only 2 negation examples**: `"I NOT KNOW"` and `"I NOT UNDERSTAND"` (from `asl_idioms`). No template generates `NOT`-bearing sentences.
Real Stage 2 CTC output will regularly produce glosses like `"I NOT GO STORE"`, `"HE NOT UNDERSTAND"`, `"YOU NOT NEED MONEY"`. The T5 model has no training signal for these → **guaranteed hallucination on negation in production**.
**Fix:** Add a `gen_negation()` template that inserts `NOT` after the subject: `"I NOT {verb} {object}"` → `"I do not {v_eng} {o_eng}."` Covers at minimum: `NOT KNOW`, `NOT UNDERSTAND`, `NOT WANT`, `NOT HAVE`, `NOT GO`, `NOT LIKE`.

### Significant — `gen_time_svo` dominates at 27.5% of the pool
With weight=28 out of 102 total, nearly 1 in 3 generated rows is a TIME SVO sentence. T5 will be heavily biased toward the pattern **"Time, Subject verb object."** and may default to it under uncertainty.
`gen_no_time_svo` at 16 (15.7%) is the second largest, meaning ~43% of training data is SVO. The model will underfit on WH-questions (8%), Y/N questions (5%), and topic-comment (2%).
**Fix:** Cap `gen_time_svo` at 18-20 and increase `gen_wh_question` to 12, `gen_yn_question` to 8, `gen_topic_comment` to 5.

### Significant — Idiom injection is wasteful before capping
`generate_idiom_variations()` injects 9,500 rows (500×19 idioms) then `groupby().head(50)` caps to 950. The 8,550 rows are generated and gc'd for nothing.
**Fix:** Replace `for _ in range(500)` with `for _ in range(50)` — inject 50×19=950 rows directly. No cap needed.

### Minor — `try/except Exception: pass` silently drops generation errors
If any generator raises (e.g., `random.choice([])` on an empty list), the row is silently skipped. Final dataset could be materially below `TARGET=50000` with no warning.
**Fix:** Collect failures in a counter and `print(f"⚠️ {failures} rows skipped due to errors.")` at the end.

---

## 3. OOD Resilience

### Critical — No compound or multi-clause glosses
All templates produce single-clause glosses (`TIME SUBJ VERB OBJ`). Real Stage 2 CTC output on continuous signing will produce compound glosses like:
- `"I GO STORE BUY FOOD"`
- `"YESTERDAY I EAT AFTER I GO SCHOOL"`
- `"HE SICK NEED DOCTOR"`

The T5 model has zero training examples of these patterns → nearly certain hallucination on multi-verb glosses in production.
**Fix:** Add a `gen_compound()` template that chains two SVO clauses: `"{T} {S} {V1} {O1} THEN {V2} {O2}"` → `"{time}, {s} {v1_eng} {o1_eng} and then {v2_eng} {o2_eng}."` Even 3-5% of training data in this pattern would significantly improve robustness.

### Significant — `gen_sov` only covers present tense
`gen_sov` calls `conjugate(v, "NOW", ...)` hardcoded. ASL topicalization occurs in all tenses. A real CTC output like `"STORE YESTERDAY I GO"` has no training analog.
**Fix:** Pass a randomly sampled `t = random.choice(times)` to `gen_sov` and include it in the gloss prefix.

### Minor — Only 8 WH-question types, no "WHEN" or "WHY reason"
`gen_wh_question` covers: `what_name`, `where_place`, `who_person`, `what_want`, `why_feel`, `how_many`. Missing:
- `"WHEN {S} {V}"` → `"When does {s} {v}?"`
- `"HOW {S} {V}"` → `"How does {s} {v}?"`
These are common in real ASL interaction and the Stage 2 model will produce these glosses.

---

## 4. Code Optimization

### Minor — `pool` list is rebuilt every run
```python
pool = []
for fn, weight in generators:
    pool.extend([fn] * weight)
```
This creates a 102-element list of function references — trivially fast. No issue at 50k iterations.

### Minor — Multiple sequential `pd.concat` calls
Lines 530–538 do two `pd.concat` + two `reset_index` calls on frames of ~50k rows each. At this scale (< 1 MB) the cost is negligible (< 1s). Not a bottleneck.

### Minor — `sample(frac=1, random_state=42)` shuffles the full frame after all concats
This is correct and efficient. No issue.

### Minor — `drop_duplicates(subset=["gloss"])` keeps first occurrence
Order-dependent: whichever generator produced the gloss first wins. Since `gen_time_svo` dominates, it will win most collisions. `gen_topic_comment` and `gen_sov` glosses that collide with `gen_time_svo` glosses will be silently discarded, further reducing the effective diversity of complex templates. This compounds the distribution imbalance noted above.

---

## Summary Priority Table

| # | Severity | Area | Issue |
|---|----------|------|-------|
| 1 | 🔴 Critical | Linguistic | `HAVE` in `gen_topic_comment` → "is having it" |
| 2 | 🔴 Critical | OOD | Zero negation templates → hallucination on `NOT` glosses |
| 3 | 🔴 Critical | OOD | No compound/multi-verb glosses → failure on continuous signing output |
| 4 | 🟠 Major | Linguistic | Mass nouns in `gen_wh_question` → "moneys" |
| 5 | 🟠 Major | Distribution | `gen_time_svo` at 27.5% → SVO overfitting |
| 6 | 🟠 Major | OOD | `gen_sov` hardcoded to NOW → no topicalized past/future |
| 7 | 🟡 Minor | Distribution | Idiom 500x injection before cap(50) — wasteful |
| 8 | 🟡 Minor | Robustness | Silent `except pass` hides generation failures |
| 9 | 🟡 Minor | OOD | No WHEN/HOW question templates |
