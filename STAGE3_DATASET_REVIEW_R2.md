# Post-Update Review — `src/generate_stage3_data.py`

All previous fixes verified. New issues introduced by the rewrite.

---

## Regressions Introduced

### 1. `gen_no_time_svo` — Semantic violations via collapsed object selection (Critical)
Line 240:
```python
o = random.choice(buyable) if cat == "buy_sell" else random.choice(things + tech + places)
```
All non-`buy_sell` categories now pull from the same pool, breaking every semantic constraint:
- `cat="eat"` → `"I EAT SCHOOL"` → `"I am eating the school."` ❌
- `cat="drink"` → `"HE DRINK CAMERA"` → `"He is drinking the camera."` ❌
- `cat="help"` → `"I HELP BOOK"` → `"I am helping the book."` ❌
- `cat="visit"` → `"SHE WALK PHONE"` → `"She is walking the phone."` ❌

The previous version had per-category object selection. This was removed and it was the load-bearing part of the semantic correctness.

### 2. `gen_no_time_svo` — Missing prepositions for visit verbs (Critical)
`obj_article(o)` returns `"the hospital"`. Visit verbs require `"to the hospital"`.
Result: `"I GO HOSPITAL"` → `"I am going the hospital."` ❌

### 3. `gen_sov` — `.lower()` corrupts `"I"` and sentence casing (Critical)
Line 258:
```python
english = f"{format_time(t)}, " + english.lower()
```
`english` was built with `s_eng.capitalize()`. When `s_eng = "I"`, `.lower()` converts it:
`"Yesterday, i am reading the book."` ❌
Lowercase `"i"` is always wrong. Also lowercases all other subjects.

### 4. `build_safety_net` removed — `all_words` is now dead code (Significant)
`all_words` is still extracted from the `.npy` folder (lines 13–14) but is never passed to any function. `build_safety_net` no longer exists and nothing uses `all_words`. The guarantee that every ASL class from the dataset folder appears at least twice is now broken.

### 5. `gen_compound` — `"GET"` past tense → `"geted"` (Significant)
`v2 = random.choice(["BUY", "SEE", "MEET", "FIND", "GET"])`. `"GET"` is not in `past_tense` dict, doesn't end in E or Y → fallback: `"geted"` ❌ (correct: `"got"`).
**Fix:** Add `"GET": "got"` to `past_tense` dict.

---

## Removed Templates (Untracked Regressions)

| Removed | Impact |
|---------|--------|
| `gen_adjective` | `adj_size`, `adj_quality`, `adj_people` are now defined but **never used anywhere** — pure dead code |
| `gen_mental` | Standalone opinion sentences (`"I think the idea is good."`) entirely absent as a template type |
| `gen_imperative` | Only coverage is 19 idiom phrases; full imperative diversity from previous version is gone |

---

## Previously Flagged Issue Still Not Fixed

**Idiom injection still at 500×** (line 351: `for _ in range(500)`).
Generates 9,500 rows, caps to 950. The recommended fix was to set it directly to `for _ in range(50)`.

---

## Summary

| # | Severity | Issue |
|---|----------|-------|
| 1 | 🔴 Critical | `gen_no_time_svo` collapsed object selection → semantic violations on eat/drink/visit/help |
| 2 | 🔴 Critical | `gen_no_time_svo` missing `"to"` preposition for visit verbs |
| 3 | 🔴 Critical | `gen_sov` `.lower()` produces `"i"` (lowercase) in English output |
| 4 | 🟠 Major | `build_safety_net` removed — `all_words` unused, class coverage not guaranteed |
| 5 | 🟠 Major | `"GET"` in `gen_compound` past tense → `"geted"` |
| 6 | 🟡 Minor | `adj_size/quality/people` defined but never used (dead code) |
| 7 | 🟡 Minor | `gen_mental`, `gen_imperative` silently dropped from generators |
| 8 | 🟡 Minor | Idiom 500× injection still not reduced to 50× |
