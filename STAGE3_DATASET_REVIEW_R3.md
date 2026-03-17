# Review R3 — `src/generate_stage3_data.py`

All R2 critical/major issues verified fixed. Remaining items below.

---

## Previous Fixes — Confirmed ✅

| R2 Issue | Status |
|----------|--------|
| `gen_no_time_svo` collapsed objects | ✅ Per-category selection restored |
| `gen_no_time_svo` missing prepositions | ✅ Restored for visit verbs |
| `gen_sov` `.lower()` on `"I"` | ✅ Explicit `"I"` check on line 267 |
| `build_safety_net` removed | ✅ Restored, called at line 470 |
| `GET` past tense `"geted"` | ✅ Added to `past_tense` dict as `"got"` |
| `gen_adjective` removed | ✅ Restored with weight 5 |
| `gen_mental` removed | ✅ Restored with weight 5 |
| `gen_imperative` removed | ✅ Function restored |
| Idiom 500× injection | ✅ Reduced to 50× |

---

## New Issues

### 1. `GET` present continuous → `"geting"` (Bug)
`GET` was added to `buy_sell` (line 60) but **not** to `double_final` (line 114). The -ing path produces:
`"get" + "ing"` = `"geting"` ❌ — should be `"getting"`.
**Fix:** Add `"GET"` to `double_final` list on line 114.

### 2. `gen_imperative` is dead code — not in generators (Significant)
`gen_imperative` is defined (lines 331-354) but **not listed in `generators`** (lines 428-441). The main loop checks `if fn == gen_imperative` (line 457) but this is never true since the function is never in the pool. All imperative coverage comes only from the 50×19 idiom rows.
**Fix:** Add `(gen_imperative, 8)` to the `generators` list.

### 3. `ARRIVE` and `LEAVE` use wrong preposition (Bug)
Both `gen_time_svo` (line 150) and `gen_no_time_svo` (line 173) include `ARRIVE` and `LEAVE` in the `"to "` preposition list:
- `ARRIVE` requires **"at"**, not "to": `"She arrived at the hospital."` ✓ vs `"She arrived to the hospital."` ❌
- `LEAVE` is transitive — **no preposition**: `"He left the school."` ✓ vs `"He left to the school."` ❌

**Fix:** Remove `LEAVE` and `ARRIVE` from the preposition list. Handle `ARRIVE` separately with `"at "`.

### 4. `gen_negation` — `EAT` can pair with non-food objects (Minor)
Line 186-187: verb pool includes `"EAT"` but object pool is `things + tech + places + consumables`. Produces: `"I NOT EAT HOSPITAL"` → `"I do not eat the hospital."` ❌
**Fix:** When `v == "EAT"`, restrict `o` to consumables only.

### 5. `gen_wh_question` f-string syntax — Python 3.12+ only (Minor)
Line 285: `f"{"MY" if art == 'my' else 'THE'} {o} WHERE"` — nested same-quote f-strings only work in Python 3.12+ (PEP 701). Earlier versions raise `SyntaxError`.
**Fix:** Extract to a variable: `gloss_art = "MY" if art == "my" else "THE"` then use `f"{gloss_art} {o} WHERE"`.

### 6. Gloss collisions between question and statement templates (Minor)
`gen_yn_question` "go_q" produces gloss `"{s} GO {place}"` → `"Is he going to the school?"`
`gen_no_time_svo` produces the same gloss → `"He is going to the school."`
After `drop_duplicates(subset=["gloss"])`, one is silently dropped — 50/50 which survives due to the pre-shuffle. This reduces effective Y/N question coverage.
**Fix:** Prefix Y/N glosses with a marker: `f"{s} GO {place} YOU-KNOW"` (already done in `feel_q`, just not in `go_q` or `have_q`).

---

## Summary

| # | Severity | Issue |
|---|----------|-------|
| 1 | 🔴 Bug | `GET` → `"geting"` — add to `double_final` |
| 2 | 🟠 Major | `gen_imperative` not in generators — dead code |
| 3 | 🟠 Major | `ARRIVE`/`LEAVE` wrong prepositions |
| 4 | 🟡 Minor | `gen_negation` EAT + non-food |
| 5 | 🟡 Minor | f-string syntax breaks below Python 3.12 |
| 6 | 🟡 Minor | Gloss collisions between yn_question and no_time_svo |

Overall the script is substantially cleaner than all prior versions. Fixing items 1-3 clears all remaining functional bugs.
