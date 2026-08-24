# Review R5 — `src/generate_stage3_data.py`

## R4 Fixes — Confirmed ✅

| R4 Issue | Status |
|----------|--------|
| `ARRIVE` preposition `"to"` → `"at"` | ✅ Fixed in all 3 generators |
| `LEAVE` transitive (no preposition) | ✅ Removed from prep lists |
| Object pronouns `"helping I"` → `"helping me"` | ✅ `object_pronoun()` helper added and used |

---

## Fixed Directly

### `MISS` third-person singular → `"misss"` (was Bug, now ✅)
`verb.lower() + "s"` on `"miss"` produced `"misss"`. Applied sibilant rule: verbs ending in `-ss`, `-sh`, `-ch`, `-x` now get `-es`. Result: `"misses"` ✓

---

## Verdict: Clear to train ✅

No remaining grammatical, semantic, or structural bugs found. All conjugation paths, prepositions, object cases, and mass-noun handling are correct across all 13 generators + safety net + idioms.
