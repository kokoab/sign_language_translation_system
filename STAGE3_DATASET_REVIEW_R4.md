# Review R4 — `src/generate_stage3_data.py`

---

## R3 Fixes — Confirmed ✅

| R3 Issue | Status |
|----------|--------|
| `GET` → `"geting"` | ✅ Added to `double_final` (line 115) |
| `gen_imperative` dead code | ✅ Restored in generators with weight 10 |
| `gen_negation` EAT + non-food | ✅ EAT restricted to consumables (line 190) |
| f-string Python 3.12 syntax | ✅ Extracted to variable (line 292) |
| Y/N gloss collisions | ✅ `YOU-KNOW` suffix on all yn templates + safety net + idiom |

---

## Still Open from R3

### 1. `ARRIVE` and `LEAVE` prepositions — unfixed (Bug)
`gen_time_svo` line 151 and `gen_no_time_svo` line 174 still include both verbs in the `"to "` preposition list.
- `ARRIVE` + `"to"` → `"She arrived to the hospital."` ❌ — requires `"at"`.
- `LEAVE` + `"to"` → `"He left to the school."` ❌ — `LEAVE` is transitive (no preposition): `"He left the school."` ✓

**Fix:**
- Remove `LEAVE` from the preposition list entirely.
- Replace `ARRIVE` in the `"to "` list with a separate check: `"at "` for `ARRIVE` only.

---

## New Finding

### 2. Object pronouns use nominative case instead of accusative (Bug)
`gen_time_svo` line 154 and `gen_no_time_svo` line 177 use `subject_english(o)[0]` when the pronoun is an **object**, not a subject:

| Gloss | Current Output ❌ | Correct ✅ |
|-------|-------------------|-----------|
| `HE HELP I` | `"he is helping I."` | `"he is helping me."` |
| `I HELP HE` | `"I am helping he."` | `"I am helping him."` |
| `I HELP SHE` | `"I am helping she."` | `"I am helping her."` |
| `I HELP THEY` | `"I am helping they."` | `"I am helping them."` |
| `I HELP WE` | `"I am helping we."` | `"I am helping us."` |
| `I HELP YOU` | `"I am helping you."` | `"I am helping you."` ✓ (lucky) |

**Fix:** Add an object-case helper:
```python
def object_pronoun(s):
    return {"I": "me", "HE": "him", "SHE": "her", "WE": "us", "THEY": "them"}.get(s, s.lower())
```
Then use `object_pronoun(o)` instead of `subject_english(o)[0]` when the pronoun is in the object position (lines 154 and 177).

---

## Summary

| # | Severity | Issue |
|---|----------|-------|
| 1 | 🟠 Bug | `ARRIVE` needs `"at"`, `LEAVE` needs no preposition |
| 2 | 🟠 Bug | Object pronouns: `"helping I/he/she/they/we"` instead of `"me/him/her/them/us"` |

Everything else is clean. Fix these two and you're good to train.
