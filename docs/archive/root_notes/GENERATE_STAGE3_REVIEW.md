# Code Review: `src/generate_stage3_data.py`

---

## Critical Issues

### 1. Silent `except` on Folder Read (Line 12)
```python
except:
    all_words = []
```
**Problem:** Swallows ALL exceptions — if the folder doesn't exist, you silently generate 0 safety-net sentences with no warning.
**Fix:** Use `except (FileNotFoundError, PermissionError) as e: print(f"Warning: {e}")` and set `all_words = []` only then.

---

### 2. Semantic Violations Still Present

**`adjective` template (Line 116–119):**
```python
o = random.choice(places + people + things + vehicles + consumables)
adj = random.choice(adjectives)
```
- "LONG" / "SHORT" paired with people → *"the long teacher"* ❌
- "TALL" / "DIRTY" paired with WATER → *"the dirty water"* is ok, but *"the tall water"* is not ❌
- "COLD" / "HOT" are in `feelings`, not `adjectives`, yet are used as feelings only — this is fine, but `adjectives` contains no size distinction.

**Fix:** Split `adjectives` into sub-groups:
```python
size_adj   = ["BIG", "SMALL", "SHORT", "LONG", "HIGH", "LOW"]  # things/places only
person_adj = ["TALL", "STRONG", "WEAK", "OLD", "YOUNG"]        # people only
quality_adj = ["GOOD", "BAD", "EASY", "HARD", "CLEAR", "WRONG", "IMPORTANT"]  # anything
```
Then pick `adj` based on what `o` is.

---

### 3. `number` Template Produces Bad Plurals (Line 127)
```python
return f"..., {s_eng} {v_eng} {num.lower()} {o.lower()}s."
```
- `"one waters"` ❌, `"two babys"` ❌, `"three mans"` ❌

**Fix:** Install `inflect` and replace the naive `+ "s"`:
```python
import inflect
p = inflect.engine()
# At usage site:
count = 1 if num == "ONE" else 2
plural_o = o.lower() if count == 1 else p.plural(o.lower())
```

---

### 4. `conjugate()` Has a Broken Fallback (Line 56)
```python
return past_tense.get(verb, f"{verb.lower()}ed").replace("eed", "ed")
```
- `SIGN` → `signed` ✓
- `USE` → `useed` → `used` ✓ (works by accident via replace)
- `CALL` → `called` ✓
- `COOK` → `cooked` ✓

But double-consonant verbs are missed:
- `RUN` → `runed` ❌ (should be `ran`, already in dict — ok)
- `STOP` → `stoped` ❌ (not in dict, needs `stopped`)

**Fix:** Add a proper doubling rule or expand `past_tense` dict for all verbs in `verbs_dict`.

---

### 5. `ing` Suffix Edge Cases (Lines 70–73)
```python
ing_word = verb.lower() + "ing"
if verb.endswith("E") and verb not in ["SEE", "FREE"]:
    ing_word = verb.lower()[:-1] + "ing"
```
- `DRIVE` → `driving` ✓
- `DANCE` → `dancing` ✓
- `PROGRAM` → `programing` ❌ (should be `programming`)
- `RUN` → `runing` ❌ (should be `running`) — but `RUN` goes through `visit` template, not `ing` path directly

**Fix:** Add a set of consonant-doubling verbs:
```python
double_consonant = {"RUN", "PROGRAM", "STOP", "DROP", "BEG"}
if verb in double_consonant:
    ing_word = verb.lower() + verb[-1].lower() + "ing"
```

---

### 6. Safety Net Sentences Are Semantically Broken (Lines 157–160)
```python
{"gloss": f"MY {word} WHERE", "text": f"Where is my {word.lower()}?"}
{"gloss": f"I LIKE {word}",   "text": f"I like the {word.lower()}."}
{"gloss": f"HE ABOUT {word}", "text": f"He is talking about the {word.lower()}."}
```
- `"Where is my yesterday?"` ❌ (word = TIME gloss)
- `"I like the A."` ❌ (word = LETTER gloss)
- `"He is talking about the drink."` (word = consumable, acceptable but weird)

**Fix:** Filter `all_words` before safety-net generation — exclude words already covered by the main templates (letters, numbers, pronouns, time words):
```python
skip = set([w.upper() for w in letters + numbers + list(times) + list(pronouns)])
content_words = [w for w in all_words if w not in skip]
```

---

### 7. `"HE ABOUT {word}"` Is Not Valid ASL Gloss Syntax (Line 161)
ASL gloss requires a verb — `ABOUT` is not a standalone verb in standard gloss notation.
**Fix:** Use `"HE TALK {word}"` → `"He is talking about the {word.lower()}."` or `"HE DISCUSS {word}"`.

---

## Minor Issues

| Line | Issue | Fix |
|------|-------|-----|
| 85 | `s_eng = ... s.lower()` — "they" stays lowercase but "you" and "she" also need a capital at sentence start (handled by `t.capitalize()` prefix so risk is low) | Low priority |
| 87 | `template` list doesn't include `"drive"` — `CAR`/`BUS` only appear in the `adjective` template | Add a `"drive"` template: `"YESTERDAY I DRIVE CAR"` → `"Yesterday, I drove the car."` |
| 127 | `"ONE book"` → `"one books"` — singular count not handled | Use `inflect` as noted above |
| 149 | Generates 30,000 then deduplicates — actual output count unknown | Print expected vs actual count |
| 162 | `sample(frac=1)` shuffles but doesn't set `random_state` — not reproducible | Add `random_state=42` |

---

## Recommendations Summary

1. **Use `inflect`** for all pluralization (numbers template + safety net).
2. **Split adjective categories** to avoid semantically wrong pairings.
3. **Expand `past_tense` dict** to cover all verbs in `verbs_dict`.
4. **Add consonant-doubling rules** for `ing` suffix.
5. **Filter letters/numbers/times** from safety-net loop.
6. **Fix `"HE ABOUT"`** gloss → `"HE TALK"`.
7. **Add a `drive` template** so `CAR`/`BUS` appear in proper vehicle context.
8. **Fix bare `except`** to catch specific exceptions.
