# Final Conjugation Audit: `conjugate()` — `generate_stage3_data.py`

Audited all 42 verbs across `verbs_dict` + `FEEL`/`SEE`/`HAVE` used inline.

---

## Verdict: 3 Bugs Found

---

### Bug 1 — `NEED` past tense → `"neded"` ❌

**Root cause:** `NEED` is not in `past_tense` dict, so fallback runs:
```
"need" + "ed" = "needed"
"needed".replace("eed", "ed") → finds "eed" at index 1 → "n" + "ed" + "ed" = "neded"
```
The `.replace("eed","ed")` was designed to fix `"useed"→"used"`, but it also corrupts legitimately-correct words like `"needed"`.

**Fix:** Add to `past_tense` dict: `"NEED": "needed"`

---

### Bug 2 — `STUDY` past tense → `"studyed"` ❌

**Root cause:** Not in `past_tense` dict. Fallback gives `"study"+"ed"` = `"studyed"`. The `y→ied` rule is not implemented anywhere.

**Fix:** Add to `past_tense` dict: `"STUDY": "studied"`

---

### Bug 3 — `PROGRAM` past tense → `"programed"` and present continuous → `"programing"` ❌

**Root cause (past):** Not in `past_tense` dict. Fallback gives `"programed"` (single `m`).
**Root cause (ing):** Not in the `["RUN", "SIT", "WIN", "CUT"]` double-consonant list. Gives `"programing"` (single `m`).

**Fix:**
- Add to `past_tense` dict: `"PROGRAM": "programmed"`
- Add `"PROGRAM"` to the double-consonant set in the `ing` logic

---

## All Other Verbs — Pass ✅

| Verb | Past Tense | Present Continuous |
|------|-----------|-------------------|
| EAT | ate (dict) ✓ | eating ✓ |
| COOK | cooked (fallback) ✓ | cooking ✓ |
| DRINK | drank (dict) ✓ | drinking ✓ |
| DRIVE | drove (dict) ✓ | driving (E-rule) ✓ |
| USE | used (eed-fix) ✓ | using (E-rule) ✓ |
| DOWNLOAD | downloaded ✓ | downloading ✓ |
| UPLOAD | uploaded ✓ | uploading ✓ |
| DELETE | deleted (eed-fix) ✓ | deleting (E-rule) ✓ |
| SAVE | saved (eed-fix) ✓ | saving (E-rule) ✓ |
| FIX | fixed ✓ | fixing ✓ |
| DEVELOP | developed ✓ | developing ✓ |
| CREATE | created (eed-fix) ✓ | creating (E-rule) ✓ |
| VISIT | visited ✓ | visiting ✓ |
| GO | went (dict) ✓ | going ✓ |
| COME | came (dict) ✓ | coming (E-rule) ✓ |
| WALK | walked ✓ | walking ✓ |
| RUN | ran (dict) ✓ | running (double-r) ✓ |
| CLIMB | climbed ✓ | climbing ✓ |
| MOVE | moved (eed-fix) ✓ | moving (E-rule) ✓ |
| LEARN | learned ✓ | learning ✓ |
| READ | read (dict) ✓ | reading ✓ |
| WRITE | wrote (dict) ✓ | writing (E-rule) ✓ |
| KNOW | knew (dict) ✓ | mental path ✓ |
| UNDERSTAND | understood (dict) ✓ | mental path ✓ |
| REMEMBER | remembered ✓ | mental path ✓ |
| FORGET | forgot (dict) ✓ | mental path ✓ |
| THINK | thought (dict) ✓ | mental path ✓ |
| WANT | wanted ✓ | mental path ✓ |
| LOVE | loved (eed-fix) ✓ | mental path ✓ |
| LIKE | liked (eed-fix) ✓ | mental path ✓ |
| BUY | bought (dict) ✓ | buying ✓ |
| SELL | sold (dict) ✓ | selling ✓ |
| PAY | paid (dict) ✓ | paying ✓ |
| GIVE | gave (dict) ✓ | giving (E-rule) ✓ |
| TAKE | took (dict) ✓ | taking (E-rule) ✓ |
| RECEIVE | received (eed-fix) ✓ | receiving (E-rule) ✓ |
| FEEL | handled inline ✓ | handled inline ✓ |
| SEE | saw (dict) ✓ | seeing (exception) ✓ |
| HAVE | handled inline ✓ | handled inline ✓ |

---

## Summary

Add 3 entries to `past_tense` dict and add `"PROGRAM"` to the double-consonant set. Everything else is airtight.
