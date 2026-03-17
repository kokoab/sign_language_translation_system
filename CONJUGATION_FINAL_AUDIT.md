# Final Conjugation Audit v2 — `conjugate()` rewrite

All 42 verbs traced through Past, Present Continuous, and Future tenses.

---

## Verdict: PASS ✅ — Safe to proceed to training

---

## Full Trace

| Verb | Past | Present Continuous | Future | Path Used |
|------|------|--------------------|--------|-----------|
| EAT | ate ✓ | is eating ✓ | will eat ✓ | dict |
| COOK | cooked ✓ | is cooking ✓ | will cook ✓ | fallback |
| DRINK | drank ✓ | is drinking ✓ | will drink ✓ | dict |
| DRIVE | drove ✓ | is driving ✓ | will drive ✓ | dict / E-rule |
| USE | used ✓ | is using ✓ | will use ✓ | E-rule (past+ing) |
| DOWNLOAD | downloaded ✓ | is downloading ✓ | will download ✓ | fallback |
| UPLOAD | uploaded ✓ | is uploading ✓ | will upload ✓ | fallback |
| DELETE | deleted ✓ | is deleting ✓ | will delete ✓ | E-rule (past+ing) |
| SAVE | saved ✓ | is saving ✓ | will save ✓ | E-rule (past+ing) |
| PROGRAM | programmed ✓ | is programming ✓ | will program ✓ | dict / double-m |
| FIX | fixed ✓ | is fixing ✓ | will fix ✓ | fallback |
| DEVELOP | developed ✓ | is developing ✓ | will develop ✓ | fallback |
| CREATE | created ✓ | is creating ✓ | will create ✓ | E-rule (past+ing) |
| VISIT | visited ✓ | is visiting ✓ | will visit ✓ | fallback |
| GO | went ✓ | is going ✓ | will go ✓ | dict |
| COME | came ✓ | is coming ✓ | will come ✓ | dict / E-rule |
| WALK | walked ✓ | is walking ✓ | will walk ✓ | fallback |
| RUN | ran ✓ | is running ✓ | will run ✓ | dict / double-n |
| CLIMB | climbed ✓ | is climbing ✓ | will climb ✓ | fallback |
| MOVE | moved ✓ | is moving ✓ | will move ✓ | E-rule (past+ing) |
| STUDY | studied ✓ | is studying ✓ | will study ✓ | dict / fallback |
| LEARN | learned ✓ | is learning ✓ | will learn ✓ | fallback |
| READ | read ✓ | is reading ✓ | will read ✓ | dict |
| WRITE | wrote ✓ | is writing ✓ | will write ✓ | dict / E-rule |
| KNOW | knew ✓ | knows / know ✓ | will know ✓ | dict / mental |
| UNDERSTAND | understood ✓ | understands / understand ✓ | will understand ✓ | dict / mental |
| REMEMBER | remembered ✓ | remembers / remember ✓ | will remember ✓ | fallback / mental |
| FORGET | forgot ✓ | forgets / forget ✓ | will forget ✓ | dict / mental |
| THINK | thought ✓ | thinks / think ✓ | will think ✓ | dict / mental |
| WANT | wanted ✓ | wants / want ✓ | will want ✓ | fallback / mental |
| NEED | needed ✓ | needs / need ✓ | will need ✓ | dict / mental |
| LOVE | loved ✓ | loves / love ✓ | will love ✓ | E-rule / mental |
| LIKE | liked ✓ | likes / like ✓ | will like ✓ | E-rule / mental |
| BUY | bought ✓ | is buying ✓ | will buy ✓ | dict |
| SELL | sold ✓ | is selling ✓ | will sell ✓ | dict |
| PAY | paid ✓ | is paying ✓ | will pay ✓ | dict |
| GIVE | gave ✓ | is giving ✓ | will give ✓ | dict / E-rule |
| TAKE | took ✓ | is taking ✓ | will take ✓ | dict / E-rule |
| RECEIVE | received ✓ | is receiving ✓ | will receive ✓ | E-rule (past+ing) |
| SEE (inline) | saw ✓ | is seeing ✓ | will see ✓ | dict / exception |
| FEEL (inline) | felt ✓ | is feeling ✓ | will feel ✓ | hardcoded |
| HAVE (inline) | had ✓ | has / have ✓ | will have ✓ | hardcoded |

---

## One Non-Critical Note

Time words `"MORNING"`, `"AFTERNOON"`, `"NIGHT"` fall into the `else` (present continuous) branch, producing sentences like:

> `"Morning, the teacher is studying the book."`

This is **grammatically valid** but slightly unnatural — `"In the morning"` is more idiomatic. Not a conjugation bug, but worth knowing for T5 training data quality. Low priority.

---

## Summary

The rewrite is clean. The removal of `.replace("eed","ed")` eliminated all prior corruption risks. All three previously broken verbs (`NEED`, `STUDY`, `PROGRAM`) are now explicitly handled in the dict. The rule-based fallback chain (dict → E-rule → Y-rule → plain `+ed`) is logically sound for all remaining verbs.
