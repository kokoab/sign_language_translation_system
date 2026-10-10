# Shared meanings and semantic fidelity: full-vocabulary experiment

Date: 2026-10-11. Scope: the installed phone's Stage 3 model and its input/data contracts. **No production code, model, dataset, or installed app was changed.**

## Result

The concern extends well beyond EAT/FOOD and I/ME. There are four separate issues:

1. **Missing permitted meanings in the data contract.** EAT permits eating/meal but rejects food; the current fine-tuning corpus contains zero targets with the word food. This is consistent with the model failing noun-food probes such as `MY EAT WHERE` → “Where is my eat?”
2. **Meaning checks that inspect words without their relationships.** All 120 deliberately corrupted role, ownership, tense, and negation-scope controls pass the existing evaluator. The corresponding faithful controls also pass. This proves a structural blind spot, not 100% failure on natural sentences.
3. **Information that is absent from Stage 3 input.** It sees glosses and confidence buckets, without pointing loci, verb direction, noun/verb motion, nonmanual question marking, or referent gender/context. Some interpretations cannot be uniquely recovered from that representation.
4. **Generation and incremental-context failures.** Short inputs can acquire invented participants; clause time can be misbound. Incremental locking discards earlier gloss context, changing later pronoun/reference interpretation. Self-consistent sentence generation does not guarantee faithful meaning.

I/ME does not need a new competing visual class: Citizen raw gloss `ME` is already mapped to canonical `I`. EAT is pinned to Citizen `EAT1`; this text experiment does **not** establish recognition coverage of every FOOD variant.

## What ran

The native harness compiles the actual production `LiveReelStage3.swift` and `LiveIncrementalTranslator` with the other LiveReel sources. It loads the same Core ML source packages and token table as the phone: `stage3_multisentence_tiny_v17_20260929`, counted sentence output, phrase overrides disabled. Every model-package file was hash-compared with the phone project resources. The current training corpus hash matches its recorded training recipe. Source hashes are in `identity.json`.

**1,188 native cases in total:**

| Suite | Cases | Purpose |
|---|---:|---|
| Every class in isolation, a short context, and a greeting/thanks wrapper | 300 | All 100 labels, not selected problematic classes |
| Explicit subject/object roles and GIVE recipients | 120 | Pronoun forms and participant preservation |
| Ordinary and postposed possession | 54 | MY/YOUR/OUR ownership |
| Single-clause time and two-clause time combinations | 117 | Time realization and attachment |
| Negation and two-clause negation scope | 27 | Preserve negative meaning and clause |
| Shared meanings and noun/verb frames | 86 | FOOD, DRINK, WORK, SIGN, HELP, ANSWER, LOVE and related readings |
| Confidence variants | 30 | Same words with .5/.2 confidence |
| Questions | 30 | Six question types across five pronouns |
| Topic-order variants | 40 | Participant preservation under alternative order; no unique ASL role asserted |
| Incomplete/reference-ambiguous inputs and multi-sentence examples | 35 | Avoid treating insufficient evidence as a unique answer |
| Saved real phone Finish inputs | 26 | Existing user recognition outputs, not newly recorded signing |
| Historical full/incremental regression controls | 323 | Exercise actual locking and compare saved outputs |

The first 865 cases contain **772 distinct gloss sequences**; 213 case rows have an exact sequence in the current training corpus. Confidence variants and repeated diagnostic conditions intentionally share inputs. These are **diagnostics, not independent held-out accuracy**. No protected Citizen/phrase visual test was accessed. No API generation or paid judge was used. No visual inference, new training, or deployment occurred.

The corpus scan covers all **52,961 training rows**. Counts are row-level English-word co-occurrences, not sign-to-word alignments. `vocabulary_audit.csv` has one row for every class, including class index, exact raw gloss/ASL-LEX code, permitted lemmas, exposure, all three outputs, and a policy/risk note. `results.csv` contains all 865 main cases; historical locking results are separate JSONL.

## Measured behavior

### EAT/FOOD is a real coverage gap; other noun/verb readings already work

| Input | Actual installed-model output | Interpretation |
|---|---|---|
| MY EAT WHERE | Where is my eat? | Malformed noun-food rendering |
| EAT HOT | Is it hot to eat? | Does not produce a food noun |
| EAT GOOD | Eat well. | Valid verb reading; cannot call it wrong without intended context |
| I WANT EAT | I want to eat. | Valid; “I want food” is another possible intended reading |
| I HAVE EAT | I have to eat. | Commits to obligation; not a safe universal interpretation of HAVE/EAT |
| I MAKE EAT | I make eat. | Malformed output rather than food preparation |
| DRINK COLD | The drink is cold. | Noun reading works |
| MY DRINK WHERE | Where is my drink? | Noun reading works |
| I LIKE DRINK | I like to drink. | Verb reading works |
| MY WORK IMPORTANT | My work is important. | Nominal work works |
| I NEED HELP | I need help. | Nominal help works |
| YOUR ANSWER DIFFERENT | Your answer is different. | Nominal answer works |
| I NEED ANSWER | I need to answer. | Chooses verb; noun reading is not offered |

In the current fine-tuning corpus: food 0, meal 33, job 0, winter 0, beverage 0, assistance 0 rows. That is not proof the pretrained model has never seen these words. It shows which realizations this training recipe admits/teaches. Zero job targets does not mean WORK fails: “work” is often already a faithful rendering. Expansion must be meaning-driven, not a demand to output every English synonym.

An in-memory whitelist ablation admits FOOD/JOB/WINTER examples previously rejected. It leaves swapped roles, invented recipients, and stronger negation accepted. **Changing aliases alone fixes admission, not the trained model or semantic validation.** No production map was edited.

### Pronouns often work in explicit contexts; missing roles and possession remain distinct

- **78/80** specified subject–verb–object probes realize the intended subject and direct-object forms. The two failures are `I LIKE WE` → “I like it with us” and `WE LIKE I` → “We like it.” These unusual inclusive-person combinations are useful stress cases, not representative accuracy estimates.
- **40/40** GIVE probes retain the declared giver, recipient and water, in both recipient-first and object-first orders. Modal/question choice was not scored by this narrow check.
- **36/36** ordinary possessive probes retain the named owner.
- In the 15 postposed possession probes excluding EAT, **2 lose or change ownership**: `CHILD MY` → “That child is with me”; `SCHOOL YOUR` → “Is it at school?” Other cases often preserve the owner but invent “here.”
- The 40 candidate topic-order inputs preserve the intended **set of participants in 31/40**. This only checks who remains, not which role is correct. Example: `WE I SEE` → “I see him” drops WE and adds a third person, regardless of the ambiguous I/WE role assignment.
- In isolation, `I` → “Is it here?” and `TALK` → “Ask.” These are clear lexical losses. A first draft of the audit also flagged DIFFERENT because the base lemma helper rejects “different”; the evaluator-specific map already patches that word. The final lexical-loss count excludes this checker false positive.
- Fragments such as `I GIVE` → “I give it to him,” `HE` → “He is here,” and `HELP` → “Help me” receive unsupported completions. Some might happen to match a user's intention, but that intention is absent from the transmitted input.

All 30 explicit wh-question probes retain the displayed question word and subject in the inspected outputs. `YOU HAPPY` becoming a question cannot be classified from glosses alone: the missing facial/nonmanual information could distinguish it from a statement.

HE is a gender-biased English label for a reference sign. The current data has 8,276 rows containing “he” versus 416 “she” rows; all HE reference examples in this sweep choose masculine wording. This documents a model/input bias, not evidence that pointing encodes male gender. `MY MOTHER SICK HE NEED HELP` yields “My mother is sick, and he needs help.” Without spatial co-reference, that could describe a different person; it cannot be assumed to refer to the mother.

### Scope errors persist even when all the right words survive

The 14 isolated YESTERDAY/TOMORROW probes render their expected past/future associations. In the deliberately paired 32 past/future two-clause probes, **8 fail**:

- `YESTERDAY I WORK TOMORROW I SLEEP` → “Yesterday I worked and I slept tomorrow.”
- `YESTERDAY I WORK TOMORROW I EAT` → “Yesterday I worked and I ate.” (TOMORROW disappears.)
- `YESTERDAY I SLEEP TOMORROW I WORK` → “Yesterday I slept tomorrow, and I will work.”

Failures are 8/16 in past-then-future order and 0/16 in reverse order. Action words and time markers alone do not establish correct clause attachment. The remaining NOW combinations are preserved for inspection; no strict tense accuracy is asserted for ambiguous “now I will…” phrasing.

All 15 simple negation probes retain negation, and the inspected 12 two-clause negation probes place it with the intended predicate. This does not validate the checker: its separate six deliberately negation-swapped controls all pass.

Greeting/thanks wrappers also change meaning in some classes: `HELLO YOU TELL I THANKYOU` → “Hello, can you tell me to thank you?” and `HELLO I READY THANKYOU` → “Hello, I am ready to thank you.” The sign inventory survives while the relationship between acts changes. Several wrappers drop THANKYOU outright. The complete per-class table exposes these cases.

### Confidence is not ambiguity awareness

The 30 .5/.2 probes overwhelmingly retain the same preferred interpretation rather than indicating uncertainty. At .2, `I WANT EAT EAT GOOD` changes from “I want to eat and eat well” to “I want to eat well.” The .2 cases are injected stress inputs below the recognizer's .4 commitment threshold; they are not evidence that the live recognizer normally commits such words. Lowering confidence is not a demonstrated solution to shared meaning.

### Incremental rendering needs context preservation

The main 765 incremental cases match their corresponding full-buffer outputs, but none activate a lock. That result alone says little about locking safety.

Therefore the experiment reruns **all 323 existing native parity controls**. Full text, incremental text and lock counts match their saved expectations **323/323**. There are **180 cases with actual locks** and **57 full-versus-incremental text differences**. Differences include harmless phrasing and harmful changes; 57 is not an error count.

Concrete harmful differences:

- `locking_gen034`: full-context “Maybe we can go see **them** tomorrow”; incremental “Maybe we can go see **her** tomorrow.” The two locked sentences contained YOUR FAMILY and THEY; the remaining buffer contains only MAYBE WE GO SEE TOMORROW. The referent context is gone.
- `locking_gen225`: full text retains “the doctor”; the incremental tail drops DOCTOR after the hospital sentence is locked.
- `locking_gen275`: full text preserves “I did not sleep yesterday.” Incremental locking first commits “I am sorry for yesterday,” and the final output loses SLEEP and moves the negation.
- `locking_gen280`: NOW moves from drinking water to being happy after locking.

This reproduces existing behavior. It is not a new regression from the HAVE visual decoder fix. Source and model hashes remained unchanged. A valid count and agreement across two renders do not prove that a sentence expresses the source meaning correctly.

## Why the old evaluation misses this

The evaluator unions permitted English forms for the whole gloss list and checks word coverage. It does not bind subjects/objects, owners, tense, or negation to clauses. Its globally permitted function-word list includes me/him/her/them/us; the base aliases also conflate I with my/mine, YOU with your/yours, WE with our, and HE with his. A lexical list is therefore both too narrow for some valid senses and too permissive for relationships.

| Deliberate corruption | Faithful controls accepted | Corrupted controls accepted |
|---|---:|---:|
| Subject/object swapped | 80/80 | 80/80 |
| Owners exchanged between nouns | 18/18 | 18/18 |
| Past/future associations swapped | 16/16 | 16/16 |
| Negation moved to other clause | 6/6 | 6/6 |

These are offline corpus/evaluation checks, **not runtime semantic safeguards**. Runtime generation currently falls back for technical/empty/length failures; it does not reject the semantic corruptions above. Additional 59 lexical/role controls and four identical-input/different-meaning demonstrations are saved. The latter establish a representation limit, not measured video recognition accuracy.

## How to approach the fix

1. **Version the semantic contract before modifying training.** Preserve the 100 class IDs, raw Citizen identities and existing datasets. For every class, separate allowed English forms, context-dependent senses, unsupported homonyms, and evidence needed. Preserve personal versus possessive pronouns. Record FOOD under the reviewed EAT sign family without pretending every FOOD variant is visually supported.
2. **Replace bag-of-words-only acceptance for scored cases.** Use explicit participant/owner/clause/negation/time expectations in the diagnostic data. Keep word coverage as one check, with counterfactual corruption tests. Ambiguous cases need multiple accepted readings or an explicit unresolved result, not an arbitrary single gold sentence.
3. **Repair the demonstrated text-generation gaps with a bounded recipe.** Add valid noun-food and other reviewed readings, postposed possession and varied clause orders, while retaining role/negation/time contrasts and existing working examples. Reserve new combinations before training. No larger model is justified solely by this audit; first remove the broken admission/scoring assumptions.
4. **Preserve relevant context during incremental translation.** Compare a full-buffer-at-Finish control with context-preserving tail generation. Require semantic fidelity of locked spans, not merely repeated text/counts. Measure phone latency before choosing a deployment change. Current inputs cannot recover missing spatial/nonmanual evidence; do not fake it through phrase-specific rules.
5. **Validate visual coverage separately.** If movement or direction is necessary to distinguish a meaning, replay approved examples of the exact sign forms and decide whether the recognizer or its output contract must retain that evidence. No current result establishes FOOD variant accuracy or an ASL-fluent assessment of the synthetic probes.

## Linguistic boundaries used in this audit

[I/ME](https://www.lifeprint.com/asl101/pages-signs/m/me.htm) share ordinary self-pointing. [Personal versus possessive pronouns](https://www.lifeprint.com/asl101/pages-signs/p/poss-3.htm) involve a handshape distinction; English subject/object case must not erase ownership. [HE/SHE reference](https://www.lifeprint.com/asl101/pages-signs/s/she.htm) does not supply gender by itself. [FOOD](https://www.lifeprint.com/asl101/pages-signs/f/food.htm) may use a single movement in fluent signing; tap count is not an absolute classifier.

[WORK](https://www.lifeprint.com/asl101/pages-signs/w/work.htm) can mean job in context, but English work=operate/succeed is a different issue. [COLD](https://www.lifeprint.com/asl101/pages-signs/c/cold.htm) may mean winter in context, but not an illness/head-cold. [LIKE](https://www.lifeprint.com/asl101/pages-signs/l/like.htm) means affinity/enjoyment, not similar/same. [DRINK](https://www.lifeprint.com/asl101/pages-signs/d/drink.htm) has handshape/manner distinctions that a broad English alias must not erase. [SIGNATURE](https://www.lifeprint.com/asl101/pages-signs/s/signature.htm) is distinct from simply treating every English “sign” sense as one class.

These sources guide candidate policies. They do not certify that the selected Citizen raw form covers every mentioned variant. Other English paraphrases in the exploratory gate controls are candidates for review, not newly approved visual-class equivalences.

## Reproduction and artifacts

- `build_experiment.py`: 555 initial cases, package/source/corpus identity, full-vocabulary and training exposure audit.
- `extend_experiment.py`: 284 targeted cases and additional lexical controls.
- `main.swift`: native full-buffer and production incremental harness; `build.log` records compilation.
- `phone_cases.json`: all26 historical phone inputs; `locking_cases.json`: all323 saved native regression controls.
- `results.jsonl`, `extended_results.jsonl`, `phone_results.jsonl`, `locking_results.jsonl`: exact native outputs and lock traces.
- `analyze.py`: rebuilds main summaries, all-class CSV and 120 counterfactual checks from saved outputs.
- `check_contracts.py`: runnable evaluation-only alias ablation and identical-input collision assertions.
- `verify.py`: final evidence/source invariants, including historical locking parity.
- `summary.json`, `locking_summary.json`, `vocabulary_audit.csv`, `results.csv`: concise machine-readable results.

The native probe command is `probe <CoreML-package-directory> <cases.json> <output.jsonl>`; compilation uses the five production LiveReel Swift sources and `main.swift`, as recorded in `reproduce.sh`. All checks use existing local resources. The first manifest lookup used Models/ rather than Runner/ and was corrected before inference. A summary-script membership test initially included isolated rows without lock data; fixed by selecting actual incremental output strings. Neither setup error affected model outputs. Existing unrelated worktree changes are preserved.
