# Phone translation history review — 2026-09-29

Latest phone session: `20260929_004215`, starting 00:42:15 PHT. Latest copied autosave covers 448.57 seconds and is marked incomplete. Six translations, seven two-hand Finish events (one empty), 26 resets, 70 word events and 337 previews. The six translations were unchanged across two snapshots.

| Recognized input | Saved English |
|---|---|
| HELLO MY NAME fs-GEC | Hello, my name is Gec. |
| HOW YOU HOW YOUR DAY | How. You. How is your day? |
| HELLO HOW MORNING GOOD DAY | Hello, how is he? In the morning, Friend. Good. |
| I HELLO FRIEND I NEED HELP | I. Hello, I am a friend. Need. Help. |
| HELLO FRIEND I HELP | Hello, I am a friend. Help. |
| USE FEEL HAPPY PLEASE SORRY HUNGRY MY NAME fs-GELO | Use. Feel happy. Please. Sorry. hungry is Gelo. |

## Findings

1. **Translation adds unsupported content.** HELLO HOW / MORNING / GOOD DAY becomes “Hello, how is he? In the morning, Friend. Good.” HE and FRIEND were absent from the recognized input; DAY is lost. This defect exists downstream of recognition.
2. **Hard pause splitting fragments phrases.** `liveSplitClauses` splits at gaps >=1.5s. The 2.00s gap between HOW and YOU yields separate HOW / YOU clauses; HELP is split from HELLO FRIEND I at a 1.533s gap. Conversely HUNGRY→MY at 1.467s remains joined, producing HUNGRY MY NAME fs-GELO, rendered “hungry is Gelo.” Timing alone is not reliably identifying sentence boundaries.
3. **The translation acceptance check is too weak.** `LiveStage3.rephrase` accepts any nonempty generated string <=300 characters. Spelling-slot validation only checks that each slot appears exactly once; it does not protect MY NAME or other recognized content. No semantic faithfulness check catches these examples.
4. **Spelling is independently unstable.** History contains fs-GEC, fs-GELO and fs-RGELCO among other fragments. HELLO MY NAME fs-GEC becomes “Hello, my name is Gec.” Here the renderer preserves what recognition supplied. The history cannot establish which exact letters/signs were physically made because it has no captured video.
5. **Finish is firing.** Seven two-hand events include six nonempty utterances, each followed by a saved translation. This does not measure exact physical hold duration.

## Recommended correction order

Add these exact saved inputs as translation regression cases; reject output that introduces unsupported content or drops protected meaning. Use reviewed rendering or a faithful literal fallback when generation fails those checks. Then replace unconditional pause-only clause splitting with a policy that also considers phrase structure and explicit Finish. Do not merely increase the global pause threshold: this session exhibits both over-splitting and inappropriate merging. Continue name recognition work separately; a renderer must not guess ANGELO from arbitrary fragments.

No app/model changes were made during this review. This is inspection of the saved outputs and current source, not a new translation replay or an accuracy estimate. Sources: `latest_snapshot.json`, `translation_pairs.json`; mobile `LiveReelStage3.swift:95`, `LiveReelDecoder.swift:649`.
