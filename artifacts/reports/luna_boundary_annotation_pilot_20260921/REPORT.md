# One-Luna-per-clip boundary pilot

Review every source/Luna interval side by side in [comparison.html](comparison.html).

Twenty-four train-only signs were assigned to twenty-four separate Luna-low runs. Confidence: {'low': 1, 'medium': 23}. 4 annotations touched the first or last review frame and are treated as censored. 19 medium/high, non-censored annotations are admitted as provisional supervision.

Across all annotated items, the median absolute difference from the source timestamps is 68 ms at start and 145 ms at end; 6/24 agree at both edges within 100 ms and 16/24 within 200 ms. Among admitted items, agreement is 6/19 within 100 ms and 14/19 within 200 ms.

These are single-reviewer pseudo-labels, not ground truth. Source agreement measures consistency, not correctness. Do not scale or train on them unless this pilot demonstrates adequate consistency and edge coverage. Citizen test remained sealed.

The comparison review corrects the interpretation of this pilot. ASLLRP's published convention excludes preparatory and release motion from the lexical sign interval and may annotate final holds separately. The Luna instruction was close but individual reviewers often selected the fuller visible articulation or hold. The ±100 ms band was therefore too strict to use as a verdict on source-annotation quality. It was not the only cause of model failure: the coherent decoder still reached only 40.46% boundary F1 at ±200 ms.

The confirmed ASLLRP issues are downstream: 21/1,160 derived crops clipped at least one annotation (22 occurrences, all mapped to OTHER), and the earlier context objective stopped every known training window before its annotated sign end. These do not establish that the original ASLLRP annotations are broadly wrong.
