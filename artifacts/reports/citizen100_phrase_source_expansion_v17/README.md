# Citizen100 phrase-source expansion

ASL Citizen itself contains isolated single-sign clips. Its official recommended-use
page cautions against treating concatenated Citizen clips as continuous signing because
natural sentences contain modulation, coarticulation and grammatical structure absent
from isolated examples.

The closest released phrase source tied to the ASL Citizen dictionary is ASL STEM Wiki.
Its April 2026 expert annotation supplement provides manual sequence glosses for nearly
500 continuous videos. One malformed annotation was rejected. The local selective acquisition now contains 267 target-bearing
videos across 19 participant IDs:

- 169 newly usable videos and 98 previously reviewed videos
- 44/100 exact locked raw-gloss strings
- 503 locked-string occurrences inside genuine continuous sentences
- 267/267 SHA-256 and full-decode checks passed
- training eligibility remains false pending signer, exact Citizen variant and boundary review

Manifest: `data/local/asl_stem_wiki_bootstrap_v1/manual_candidate_expansion_manifest.json`

The next usable source after this pool is FLEURS-ASL. It contains 1,749 sentences from
five Certified Deaf Interpreters. The released pseudo annotations suggest locked-vocabulary
overlap, but they are model-generated rather than expert gloss labels, so they should be
used only after the manual ASL STEM Wiki expansion and must remain review-only.

Official sources:

- https://www.microsoft.com/en-us/research/project/asl-citizen/recommended-use/
- https://www.microsoft.com/en-us/research/project/asl-stem-wiki/
- https://machinelearning.apple.com/research/sign-language-annotations
- https://www.bu.edu/asllrp/ASL-SignBank-and-other-Resources.html
