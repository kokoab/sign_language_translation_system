# Flores OTHER completion review

All four arms completed; completion notification returned0. Higher MPS cap avoided
the previous observed allocation failure. Checkpoints exist, original source weights
and validation sources match across arms, only141Flores training sequences are added,
and all saved behavior edit counts were independently recomputed successfully.

Flores improves ASLLRP WER in both seeds54.17→50.00% and62.50→41.67%, and NCSLGR
92→84% and98→86%. Local WER worsens48.52→55.56% in17321, improves50.19→48.33%
in17322. Isolated exact83.92→83.41% and83.41→81.12%. No paired gate passes (0/2).
The stronger behavior warning is local deletions68→204 and52→126, while exact phrases
fall42→14/200 and43→29/200. Lower insertion counts do not establish better recognition.
Synthetic hold exact10→7/20 and6→8/20; repeat7→7/20 and6→4/20. Known isolated signs
with OTHER and no expected sign3→3/1356 and0→8/1356. This diagnostic does not prove
that OTHER caused all phrase deletions; it does not identify a semantic data defect.

No promotion. Flores contains useful supervision for some development domains, but
this exact OTHER-span/10%sample-weight recipe sacrifices local completeness and isolated
retention. These results neither establish unusable Flores data nor justify expanding
acquisition. Existing development sets were reused, not a fresh unbiased test. Protected
test/devtest data remain untouched. Any future change needs a separate matched recipe.
