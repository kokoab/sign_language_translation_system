# Corrected local-phrase matched rerun

User confirmed every video retained in PHRASES FIXED was checked against its folder phrase.
The fixed folder is an exact-content subset: 685 retained, 95 removed, no edits/relabels.
The existing signer-disjoint cache admits 232 local train and 200 local validation clips;
55 old training clips are removed. GOOD_MORNING has no remaining local train examples.
All retained feature/target arrays and signer roles are unchanged. Excluded uncertain
cache rows and the three out-of-recipe phrase families remain excluded.

Reuse the exact baseline/pretrained initial heads for seeds17321/17322 from the first
pilot. Do not rerun YouTube pretraining or extraction. Run both supervised CTC arms for
18 epochs with the corrected root, identical decoder and selection rule. Keep ASLLRP,
NCSLGR, Citizen, SemLex and the frozen Apple recognizer unchanged. Do not add Flores or
O5S5 to this comparison: doing so would confound the label correction with new data.

Audit, data preparation and checks stay in-session. Detach only training after its
first optimizer step; no polling. Write comparison/behavior reports and a macOS
completion notification. No automatic promotion or protected test access.

Flores was previously tested (155 sentences; mixed outcome). It remains eligible for
a separately controlled pretraining arm; its earlier failure is not a permanent ban.
O5S5 positive intervals require a partial-supervision recipe, not inferred blank gaps.
