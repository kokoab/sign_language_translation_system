# Paper checkpoint lineage — 2026-10-13

The current primary output's95.50% is3779/3957 across admitted isolated validation
(Citizen364/378;SemLex846/953;local2569/2626). It is NOT the old landmark-only95.50%
(361/378Citizen), and NOT a direct descendant of the historical96.30%fusion checkpoint.
Its Citizen-only score is364/378=96.30%, numerically equal to the old fusion result but
a different model. Do not interpret95.50vs96.30 as a matched accuracy drop.

Historical branch: full-roll orientation model a7490409 (95.77%Citizen) → local
mouth-masked replay promotion12a74a18 (95.50%Citizen) +hand7e920d09 → historical
fusion1caeadf4 (96.30%Citizen) → historical phrase/interval checkpoints.

Current branch: partwise/global5c40b133 (96.83%Citizen,epoch100) → run_v2 local
replay promotion e5200e5f → run_v2 fusion5d2b680a → run_v2 phrase_adapt →
run_frag25_reject interval checkpoint99a18801 → retained original output in current
ConfidenceDual export, alongside transition-refined outputcd900f59.
Source-checkpoint metadata and local replay initialization records establish ancestry;
recipe explicitly fine-tunes5c40b133, full_roll_probability0,mild_roll_degrees12.
This branch is not the earlier full-roll model. Full-roll robustness measurements
froma7490409 must not be assigned to it without matched testing.

Current fusion saved validation:Citizen366/378,SemLex864/978,local2570/2626.
Current primary95.50usesSemLex953 ratherthan978; evaluate all stages on one pinned
membership before constructing a new performance ladder. Shared standalone hand parent
7e920d09 is present in both fusion source records; common hand ancestry does not make
the landmark/fusion lineage the same. No new accuracy inference, training, manuscript
change or production change in this audit. Hash checks inhash_checks.json.

Sources: run_v2/recipe.json; run_v2/local_replay/result.json; both fusion checkpoints'
source_checkpoints metadata; run_v2/phrase_adapt/result.json;run_frag25_reject/history.json;
my_transition_recovery_v17_20261011/confidence_training_recipe.json andtrain_confidence.py;
review_assets/20261013_verification/verification.json.
