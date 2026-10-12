# Current-lineage component evaluation — 2026-10-13

User authorized cached validation within five minutes. Seven outputs: standalone
landmark and hand branches feeding current-lineage fusion, fusion itself, phrase
adaptation, interval adaptation, current primary and transition-adapted output.
Identical raw validation files and ordered IDs for all outputs. Verify standalone
parent hashes and tensor identity; reuse shared frozen encoders for transition output.
No training, extraction, protected-test access, deployment or manuscript changes.
Results per domain and total correct/total examples, not mean of domain percentages.
The transition output alone does not establish the conditional runtime accuracy.

Execution order: evaluate.py produces all cached predictions; apply_roles.py preserves
those and applies existing isolated_role_check exclusions equally to every output;
verify.py independently checks counts and reproduces prior primary/adapted totals.
25 SemLex records excluded from reporting; all outputs use3957remaining examples.
Completed; inference loop42.2seconds excluding model loading and verification.
