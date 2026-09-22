# Twenty-minute feasibility check

User authorized the Zuo experiment only if it takes less than20minutes. No new training
launched: a meaningful changed-recipe experiment is not currently prepared or timed.
Measured earlier O5S5window training took108.43seconds for12epochs/564updates onMPS;
this is training time only, not data preparation, implementation, validation or fullstream
evaluation. Re-running it would not constitute a new Zuo comparison.

Code inspection confirms active/v17/train_stage1_window_v17.py already has contextual
foreground masks, background label, category/source/class-balanced sampling, window CE,
0.5foregroundCE and isolatedteacherKL. Earlier shorthand 'addsaliency/background' is
therefore not a sufficient new intervention. Its objective differs from the paper's
instance plus groupedgloss classification and saliency recipe; architecture also differs.
Historical O5S5readycontext covers73/100classes, not complete100context coverage.

Before a new result can be called meaningful: specify the missing paper component(s),
prepare a current approved input contract without olddefault bypass, verify loss/streaming
behavior, then pairedtraining+evaluation. No measured total-runtime estimate exists for
that work. Cannot responsibly promise complete study+results within20minutes from now.
No acquisition/checkpoint/data changes. FullfaithfulTwoStream replication not timed.
