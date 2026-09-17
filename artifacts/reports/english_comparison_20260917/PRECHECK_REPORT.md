# Approved English comparison: precheck

The user approved one BART-base pilot and a no-retraining vocabulary-trimmed mT5 comparison. The current baseline is left unchanged. This is preparation, not a translation-quality result.

- Frozen BART revision: aadd2ab0ae0c8268c7c9693540e9904811f36177; local weights and tokenizer assets hashed in manifest.json.
- Same 994 translation training utterances and 1,901 isolated replay examples; Citizen test remains sealed. BART target length mean 21.56, maximum 63 tokens; bounded padding never truncates.
- Trim vocabulary: 64,000 entries from original 250,100 SentencePiece entries, using 500 Brown documents and frozen training/prefix/special/basic-character coverage. All protected token sequences and decoded strings match after ID remapping. Actual full hybrid parameter count will be recorded after trimming; no checkpoint was altered during preparation.
- Six focused CPU checks cover existing model interfaces/gradients, deterministic full coverage, process-exit events, target buckets, BART BOS/EOS and padding, and exact retained mT5 logits/token remapping. See tests.log and verification.json for final test status.
- BART released repeat suppression is explicitly disabled. No repeat-removal rule is used. Learning rates, full-epoch coverage, 20-epoch schedule and isolated auxiliary loss follow the approved proposal.
- Detached supervisor waits for the baseline process exit via a kernel event, then checks successful completion. If baseline chrF does not beat its zero-visual control, it writes a stop report instead of launching BART. Otherwise it evaluates the trimmed checkpoint, executes a disposable 36-update BART training-only preflight, resets weights, trains, evaluates controls/retention and writes REPORT.md or FAILURE.md. No assistant polling or overlapping GPU training.
- Review findings corrected: failed worker return codes now propagate through the supervisor, and canonical status transitions are recorded. Special-token and weight-preservation integration reviewed.

Twelve paired development sentences are insufficient to establish no meaningful accuracy loss. Results are a screening comparison; there is no automatic deployment, mobile-readiness claim, or additional model search. Notification is configured on completion, failure or the baseline-grounding stop.
