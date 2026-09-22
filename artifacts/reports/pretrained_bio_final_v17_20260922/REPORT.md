# BIO-preserving final planned adaptation

Trained 12 epochs. Selected epoch 0 (0 means frozen fallback). No automatic deployment.

| Split | Frozen WER | Selected WER | Frozen correct | Selected correct | Frozen insertions | Selected insertions |
|---|---:|---:|---:|---:|---:|---:|
| calibration | 49.33% | 49.33% | 82/150 | 82/150 | 6 | 6 |
| confirmation | 43.42% | 43.42% | 46/76 | 46/76 | 3 | 3 |

selected.pth is the recognition-selected result; best_trained.pth preserves the best actually fine-tuned candidate even if frozen wins.
Only last attention block and original BIO head adapted. Unknown/overlapping labels ignored; no invented background. Frozen BIO KL is a regularizer, not human ground truth. Original decoder/500msfuture preserved.
Existing59/30local partitions, familiar3signers/sixphrases/15glosses; reused development, not fresh independent test. Expanded72 untouched. No accuracy guarantee or mobile readiness claim.
