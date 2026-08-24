# Current Stage 1 external-signer audit

**Purpose:** domain-shift diagnostic only  
**Checkpoint:** `src_v16/output_v16_d384/best_model.pth`  
**Coordinate schema:** `legacy_anisotropic`  
**Videos:** 72

The checkpoint was selected using the existing 310-class, random-file evaluation. These ASL Citizen samples contain public participant IDs not used to construct that local split. Dataset labels and variants have not yet received the required ASL review.

## Results

- Top-1 accuracy: **40.28%**
- Top-5 accuracy: **55.56%**
- Most frequent top-1 outputs: `{"GOOD": 12, "HELP": 6, "DRINK": 4, "LOVE": 4, "SCHOOL": 4, "HELLO": 3, "ELEVEN": 3, "MOTHER": 2, "YOU": 2, "HOSPITAL": 2}`

### Citizen official-split identities

All three groups are external to the local seven-person dataset. The split names are preserved only for future dataset construction.

| Citizen split | Videos | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| train | 24 | 45.8% | 62.5% |
| val | 24 | 45.8% | 58.3% |
| test | 24 | 29.2% | 45.8% |

### Per-sign results

| Sign | Top-1 | Top-5 | Observed top-1 outputs |
| --- | ---: | ---: | --- |
| COME | 0.0% | 0.0% | PAST (2), COOK (1), WRONG (1), DIRTY (1), KNOW (1) |
| DRINK | 50.0% | 66.7% | DRINK (3), IF (1), WOMAN (1), YESTERDAY (1) |
| GOOD | 100.0% | 100.0% | GOOD (6) |
| GOODBYE | 0.0% | 0.0% | AT (1), THEIR (1), ALSO (1), TAKE (1), MOTHER (1), COMPANY (1) |
| HELLO | 50.0% | 66.7% | HELLO (3), LOUD (1), MOTHER (1), DRINK (1) |
| HELP | 83.3% | 100.0% | HELP (5), STOP (1) |
| HOSPITAL | 16.7% | 50.0% | SIXTEEN (1), DAY (1), OR (1), HOSPITAL (1), LONG (1), LOOK (1) |
| LOVE | 66.7% | 83.3% | LOVE (4), HELP (1), SLEEP (1) |
| SCHOOL | 66.7% | 83.3% | SCHOOL (4), GOOD (1), STOP (1) |
| THANKYOU | 0.0% | 33.3% | GOOD (5), MY (1) |
| WHAT | 16.7% | 33.3% | ELEVEN (2), WANT (1), DRIVE_CAR (1), WHAT (1), PAY (1) |
| YOU | 33.3% | 50.0% | YOU (2), HE (1), HOSPITAL (1), NO (1), ELEVEN (1) |

This result must not be compared directly with a future signer-locked 100-class score. It measures the current checkpoint under external dataset shift and is useful mainly for deciding whether retraining is necessary.
