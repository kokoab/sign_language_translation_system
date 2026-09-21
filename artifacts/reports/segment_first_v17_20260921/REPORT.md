# Segment-first Stage 2 experiment

This run trains one class-independent OUTSIDE/START/SIGNING/END head, one internal KNOWN/UNKNOWN gate, and the locked 100-gloss Stage-1 classifier. It uses only the confident manifest. Questionable annotations are masked; O5S5 never supplies background; annotation gaps are not a transition class.

## Data and training

Full coverage used 7,788 boundary windows and 5,283 segment/replay examples per epoch for 8 epochs on MPS. The public vocabulary remained exactly 100; Citizen test stayed sealed.

## Results

| Measure | Result |
| --- | ---: |
| Online boundary F1 at ±100 ms | 14.44% |
| Online boundary F1 at ±200 ms | 24.41% |
| Exact-core known/unknown balanced accuracy | 63.34% |
| Exact-core known gloss accuracy | 74.78% |
| Online matched known gloss accuracy | 71.70% |
| Online visible WER | 204.69% |
| Citizen + SemLex validation gloss accuracy | 85.77% |
| Synthetic held signs emitted once | 6/20 |
| Synthetic intentional repeats emitted twice | 2/20 |

The synthetic hold/repeat rows are time-warped feature probes, not independent recorded performances. Promotion remains false until the measured gates justify changing the live runtime. Full counts, confusion matrices, source breakdowns, and thresholds are in `metrics.json`.
