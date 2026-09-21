# Segment-first experiment failed

```text
Traceback (most recent call last):
  File "/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/segment_first_v17_20260921/run_experiment.py", line 731, in worker
    train_worker(); status = "completed"
  File "/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/segment_first_v17_20260921/run_experiment.py", line 562, in train_worker
    train_segments += isolated_samples(SEMLEX_TRAIN, labels, "semlex", 5)
  File "/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/segment_first_v17_20260921/run_experiment.py", line 186, in isolated_samples
    raise FileNotFoundError(f"missing {source}/{label}")
FileNotFoundError: missing semlex/THEY
```
