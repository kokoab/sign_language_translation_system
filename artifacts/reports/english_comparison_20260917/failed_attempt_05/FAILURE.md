# Comparison failed

```
Traceback (most recent call last):
  File "/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/english_comparison_20260917/run_comparison.py", line 327, in <module>
    try:status=work()
  File "/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/english_comparison_20260917/run_comparison.py", line 203, in work
    b,torch,manifest,data=runtime()
  File "/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/english_comparison_20260917/run_comparison.py", line 107, in runtime
    b.check_checkpoint_space(startup=True)
  File "/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage1_direct_translation_20260917/run_experiment.py", line 267, in check_checkpoint_space
    raise RuntimeError('Insufficient checkpoint space: need 8 GiB at startup, 4 GiB during training')
RuntimeError: Insufficient checkpoint space: need 8 GiB at startup, 4 GiB during training
```
