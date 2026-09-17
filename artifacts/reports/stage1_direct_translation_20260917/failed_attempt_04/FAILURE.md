# Direct-translation experiment failed

No model promotion.

```
Traceback (most recent call last):
  File "/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage1_direct_translation_20260917/run_experiment.py", line 478, in work
    checkpoint,history=train(model,tokenizer,data,resume=resume)
  File "/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage1_direct_translation_20260917/run_experiment.py", line 408, in train
    tmp.replace(checkpoint)
  File "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/pathlib.py", line 1385, in replace
    self._accessor.replace(self, target)
FileNotFoundError: [Errno 2] No such file or directory: '/Volumes/secret/SLT_checkpoints_20260918/stage1_direct_translation_20260917/latest.tmp.pth' -> '/Volumes/secret/SLT_checkpoints_20260918/stage1_direct_translation_20260917/latest.pth'
```
