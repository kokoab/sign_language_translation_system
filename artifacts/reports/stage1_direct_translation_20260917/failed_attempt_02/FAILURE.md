# Direct-translation experiment failed

No model promotion.

```
Traceback (most recent call last):
  File "/Users/frnzlo/Library/Python/3.9/lib/python/site-packages/torch/serialization.py", line 967, in save
    _save(
  File "/Users/frnzlo/Library/Python/3.9/lib/python/site-packages/torch/serialization.py", line 1268, in _save
    zip_file.write_record(name, storage, num_bytes)
RuntimeError: [enforce fail at inline_container.cc:863] . PytorchStreamWriter failed writing file data/541: file write failed

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage1_direct_translation_20260917/run_experiment.py", line 449, in work
    checkpoint,history=train(model,tokenizer,data,resume=resume)
  File "/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage1_direct_translation_20260917/run_experiment.py", line 378, in train
    torch.save(dict(format='slt_stage1_direct_translation_v17',epoch=epoch,
  File "/Users/frnzlo/Library/Python/3.9/lib/python/site-packages/torch/serialization.py", line 974, in save
    return
  File "/Users/frnzlo/Library/Python/3.9/lib/python/site-packages/torch/serialization.py", line 798, in __exit__
    self.file_like.write_end_of_file()
RuntimeError: [enforce fail at inline_container.cc:664] . unexpected pos 1588742400 vs 1588742288
```
