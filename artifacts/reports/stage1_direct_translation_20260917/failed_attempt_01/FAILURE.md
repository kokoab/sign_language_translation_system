# Direct-translation experiment failed

No model promotion.

```
Traceback (most recent call last):
  File "/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage1_direct_translation_20260917/run_experiment.py", line 388, in work
    checkpoint,history=train(model,tokenizer,data)
  File "/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage1_direct_translation_20260917/run_experiment.py", line 312, in train
    optimizer.step()
  File "/Users/frnzlo/Library/Python/3.9/lib/python/site-packages/torch/optim/optimizer.py", line 516, in wrapper
    out = func(*args, **kwargs)
  File "/Users/frnzlo/Library/Python/3.9/lib/python/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
  File "/Users/frnzlo/Documents/machine_learning/SLT/venv/lib/python3.9/site-packages/transformers/optimization.py", line 892, in step
    update = (grad**2) + group["eps"][0]
RuntimeError: MPS backend out of memory (MPS allocated: 7.31 GiB, other allocations: 22.32 GiB, max allowed: 30.19 GiB). Tried to allocate 732.75 MiB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).
```
