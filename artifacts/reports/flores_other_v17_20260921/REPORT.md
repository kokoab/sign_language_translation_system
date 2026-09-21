# Flores OTHER experiment failed

The first without-Flores arm completed. The with-Flores arm completed one epoch, then failed with MPS out of memory:1.09GiB tensor allocations+1.08GiB other allocations exceeded the2.13GiB process cap. No complete paired comparison exists.

The previous single-long-sequence/tiny-training preflight did not cover sustained full-batch allocations. This is a resource failure, not evidence that Flores supervision hurts recognition. Raw traceback is in status.json.

CPU retry uses separate output roots and restarts both arms; previous baseline is preserved.
