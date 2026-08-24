import runpy
import sys

sys.path.insert(0, "src")

if __name__ == "__main__":
    runpy.run_path("scripts/extraction/extract_batch_fast_v2.py", run_name="__main__")
else:
    from scripts.extraction.extract_batch_fast_v2 import *  # noqa: F401,F403
