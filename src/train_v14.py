import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

if __name__ == "__main__":
    runpy.run_module("legacy.dsgcn_src.train_v14", run_name="__main__")
else:
    from legacy.dsgcn_src.train_v14 import *  # noqa: F401,F403
