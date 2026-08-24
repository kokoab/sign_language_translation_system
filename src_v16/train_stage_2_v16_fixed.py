import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    runpy.run_module("active.v16.train_stage_2_v16_fixed", run_name="__main__")
else:
    from active.v16.train_stage_2_v16_fixed import *  # noqa: F401,F403
