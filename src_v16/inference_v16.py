import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active.v16.inference_v16 import *  # noqa: F401,F403

if __name__ == "__main__":
    from active.v16.inference_v16 import main

    main()
