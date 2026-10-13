#!/bin/sh
# Sequential training of all lookahead settings; wall-clock logged in training_log.json.
cd "$(dirname "$0")"
for k in 1 0 2 full; do ../../../venv/bin/python tagger.py $k 5 || exit 1; done
