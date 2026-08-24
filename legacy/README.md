# Legacy Code

This directory stores code that is preserved for history or compatibility but is not the current v16 pipeline.

- `copies/` contains duplicate `* copy.py` files moved out of `src/`.
- `dsgcn_src/` contains the older v10-v15 DS-GCN model/training variants.

The main older DS-GCN stage files still remain in `src/`. Versioned legacy variants in `legacy/dsgcn_src/` have thin wrappers in `src/` so existing imports keep working.
