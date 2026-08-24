# Artifacts

Generated outputs belong here when they are not required by active scripts.

- `model_assets/` holds local `models/` and `weights/`; root-level names are symlinks.
- `generated/` holds build/runtime output directories.
- `reports/` holds generated charts, logs, metrics, and CSV outputs.
- `metrics/` holds evaluation JSON and output directories.
- `charts/` is reserved for future generated plot groupings.

Large checkpoints and datasets are not moved automatically during cleanup because many scripts reference their current paths.
