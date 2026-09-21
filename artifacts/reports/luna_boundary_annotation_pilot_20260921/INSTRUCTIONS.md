# Blind boundary annotation

Inspect only `blind_manifest.json` and the corresponding sheet. For each item, choose the first frame clearly belonging to the named target sign and the last frame still belonging to it. Exclude setup, release, and movement into another sign. If the target cannot be isolated confidently, reject it. Do not inspect `reference.json`; it contains the source annotations reserved for agreement analysis.

Write a JSON array with: `item`, `start_frame`, `end_frame`, `confidence` (`high`, `medium`, or `low`), and `reason`. Rejected items use null frames and confidence `reject`.
