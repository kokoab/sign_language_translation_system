# Text-to-sign genuine retrieval baseline

[Watch `hello how you`](phrase.mp4) · [Inspect the report](report.json)

This is the first safe reverse path: text is normalized to an exact known gloss
sequence, then a complete genuine local phrase trajectory is rendered as landmarks.
The current catalog contains the nine local phrases. An unknown combination fails
closed; it is not stitched from isolated signs.

The landmark rig preserves which hands participate. A one-handed source remains
one-handed, and generated/render-only data remains ineligible for recognizer training.
