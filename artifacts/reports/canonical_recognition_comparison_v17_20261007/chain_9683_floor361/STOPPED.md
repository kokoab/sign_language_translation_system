# Stopped 2026-10-07 at epoch 9

Stopped deliberately: this Variant B run kept the August 35% full-circle roll augmentation.
The 96.83 parent was trained before full roll existed (provenance: augmentation without
roll fields; a7490409 is the same recipe plus full roll 0.35). The roll shift, not the local
data, caused the Citizen drop (epochs 1-9 here: 94.71-95.50%), matching the earlier
paired fine-tune full-roll arm. Superseded by chain_9683_floor361_mildroll/. Not audited.
