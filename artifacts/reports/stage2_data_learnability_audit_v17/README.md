# Stage 2 data and architecture diagnosis

The complete audit supports the user's diagnosis in part: the continuous data path is
the bottleneck, but the raw videos are not simply too fast or unusable. Fixed-window
labels, incomplete exposure, thin per-class signer coverage, and a Stage 1/Stage 2
interface that prevents sequence loss from shaping the visual encoder are the measured
failure factors.

- [Complete data audit](DATA_AUDIT.md)
- [Architecture research and recommendation](ARCHITECTURE_RESEARCH.md)
- [Machine-readable evidence](audit.json)
- [Verification record](verification.json)

Recommended next model: reuse the existing Stage-1 Squeezeformer as a shared unpooled
frame encoder; attach a shallow causal two-scale Conv1D CTC head; train it jointly with
the existing isolated classifier and exact-core/verified-blank auxiliary losses. Keep
O5S5 positive-only. Do not run another pooled-window or frozen-encoder Stage-2 variant.

No model was trained in this diagnostic step, no runtime/default changed, and the
Citizen test was not accessed.
