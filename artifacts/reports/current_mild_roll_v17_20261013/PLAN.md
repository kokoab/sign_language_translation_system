# Mild-roll verification — 2026-10-13

Verify current run_v2 landmark adaptation recipe: full-roll probability0, mild roll12°.
Existing related saved evaluations use0/17/37/73/90/123/180/270 or0/37/90/180, not the
requested ±12° range and not the same current checkpoint. New targeted inference on
base5c40b133 and adapted e5200e5f at−12/−6/0/+6/+12degrees, same3957admittedinputs,
using geometry-preserving rotate_camera_roll_v17. Includesperdomaincounts.
No real video rotation, new extraction, training or protected-test access.
Training differences include data adaptation as well as augmentations, so before/after
scores do not isolate the effect of camera-roll augmentation. No claim about unmeasured
angles between grid points, pitch/yaw/perspective, or final multistream runtime.
