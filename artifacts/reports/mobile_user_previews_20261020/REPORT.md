# Owner-supplied mobile previews — 2026-10-20 PHT

Replaced all 100 matching previews in the shared Flutter bundle (Android/iOS Glosses)
and native iOS Practice bundle from `/Users/frnzlo/Downloads/signs`. The user confirmed
IMG_4068.MOV as STOP and IMG_4122.MOV as YOU; neither uses an incumbent fallback.

The folder contains exactly100clips:66named(Glosses33,HOW-SIGN33) and34IMG files.
Named videos retain supplied labels. For unnamed files, compiled the current native
Swift Vision/crop/FP16 primary-recognizer path and ran full-clip inference on the
same normalized bytes delivered to the app. Expected labels are not inference inputs.
32unnamed top1predictions are unused vocabulary labels with scores>=0.6; no filename
sequence or forced assignment supplied labels. Two low-confidence conflicts:
IMG_4068.MOV -> SCHOOL44.98%,PLEASE17.31%,STOP12.81%; IMG_4122.MOV -> ASK40.19%,
NEED21.72%,YOU10.80%. User explicitly confirmed these as STOP and YOU. confirmed_labels.json records
the overrides, and both catalog rows identify preview_label_source=user_confirmed.
These predictions assist labeling; they are not correctness/linguistic certification
or an accuracy benchmark. No training, protected-test access, data acquisition or
training-manifest changes occurred; new rows are preview_only/training_eligible=false.

Rendering applies source rotation, preserves full duration and framing with
aspect-preserving scaling to720maximum pixels, H264Main/yuv420p,30fps and no audio.
No source files were renamed, modified or removed. Source hashes in inventory.json
and prepared.json; model ranks in unnamed_predictions.json; selection.json maps clips
to labels. Old destination bytes are preserved in before/flutter and before/native_ios.
Publishing uses temporary files plus atomic replacement, avoiding large exFAT rewrites.
Catalogs and every staged/delivered asset agree across both app bundles.

Validation: all100videos fully decode without errors; native iOS AVPlayer test
`testAllPracticeReferencesPlayAndLoop` passes(all100clips), TESTSUCCEEDED.
AndroidDebug build succeeds. Receipts:media_validation.json,delivery.json,
ios_playback_final.log and android_final_build.log. Both model families and app runtime code
unchanged. Fresh signed iOS Release built and installed successfully on connected jaz iPhone 16.
Automatic launch was blocked because the device is locked; physical viewing remains
for the user after unlocking. Android APK built but no Android device was connected.
See ios_release_install.log and build_receipt.json.
The preceding recording-timing investigation was interrupted by this new request;
no recorder fix was implemented as part of this preview task.

Next safe action: verify the updated previews on the target phone. Preserve explicit
user labels and source hashes; do not substitute historical examples based on model
confidence. All 100 supplied clips are included (66 filename labels, 32 model-assisted
labels and two user-confirmed labels).
