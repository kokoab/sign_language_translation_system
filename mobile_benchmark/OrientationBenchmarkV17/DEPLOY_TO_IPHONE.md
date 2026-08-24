# Deploy locked-100 SLT v17 to an iPhone

This guide installs the current file-video recognizer and bounded English naturalizer
on a physical iPhone. The Xcode project targets iOS 17.0 or newer and uses automatic
signing. The repository intentionally does not track model packages, datasets, or
compiled apps; the required model packages must remain at their documented local
paths on the Mac used for the build.

## 1. Confirm the local model assets

From the repository root:

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT

test -d artifacts/coreml/MobileCLIP2S0ImageEncoderV17FP32.mlpackage
test -d artifacts/coreml/Stage2FrozenEncoderV17FP32.mlpackage
test -d artifacts/coreml/Stage2CompactContextV17FP32.mlpackage
test -f mobile_benchmark/OrientationBenchmarkV17/OrientationBenchmarkV17/Stage2MobileV17_manifest.json
test -f active/v17/citizen100_manifest.json
test -f active/v17/stage3_mobile_naturalizer_manifest_v17.json
```

All six commands must exit without printing an error. The three `.mlpackage`
directories are referenced by the Xcode project and compiled into the app during the
build. They are intentionally ignored by Git because they are binary model assets.

The selected model provenance should remain:

- Stage-2 checkpoint SHA-256:
  `623f9b56141643704b3562a8d2fdcebe44269985b2f618eb8f0a471e857a2cf5`
- Stage-2 contract SHA-256:
  `8be66a44d337dd99484d3ee3140f3124c2e121abe20e93ce7f09b94d96ecc30d`
- Stage-3 manifest SHA-256:
  `68c7ce67632f66ee70fa3b3d36eb8df33ad72dc674edbf3b720e93c1240f84a6`

Verify the two regular files with:

```bash
shasum -a 256 \
  active/v17/stage2_to_stage3_contract_v17.json \
  active/v17/stage3_mobile_naturalizer_manifest_v17.json
```

## 2. Prepare the iPhone and Xcode

1. Update the iPhone to iOS 17 or newer.
2. Install and open the current Xcode release on the Mac.
3. Connect the iPhone to the Mac with a data-capable USB cable for the first install.
4. Unlock the iPhone and accept **Trust This Computer** if it appears.
5. In Xcode, open **Xcode > Settings > Accounts**.
6. Add the Apple ID that will sign the app. A free Personal Team is sufficient for
   local testing, although its provisioning profile expires periodically.
7. If the iPhone asks for Developer Mode, open
   **Settings > Privacy & Security > Developer Mode**, enable it, restart the phone,
   and confirm after restart.

Wireless debugging can be enabled later from Xcode's **Window > Devices and
Simulators**. Keep the cable connected until the first successful installation.

## 3. Open the correct project

From Terminal:

```bash
cd /Users/frnzlo/Documents/machine_learning/SLT
open mobile_benchmark/OrientationBenchmarkV17/OrientationBenchmarkV17.xcodeproj
```

Use the `.xcodeproj` file shown above. This project has no CocoaPods workspace.

## 4. Configure signing

1. In Xcode's left project navigator, select the blue
   **OrientationBenchmarkV17** project.
2. Under **TARGETS**, select **OrientationBenchmarkV17**.
3. Open **Signing & Capabilities**.
4. Keep **Automatically manage signing** enabled.
5. Choose your Apple ID's Team.
6. If Xcode says the bundle identifier is unavailable, replace
   `com.local.slt.OrientationBenchmarkV17` with a unique reverse-domain identifier,
   for example `com.kokoab.slt.OrientationBenchmarkV17`.
7. Wait for Xcode to report that the signing certificate and provisioning profile are
   ready.

Changing the bundle identifier does not change model behavior or report hashes.

## 5. Select the physical iPhone

At the top of Xcode:

1. Select the **OrientationBenchmarkV17** scheme.
2. Open the destination menu beside the scheme.
3. Under **iOS Devices**, select the connected iPhone—not an iPhone simulator and not
   **Any iOS Device**.
4. Keep the iPhone unlocked while Xcode prepares developer support.

If the phone is shown as unavailable, open **Window > Devices and Simulators**, select
it, and read the specific status message. Common causes are a locked phone, Developer
Mode being disabled, or an iOS version newer than the installed Xcode supports.

## 6. Build and install

Press **Command-R**, or choose **Product > Run**.

Xcode will:

1. compile the Swift app for arm64 iPhoneOS;
2. compile the three Core ML packages;
3. copy the vocabulary, Stage-2, and Stage-3 manifests;
4. sign the app with the selected Team;
5. install it on the selected iPhone; and
6. launch it.

The first clean build may take longer because Core ML must compile all three models.
The validated unsigned Release bundle is approximately 105 MiB; the signed installed
size may differ.

If Xcode reports stale build output, use **Product > Clean Build Folder** while holding
Option, then press **Command-R** again. Do not delete the model packages under
`artifacts/coreml/`.

## 7. Trust the developer certificate if requested

This is usually automatic for an Xcode-connected phone. If iOS blocks launch:

1. Open **Settings > General > VPN & Device Management** on the iPhone.
2. Select the developer profile associated with the Apple ID.
3. Tap **Trust** and confirm.
4. Return to Xcode and press **Command-R** again.

The Mac and phone may need internet access the first time Apple creates or verifies a
development provisioning profile.

## 8. Run a video through the model

1. Put the test video in the iPhone's **Files** app. A video available through a
   document-provider location such as iCloud Drive also works.
2. Open **Orientation v17 Benchmark**.
3. Tap **Choose video** and select the movie.
4. Enable **Pixels are mirrored** only if the saved file itself is horizontally
   mirrored. Do not enable it merely because the video came from a front camera;
   inspect the saved video first.
5. Choose an expected isolated gloss if the run is an accuracy check. This field only
   scores the report; it does not constrain the prediction.
6. Start with **20 iterations** for a quick functional test. Use **200 iterations**
   only for a sustained benchmark.
7. Tap **Extract and benchmark** and keep the app in the foreground.

The app accepts portrait, landscape, square, and rotated files. It preserves aspect
ratio, selects the nearest upright quadrant, and lets the trained augmentation handle
the remaining roll. It samples at most 256 source frames and processes only one
32-frame window at a time.

## 9. Read and export the result

The result screen shows:

- the ordered recognized gloss sequence;
- bounded natural English;
- literal gloss-preserving English;
- whether Stage 3 used a reviewed template or the safe literal fallback;
- Apple Vision extraction time and chosen coarse rotation;
- Core ML median and p90 inference time;
- thermal state and resident-memory readings; and
- the compiled model size.

Tap **Export JSON** to save or share the complete report. Preserve the JSON when
comparing builds because it pins the checkpoint, vocabulary, Core ML package hashes,
Stage-3 manifest, environment, and input hashes.

## 10. Interpret physical-device evidence correctly

A report created by this installed app on the physical phone is valid evidence for
that phone, OS version, selected video, and run conditions. For repeatable results:

1. close other demanding apps;
2. let the phone cool to a normal thermal state;
3. use the same video and iteration count;
4. run at least three sustained trials; and
5. report median and p90 rather than only the fastest inference.

The current UI uses selected video files. It does not implement live-camera capture.
The Stage-3 component is a bounded naturalizer for the locked 100-gloss recognizer,
not a general conversational ASL translator.

## Troubleshooting

### Xcode says a Core ML package is missing

Run the six checks in step 1. Git does not download those model packages. Restore them
to `artifacts/coreml/` on this Mac, keeping the exact filenames.

### Signing requires a development team

Choose the Team under **Signing & Capabilities**. If no Team is listed, add the Apple
ID under **Xcode > Settings > Accounts** first.

### The bundle identifier cannot be registered

Change it to a unique identifier under **Signing & Capabilities**. Do not change the
product name or model resource filenames.

### The phone does not appear as a run destination

Unlock and reconnect it, accept Trust, enable Developer Mode, then inspect
**Window > Devices and Simulators**. Also ensure the installed Xcode supports the
phone's iOS version.

### The app installs but will not open

Trust the developer profile under **Settings > General > VPN & Device Management**.
For a free Personal Team, rebuild from Xcode when the temporary provisioning profile
expires.

### The app reports an empty or unexpected gloss sequence

First test a clear, front-facing isolated sign from the locked vocabulary. Check the
mirrored toggle, make sure the signer and hands are visible throughout the clip, and
export the JSON. Do not interpret simulator HELLO evidence as independent physical
capture accuracy.
