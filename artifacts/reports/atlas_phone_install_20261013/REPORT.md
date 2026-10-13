# ATLAS installation on jaz — 2026-10-13

Requested installation is complete. The connected **jaz iPhone 16** now has
**ASL Translator 1.0.0 (build 1)**, bundle `com.kokoab.sltMobileApp`, built in
Release from `ATLAS/mobile_app/slt_mobile_app/ios/Runner.xcworkspace`.

- Generic and target-specific signed Release builds pass using the existing
  development team. No signing-project or inference-source edits.
- Developer Mode initially disabled; user enabled it and device verification passed.
- Target provisioning includes jaz and the prior iPhone 13. The new profile expires
  October 20, 2026 at 13:39 PHT.
- `codesign --verify --deep --strict` passes.
- `devicectl device install app` succeeds.
- First launch was blocked by iOS security before developer-profile trust. User
  approved trust on the phone; retry launch succeeds and process **589** was
  independently present in the device process inventory.

Build product:
`/Users/frnzlo/slt_atlas_jaz_release_build/Build/Products/Release-iphoneos/Runner.app`.
Installation and launch receipts: `install.json`, `launch.json`, `processes.json`.
Initial trust failure retained in `launch_untrusted.json` and `launch_untrusted.log`.
Build logs: `build.log`, `device_build.log`; state receipt: `status.json`.

This includes the current ConfidenceDual pipeline and latest preview assets. No
model training, protected evaluation, acquisition, uninstall, app-data deletion
or change to the prior iPhone 13 installation. Launch verification is not a new
signer-accuracy, latency or thermal measurement. Next: use ASL Translator on jaz
and record any further device observations with their actual scope.
