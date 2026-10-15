#!/bin/zsh
# iPhone builds and native tests from an internal-disk mirror of the canonical Flutter project.
#
# The repository volume is exFAT (FSKit). Rewriting large existing files there (Flutter's
# kernel snapshots, test caches, build products) fails with ENXIO / EIO and briefly makes the
# volume unreadable, so builds never write into the repository: the project is mirrored to the
# internal disk (sources only; build/, .dart_tool/ and android/ are regenerated or unused),
# Flutter's iOS config is regenerated for the mirror path, and Xcode builds there.
#
#   ATLAS/mobile_app/build_ios_release_mirror.sh [device-udid] [--install]
#       signed Release for the phone (default angelo iPhone 13), optionally installed and launched
#   ATLAS/mobile_app/build_ios_release_mirror.sh --test <simulator-udid> [RunnerTests/RunnerTests/testName ...]
#       Debug simulator tests (all RunnerTests when no test is named)
set -euo pipefail

SOURCE="$(cd "$(dirname "$0")/slt_mobile_app" && pwd)"
MIRROR="${SLT_IOS_MIRROR:-$HOME/slt_mobile_build_mirror/slt_mobile_app}"

mirror() {
  mkdir -p "$MIRROR"
  rsync -a --delete --exclude '/build/' --exclude '/.dart_tool/' --exclude '/android/' \
    --exclude '/ios/Flutter/Generated.xcconfig' --exclude '/ios/Flutter/flutter_export_environment.sh' \
    "$SOURCE/" "$MIRROR/"
  cd "$MIRROR"
  flutter pub get >/dev/null
  flutter build ios "$1" --config-only >/dev/null
  grep -q "FLUTTER_APPLICATION_PATH=$MIRROR" ios/Flutter/Generated.xcconfig
}

if [[ "${1:-}" == "--test" ]]; then
  SIMULATOR="${2:?simulator udid}"
  shift 2
  mirror --debug
  only=()
  for t in "$@"; do only+=("-only-testing:$t"); done
  xcodebuild -workspace ios/Runner.xcworkspace -scheme Runner -configuration Debug \
    -destination "platform=iOS Simulator,id=$SIMULATOR" \
    -derivedDataPath "${SLT_IOS_TEST_DERIVED:-$HOME/slt_mobile_controls_build}" \
    -parallel-testing-enabled NO -collect-test-diagnostics never "${only[@]}" test
  exit
fi

DERIVED="${SLT_IOS_DERIVED:-$HOME/slt_mobile_controls_device}"
DEVICE="${1:-00008110-00111D1A0130A01E}"
INSTALL="${2:-}"
mirror --release
xcodebuild -workspace ios/Runner.xcworkspace -scheme Runner -configuration Release \
  -destination "id=$DEVICE" -derivedDataPath "$DERIVED" -allowProvisioningUpdates build

if [[ "$INSTALL" == "--install" ]]; then
  APP="$DERIVED/Build/Products/Release-iphoneos/Runner.app"
  xcrun devicectl device install app --device "$DEVICE" "$APP"
  xcrun devicectl device process launch --device "$DEVICE" com.kokoab.sltMobileApp
fi
