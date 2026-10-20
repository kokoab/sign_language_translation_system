#!/bin/zsh
# Build Android on internal disk; avoid large in-place rewrites on the exFAT repository.
set -euo pipefail
SOURCE="$(cd "$(dirname "$0")/slt_mobile_app" && pwd)"
MIRROR="${SLT_ANDROID_MIRROR:-$HOME/slt_android_build_mirror/slt_mobile_app}"
[[ "$MIRROR" != "$SOURCE" && "$MIRROR" != / && "$MIRROR" != "$HOME" ]] || exit 2
mkdir -p "$MIRROR"
rsync -a --delete --exclude '/build/' --exclude '/.dart_tool/' --exclude '/ios/' \
  --exclude '/android/.gradle/' --exclude '/android/local.properties' "$SOURCE/" "$MIRROR/"
# Existing SDK and Java installations are local environment inputs, not source changes.
SDK="${ANDROID_HOME:-$(sed -n 's/^sdk.dir=//p' "$SOURCE/android/local.properties")}"
export JAVA_HOME="${SLT_ANDROID_JAVA_HOME:-$(cd "$SOURCE/../../.." && pwd)/artifacts/generated/android_tools/jdk17/Contents/Home}"
cd "$MIRROR"
printf 'sdk.dir=%s\nflutter.sdk=%s\n' "$SDK" "$(dirname "$(dirname "$(command -v flutter)")")" > android/local.properties
flutter pub get >/dev/null
cd android
./gradlew --no-daemon --max-workers=2 -Dorg.gradle.jvmargs=-Xmx2048m "${@:-assembleDebug}"
