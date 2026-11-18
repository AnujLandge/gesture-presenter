#!/bin/bash
# Builds "Gesture Presenter.app" (Apple Silicon). Run from the gesture-presenter folder:
#   bash build_mac.sh
# The app is built OUTSIDE iCloud-synced folders (Desktop/Documents), because iCloud
# adds file attributes that break code signing. Result: ~/Applications/Gesture Presenter.app
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt pyinstaller

MODEL=gesture_presenter/models/hand_landmarker.task
mkdir -p gesture_presenter/models
[ -f "$MODEL" ] || curl -sSL -o "$MODEL" \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

APP="Gesture Presenter"
WORK="$HOME/.cache/gesture-presenter-build"
OUT="$HOME/Applications"
mkdir -p "$WORK" "$OUT"
rm -rf "$WORK"/* "$OUT/$APP.app" build dist "$APP.spec"

# App icon: turn assets/icon.png into a macOS .icns file
ICON_ARGS=()
if [ -f assets/icon.png ]; then
  SET="$WORK/icon.iconset"
  mkdir -p "$SET"
  for size in 16 32 128 256 512; do
    sips -z $size $size assets/icon.png --out "$SET/icon_${size}x${size}.png" >/dev/null
    sips -z $((size * 2)) $((size * 2)) assets/icon.png --out "$SET/icon_${size}x${size}@2x.png" >/dev/null
  done
  iconutil -c icns "$SET" -o "$WORK/icon.icns"
  ICON_ARGS=(--icon "$WORK/icon.icns")
fi

pyinstaller --noconfirm --windowed --name "$APP" "${ICON_ARGS[@]}" \
  --osx-bundle-identifier edu.mitadt.merismos.gesturepresenter \
  --collect-all mediapipe --collect-all pymupdf \
  --add-data "$PWD/$MODEL:gesture_presenter/models" \
  --workpath "$WORK/build" --distpath "$WORK/dist" --specpath "$WORK" \
  "$PWD/launcher.py" 2>&1 | grep -E "ERROR|Build complete" || true

BUNDLE="$WORK/dist/$APP.app"
[ -d "$BUNDLE" ] || { echo "Build failed - see output above"; exit 1; }
PLIST="$BUNDLE/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Add :NSCameraUsageDescription string 'Gesture Presenter uses the camera to read your hand gestures. Video never leaves your Mac.'" "$PLIST"
/usr/libexec/PlistBuddy -c "Add :NSAppleEventsUsageDescription string 'Used to open Keynote files.'" "$PLIST"
xattr -cr "$BUNDLE"
codesign --force --deep -s - "$BUNDLE"
codesign --verify --deep "$BUNDLE" && echo "Signature OK"

ditto "$BUNDLE" "$OUT/$APP.app"
ditto -c -k --keepParent "$BUNDLE" "../$APP.zip"
echo ""
tccutil reset Camera edu.mitadt.merismos.gesturepresenter >/dev/null 2>&1 || true  # new build = new camera prompt
echo "Installed: $OUT/$APP.app"
echo "Share:     $(cd .. && pwd)/$APP.zip"
