#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
npm --prefix renderer run build
CLANG_MODULE_CACHE_PATH="$PWD/.cache/clang" swift build --package-path native -c release --scratch-path "$PWD/.build/swift"
app="$PWD/build/Local Companion.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources/renderer"
cp .build/swift/release/Companion "$app/Contents/MacOS/Companion"
find "$app/Contents/Resources/renderer" -type f -delete
cp -R renderer/dist/. "$app/Contents/Resources/renderer/"
pwd > "$app/Contents/Resources/workspace.txt"
cat > "$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd"><plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>local.companion.desktop</string><key>CFBundleName</key><string>Local Companion</string><key>CFBundleDisplayName</key><string>Local Companion</string><key>CFBundleExecutable</key><string>Companion</string><key>CFBundlePackageType</key><string>APPL</string><key>CFBundleVersion</key><string>1</string><key>CFBundleShortVersionString</key><string>0.1.0</string><key>LSMinimumSystemVersion</key><string>15.0</string><key>NSMicrophoneUsageDescription</key><string>Listen when you explicitly use Talk or Hands-free. Local mode processes audio on this Mac. Codex mode sends audio to OpenAI services.</string><key>NSHighResolutionCapable</key><true/><key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict></dict></plist>
PLIST
codesign --force --deep --sign - "$app"
printf 'Built %s\n' "$app"
