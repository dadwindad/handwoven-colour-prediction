#!/bin/zsh
# Build the macOS app (universal: Apple Silicon + Intel) from ../app.
#
#   ./macos/build.sh                 -> dist/สีผ้าทอโทเร.app, .zip and .dmg
#   SIGN_ID="Developer ID Application: …" ./macos/build.sh   sign for distribution
#
# Needs only the Xcode command line tools (swiftc, sips, iconutil, hdiutil).
set -euo pipefail

HERE=${0:A:h}
ROOT=${HERE:h}
WEB=$ROOT/app
DIST=$ROOT/dist
NAME="สีผ้าทอโทเร"
EXE="ToraiWeaveColour"
BUNDLE_ID="th.ac.bru.toraiweavecolour"
MIN_OS="12.0"                       # <dialog> and WKWebView features used by the app
SIGN_ID=${SIGN_ID:--}               # "-" = ad-hoc signature
# version follows the web app's cache version in app/sw.js (const VERSION = "v2" -> 1.2)
WEB_VERSION=$(sed -nE 's/^const VERSION = "v([0-9]+)".*/\1/p' $WEB/sw.js)
VERSION="1.${WEB_VERSION:-0}"

APP=$DIST/$NAME.app
BUILD=$DIST/build
rm -rf $APP $BUILD
mkdir -p $BUILD $APP/Contents/MacOS $APP/Contents/Resources

echo "• compiling ($VERSION)"
for arch in arm64 x86_64; do
  swiftc -O -target $arch-apple-macos$MIN_OS -o $BUILD/$EXE-$arch $HERE/main.swift
done
lipo -create -output $APP/Contents/MacOS/$EXE $BUILD/$EXE-arm64 $BUILD/$EXE-x86_64

echo "• copying web app"
rsync -a --exclude tests --exclude package.json --exclude node_modules --exclude '.*' $WEB/ $APP/Contents/Resources/web/

echo "• icon"
ICONSET=$BUILD/AppIcon.iconset
mkdir -p $ICONSET
for s in 16 32 128 256 512; do
  sips -z $s $s $WEB/icons/icon-512.png --out $ICONSET/icon_${s}x${s}.png >/dev/null
  d=$((s * 2))
  sips -z $d $d $WEB/icons/icon-512.png --out $ICONSET/icon_${s}x${s}@2x.png >/dev/null
done
iconutil -c icns $ICONSET -o $APP/Contents/Resources/AppIcon.icns

cat > $APP/Contents/Info.plist <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$NAME</string>
  <key>CFBundleDisplayName</key><string>$NAME</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleExecutable</key><string>$EXE</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleDevelopmentRegion</key><string>th</string>
  <key>LSMinimumSystemVersion</key><string>$MIN_OS</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSHumanReadableCopyright</key><string>ข้อมูลจากงานวิจัย มรภ.บุรีรัมย์ (2564)</string>
</dict>
</plist>
PLIST

echo "• signing ($SIGN_ID)"
if [[ $SIGN_ID == "-" ]]; then
  codesign --force --deep --sign - $APP
else
  codesign --force --deep --options runtime --timestamp --sign "$SIGN_ID" $APP
fi
codesign --verify --strict $APP

echo "• packaging"
rm -f "$DIST/$NAME-$VERSION.zip" "$DIST/$NAME-$VERSION.dmg"
ditto -c -k --keepParent $APP "$DIST/$NAME-$VERSION.zip"
STAGE=$BUILD/dmg
mkdir -p $STAGE
cp -R $APP $STAGE/
ln -s /Applications $STAGE/Applications
hdiutil create -quiet -volname "$NAME" -srcfolder $STAGE -fs HFS+ -format UDZO "$DIST/$NAME-$VERSION.dmg"
rm -rf $BUILD

echo "done: $APP"
ls -lh $DIST
