#!/bin/sh
# Isolated Android10 runtime for verification; leaves user's existing AVDs alone.
set -eu
case "${1:---headless}" in
    --headless) TASK_WINDOW_FLAG=-no-window ;;
    --window) TASK_WINDOW_FLAG= ;;
    *) echo "Usage: sh tools/android10_emulator.sh [--headless|--window]" >&2; exit 2 ;;
esac
TASK_ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
TASK_RUNTIME="$TASK_ROOT/.android-runtime"
TASK_SDK=${ANDROID_SDK_ROOT:-${ANDROID_HOME:-"$HOME/Library/Android/sdk"}}
mkdir -p "$TASK_RUNTIME"
if [ ! -f "$TASK_RUNTIME/image/x86_64/source.properties" ]; then
    curl -L --fail --max-time 600 https://dl.google.com/android/repository/sys-img/google_apis/x86_64-29_r13.zip -o "$TASK_RUNTIME/image.zip"
    python3 - "$TASK_RUNTIME" <<'PY'
import hashlib,sys,zipfile
from pathlib import Path
root=Path(sys.argv[1]); archive=root/'image.zip'; digest=hashlib.sha1()
with archive.open('rb') as f:
 for data in iter(lambda:f.read(1048576),b''): digest.update(data)
if digest.hexdigest()!='50c2953ab13f312ed797a961577ee74a911d087d': raise RuntimeError('Android image checksum mismatch')
with zipfile.ZipFile(archive) as z: z.extractall(root/'image')
archive.unlink()
PY
fi
mkdir -p "$TASK_RUNTIME/avds/DSwinIR_API29.avd"
cat > "$TASK_RUNTIME/avds/DSwinIR_API29.ini" <<EOF
avd.ini.encoding=UTF-8
path=$TASK_RUNTIME/avds/DSwinIR_API29.avd
target=android-29
EOF
cat > "$TASK_RUNTIME/avds/DSwinIR_API29.avd/config.ini" <<EOF
AvdId=DSwinIR_API29
avd.ini.encoding=UTF-8
abi.type=x86_64
hw.cpu.arch=x86_64
hw.cpu.ncore=2
hw.ramSize=2048
hw.lcd.width=720
hw.lcd.height=1280
hw.lcd.density=320
hw.gpu.enabled=yes
hw.gpu.mode=swiftshader_indirect
hw.keyboard=yes
hw.audioInput=no
hw.sdCard=no
image.sysdir.1=$TASK_RUNTIME/image/x86_64/
tag.id=google_apis
PlayStore.enabled=false
disk.dataPartition.size=2G
showDeviceFrame=no
EOF
export ANDROID_AVD_HOME="$TASK_RUNTIME/avds"
export ANDROID_SDK_ROOT="$TASK_SDK"
# Avoid the Vulkan presentation path on the Intel Mac verification runtime.
# This controls emulator rendering; it does not enable a LiteRT GPU delegate.
exec "$TASK_SDK/emulator/emulator" -avd DSwinIR_API29 -no-snapshot -no-audio ${TASK_WINDOW_FLAG:+$TASK_WINDOW_FLAG} -no-boot-anim -gpu swiftshader_indirect -feature -Vulkan -memory 2048 -cores 2
