#!/bin/sh
# Build, start the project's visible emulator and open the app from any IDE.
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
TASK_SDK=${ANDROID_SDK_ROOT:-${ANDROID_HOME:-"$HOME/Library/Android/sdk"}}
TASK_ADB="$TASK_SDK/platform-tools/adb"
TASK_APK="$TASK_ROOT/android/app/build/outputs/apk/debug/app-debug.apk"
TASK_MODE=${1:---build}
case "$TASK_MODE" in --build|--apk) ;; *) echo "Usage: sh tools/run_android.sh [--build|--apk]" >&2; exit 2;; esac
if [ ! -x "$TASK_ADB" ]; then echo "Không tìm thấy Android SDK: $TASK_SDK" >&2; exit 1; fi
if [ "$TASK_MODE" = --build ]; then
    echo "Đang build APK từ mã nguồn hiện tại…"
    ANDROID_HOME="$TASK_SDK" "$TASK_ROOT/android/gradlew" -p "$TASK_ROOT/android" :app:assembleDebug --console=plain
fi
if [ ! -f "$TASK_APK" ]; then echo "Chưa có APK. Chạy lại với --build." >&2; exit 1; fi
"$TASK_ADB" start-server
find_device() {
    for serial in $("$TASK_ADB" devices | awk '$1 ~ /^emulator-/ && $2 == "device" {print $1}'); do
        name=$("$TASK_ADB" -s "$serial" emu avd name 2>/dev/null | tr -d '\r' | head -n 1)
        if [ "$name" = DSwinIR_API29 ]; then echo "$serial"; return; fi
    done
}
TASK_DEVICE=$(find_device)
if [ -z "$TASK_DEVICE" ]; then
    mkdir -p "$TASK_ROOT/.android-runtime"
    echo "Đang mở cửa sổ máy ảo Android 10…"
    # Separate session: IDE/tool terminals can terminate their process group
    # when this task exits. nohup alone only protects against SIGHUP.
    python3 - "$TASK_ROOT" <<'PY'
import subprocess,sys
from pathlib import Path
root=Path(sys.argv[1])
with (root/'.android-runtime/emulator-window.log').open('ab') as log:
    subprocess.Popen(['sh',str(root/'tools/android10_emulator.sh'),'--window'],
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True,close_fds=True)
PY
fi
TASK_ATTEMPT=0
while [ "$TASK_ATTEMPT" -lt 180 ]; do
    TASK_DEVICE=$(find_device)
    if [ -n "$TASK_DEVICE" ]; then
        TASK_BOOT=$("$TASK_ADB" -s "$TASK_DEVICE" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')
        if [ "$TASK_BOOT" = 1 ]; then break; fi
    fi
    TASK_ATTEMPT=$((TASK_ATTEMPT + 1))
    if [ $((TASK_ATTEMPT % 15)) -eq 0 ]; then echo "Đang chờ Android khởi động…"; fi
    sleep 2
done
if [ -z "$TASK_DEVICE" ] || [ "${TASK_BOOT:-}" != 1 ]; then
    echo "Máy ảo chưa sẵn sàng. Xem .android-runtime/emulator-window.log" >&2; exit 1
fi
"$TASK_ADB" -s "$TASK_DEVICE" install -r "$TASK_APK"
"$TASK_ADB" -s "$TASK_DEVICE" shell am start -W -n vn.edu.dswinir/.MainActivity
echo "Đã mở DSwinIR Lab trên $TASK_DEVICE. Bạn thao tác trong cửa sổ Android Emulator."
