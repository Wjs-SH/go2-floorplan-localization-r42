#!/usr/bin/env bash
# pantilt_console_sci.sh [--execute] [--first-test] — start the SCI copy of aim_console (pan/tilt + IMX477 HTTP console :8088).
# DEFAULT = DRY RUN (prints checks + exact command). --execute additionally requires env SCI_PANTILT_APPROVED=1.
# PHYSICAL EFFECTS when executed:
#   1) opening /dev/ttyUSB0 resets the Nano -> servos jump to firmware pose (pan 110, tilt 160)
#   2) console then moves to calibrated HOME (pan 95.35, tilt 157.19)
#   3) EEPROM write is DISABLED here (--no-eeprom-sync); captures/logs go to ~/sci_ws/pantilt_out (not ~/pantilt_calib)
#   4) afterwards the servos move ONLY on /aim, /goto, /home, /aim_marker requests (SCI sends /aim only with approval)
# --first-test narrows the servo limits to HOME +-15 deg for the first supervised trial.
# STOP: bash scripts/pantilt_console_sci.sh --stop  (SIGINT to the SCI console only; servos hold their last position)
set -uo pipefail
WS=$HOME/sci_ws; PIDF=$WS/state/pids/pantilt_console.pid; OUT=$WS/pantilt_out
CAL=$HOME/pantilt_calib/configs/pantilt_calib_multi_20260928.json
MODE=dry; LIM=""
for a in "$@"; do case "$a" in --execute) MODE=exec;; --first-test) LIM="--pan-lim 80.4 110.4 --tilt-lim 142.2 172.2";; --stop) MODE=stop;; esac; done
if [ $MODE = stop ]; then
  [ -f $PIDF ] && p=$(cat $PIDF) && grep -q sci_legacy_e5/aim_console.py /proc/$p/cmdline 2>/dev/null && kill -INT $p && echo "SIGINT -> $p"; rm -f $PIDF; exit 0
fi
ok=1
[ -e /dev/ttyUSB0 ] && echo "[ok] /dev/ttyUSB0 present" || { echo "[BLOCK] no /dev/ttyUSB0"; ok=0; }
h=$(fuser /dev/ttyUSB0 2>/dev/null); [ -z "$h" ] && echo "[ok] /dev/ttyUSB0 not held" || { echo "[BLOCK] /dev/ttyUSB0 held by $h"; ok=0; }
ss -lnt | grep -q ":8088 " && { echo "[BLOCK] port 8088 in use"; ok=0; } || echo "[ok] port 8088 free"
[ -f $CAL ] && echo "[ok] calibration $CAL (read-only)" || { echo "[BLOCK] calibration missing"; ok=0; }
ls /sys/class/video4linux/*/name >/dev/null 2>&1 && grep -qi arducam /sys/class/video4linux/*/name && echo "[ok] Arducam IMX477 present" || { echo "[BLOCK] Arducam not found"; ok=0; }
CMD="python3 $WS/src/sci_legacy_e5/aim_console.py --config $CAL --no-eeprom-sync --out-root $OUT --use-depth --odom-topic /utlidar/robot_odom $LIM"
echo "command: $CMD"
[ $ok = 1 ] || exit 3
if [ $MODE != exec ]; then echo "DRY RUN — nothing started. Physical start needs: SCI_PANTILT_APPROVED=1 bash $0 --execute [--first-test]"; exit 0; fi
[ "${SCI_PANTILT_APPROVED:-0}" = 1 ] || { echo "[BLOCK] SCI_PANTILT_APPROVED!=1 (operator approval required)"; exit 4; }
mkdir -p $OUT $WS/state/pids; set +u; source /opt/ros/foxy/setup.bash; set -u
cd $OUT && nohup $CMD > $WS/logs/pantilt_console_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > $PIDF; echo "started console pid $(cat $PIDF); UI http://<robot>:8088/"
