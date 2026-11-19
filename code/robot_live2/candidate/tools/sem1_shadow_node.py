#!/usr/bin/env python3
"""SEM1 A3 launcher: the REAL e5_onboard entry point with --source ros_readonly in SHADOW mode, plus a hard run limit
counted from the moment the main loop starts (i.e. after the SigLIP model load/warm-up, which the supervisor already
watches). Run limit via SCI_SEM1_RUN_S (<= 40). Nothing here can actuate: shadow => request/arm off, no
--pantilt-execute, SCI_PANTILT_APPROVED unset, console URL 127.0.0.1:1 (refused)."""
import os
import sys
import threading
import time
from pathlib import Path

tree = Path(os.environ["SCI_WS"])
sys.path.insert(0, str(tree / "external" / "sci_event" / "e5_onboard"))
assert os.environ.get("SCI_E5_OBSERVE_MODE") == "shadow" and not os.environ.get("SCI_PANTILT_APPROVED"), "shadow only"
run_s = float(os.environ.get("SCI_SEM1_RUN_S", "35"))
assert run_s <= 40.0, "GPU block limited to 40 s"
import e5_onboard as OB  # noqa: E402

N = OB.N
_loop = OB.GatedNode.loop


def loop(self, src):
    t0 = time.monotonic()
    print(f"SEM1_LOOP_START mono={t0:.3f} run_s={run_s}", flush=True)

    def stop():
        time.sleep(run_s)
        print(f"SEM1_RUN_LIMIT reached after {time.monotonic() - t0:.1f}s", flush=True)
        N.STOP[0] = True
    threading.Thread(target=stop, daemon=True).start()
    return _loop(self, src)


OB.GatedNode.loop = loop
N.main()
