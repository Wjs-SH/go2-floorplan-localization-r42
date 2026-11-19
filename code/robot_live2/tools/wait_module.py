#!/usr/bin/env python3
"""wait_module.py — wait for LIVE2-owned module supervisors to end, by identity (PID + /proc starttime), with a timeout.

  wait_module.py MODULE [MODULE ...] --timeout-s 3600 [--poll-s 5]

Never matches process command lines (a `pgrep -f NAME` waiter matches its own shell and never ends), never signals anything.
Exit 0 = all ended, 2 = timeout (processes left running untouched), 3 = no owner record.
"""
import argparse
import json
import sys
import time
from pathlib import Path

OWNED = Path(__file__).resolve().parent.parent / "runtime" / "owned"


def starttime(pid):
    try:
        s = Path(f"/proc/{pid}/stat").read_text()
        return int(s[s.rindex(")") + 2:].split()[19])   # field 22 (starttime, clock ticks since boot)
    except (OSError, ValueError, IndexError):
        return None


def alive(rec):
    return rec.get("pid") is not None and starttime(rec["pid"]) == rec.get("start")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modules", nargs="+")
    ap.add_argument("--timeout-s", type=float, required=True)
    ap.add_argument("--poll-s", type=float, default=5.0)
    a = ap.parse_args()
    recs = {}
    for m in a.modules:
        p = OWNED / f"{m}.json"
        if not p.exists():
            print(f"{m}: no owner record", flush=True)
            sys.exit(3)
        recs[m] = json.loads(p.read_text())
    t_end = time.time() + a.timeout_s
    left = {m for m, r in recs.items() if alive(r)}
    for m in sorted(set(recs) - left):
        print(f"{m}: already ended (pid {recs[m]['pid']} gone or reused)", flush=True)
    while left and time.time() < t_end:
        time.sleep(a.poll_s)
        for m in sorted(left):
            if not alive(recs[m]):
                left.discard(m)
                sup = OWNED / f"{m}.sup_state.json"
                st = json.loads(sup.read_text()) if sup.exists() else {}
                print(f"{m}: ended {st.get('status')} {st.get('end_reason')}", flush=True)
    if left:
        print(f"TIMEOUT after {a.timeout_s:.0f}s, still running (not touched): {sorted(left)}", flush=True)
        sys.exit(2)
    print("all ended", flush=True)


if __name__ == "__main__":
    main()
