#!/usr/bin/env python3
"""SEM1 pre-launch supervisor: becomes ready BEFORE the child (model load included) and owns ONLY its child tree.

  start gates : max sensor temp < --start-temp (72), MemAvailable >= --start-mb, protected processes alive,
                no other siglip_worker (no duplicate GPU worker), optional approval record valid (not expired).
  run checks  : every 0.5 s: temp >= --abort-temp (80), MemAvailable < --abort-mb (2500), protected process vanished,
                duration > --max-s, child gone. Heartbeat file written each cycle (a stale heartbeat = supervisor lost).
  ownership   : child started in its own session; the supervisor tracks the child PID + /proc starttime + PGID and every
                DESCENDANT by walking the PPID tree. A descendant that left the PGID (setsid / double fork) is reported
                as ESCAPED_PGID and still terminated by PID (identity verified by starttime). Nothing else is ever signalled.
  stop        : SIGINT to the PGID (+ escaped descendants), wait, then SIGKILL only those.
Records: --log JSONL (samples + events) and a final summary line.
"""
import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time

HZ = os.sysconf("SC_CLK_TCK")


def temps():
    v = []
    for z in glob.glob("/sys/devices/virtual/thermal/thermal_zone*/temp"):
        try:
            v.append(int(open(z).read()) / 1000.0)
        except (OSError, ValueError):
            pass
    return max(v) if v else None


def avail_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return None


def stat(pid):
    try:
        s = open(f"/proc/{pid}/stat").read()
        rest = s[s.rindex(")") + 2:].split()
        return dict(ppid=int(rest[1]), pgid=int(rest[2]), utime=int(rest[11]), stime=int(rest[12]), start=int(rest[19]),
                    rss_mb=int(rest[21]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20)
    except (OSError, ValueError, IndexError):
        return None


def cmdline(pid):
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def all_pids():
    return [int(p.split("/")[2]) for p in glob.glob("/proc/[0-9]*")]


def descendants(root):
    kids, st = {}, {}
    for p in all_pids():
        s = stat(p)
        if s:
            st[p] = s
            kids.setdefault(s["ppid"], []).append(p)
    out, todo = [], [root]
    while todo:
        x = todo.pop()
        for c in kids.get(x, []):
            out.append(c)
            todo.append(c)
    return out, st


def _ancestors():
    out, p = set(), os.getpid()
    while p > 1:
        out.add(p)
        s = stat(p)
        if not s:
            break
        p = s["ppid"]
    return out


def find_pids(pattern):
    """PIDs whose argv contains a token ENDING with pattern (a shell command line merely mentioning it does not match);
    the supervisor and its own ancestors are excluded."""
    anc, out = _ancestors(), []
    for p in all_pids():
        if p in anc:
            continue
        try:
            argv = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if any(t.decode("utf-8", "replace").endswith(pattern) for t in argv if t):
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--max-s", type=float, required=True)
    ap.add_argument("--start-temp", type=float, default=72.0)
    ap.add_argument("--abort-temp", type=float, default=80.0)
    ap.add_argument("--start-mb", type=float, default=4300.0)
    ap.add_argument("--abort-mb", type=float, default=2500.0)
    ap.add_argument("--protect", action="append", default=[], help="cmdline substring of a protected process that must stay alive")
    ap.add_argument("--no-dup", action="append", default=[], help="cmdline substring that must NOT already run (e.g. siglip_worker.py)")
    ap.add_argument("--approval", default=None, help="approval record json with expires_epoch")
    ap.add_argument("--wait-start-s", type=float, default=600.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    log = open(a.log, "a")

    def ev(**k):
        k.setdefault("t", time.time())
        k.setdefault("mono", time.monotonic())
        log.write(json.dumps(k) + "\n")
        log.flush()

    ev(kind="SUPERVISOR_READY", pid=os.getpid(), cmd=cmd, args={k: v for k, v in vars(a).items() if k != "cmd"})
    if a.approval:
        rec = json.load(open(a.approval))
        if time.time() > float(rec["expires_epoch"]):
            ev(kind="REFUSED", why="APPROVAL_EXPIRED")
            print("REFUSED approval expired")
            return 5
    prot0 = {p: find_pids(p) for p in a.protect}
    t0w = time.time()
    while True:
        t, m = temps(), avail_mb()
        dups = {d: find_pids(d) for d in a.no_dup}
        missing = [p for p, v in prot0.items() if not v]
        if not missing and t is not None and t < a.start_temp and m >= a.start_mb and not any(dups.values()):
            break
        if time.time() - t0w > a.wait_start_s:
            ev(kind="REFUSED", why="START_GATE", temp=t, avail=m, dups=dups, missing_protected=missing)
            print("REFUSED start gate", t, m, dups, missing)
            return 4
        time.sleep(2.0)
    ev(kind="START_GATE_OK", temp=t, avail=m, protected={p: v for p, v in prot0.items()}, waited_s=round(time.time() - t0w, 1))
    child = subprocess.Popen(cmd, start_new_session=True)
    cs = stat(child.pid)
    ident = dict(pid=child.pid, pgid=os.getpgid(child.pid), start=cs["start"] if cs else None)
    ev(kind="CHILD_STARTED", **ident)
    t_start, status, peak = time.monotonic(), "COMPLETED", dict(temp=0.0, avail_min=10 ** 9, rss_mb=0.0)
    known, escaped, last_cpu = {}, set(), {}
    while True:
        rc = child.poll()
        desc, st = descendants(child.pid)
        for d in desc:
            s = st.get(d)
            if s and d not in known:
                known[d] = s["start"]
                ev(kind="DESCENDANT", pid=d, pgid=s["pgid"], cmd=cmdline(d)[:160])
            if s and s["pgid"] != ident["pgid"] and d not in escaped:
                escaped.add(d)
                ev(kind="ESCAPED_PGID", pid=d, pgid=s["pgid"], cmd=cmdline(d)[:160])
        t, m = temps(), avail_mb()
        rss = sum((st.get(d) or {}).get("rss_mb", 0) for d in [child.pid] + desc)
        peak.update(temp=max(peak["temp"], t or 0), avail_min=min(peak["avail_min"], m or 0), rss_mb=max(peak["rss_mb"], rss))
        prot_now = {p: find_pids(p) for p in a.protect}
        prot_cpu = {}
        for p, pids in prot_now.items():
            for pid in pids:
                s = stat(pid)
                if s:
                    tot = s["utime"] + s["stime"]
                    if pid in last_cpu:          # sum over all matching PIDs (launch wrappers match the pattern too)
                        prot_cpu[p] = round(prot_cpu.get(p, 0.0) + (tot - last_cpu[pid][0]) / HZ / max(1e-6, time.monotonic() - last_cpu[pid][1]) * 100, 1)
                    last_cpu[pid] = (tot, time.monotonic())
        ev(kind="SAMPLE", temp=t, avail=m, own_rss_mb=round(rss, 1), n_desc=len(desc), protected_cpu=prot_cpu,
           elapsed=round(time.monotonic() - t_start, 2))
        reason = None
        if rc is not None and not desc:
            status = f"CHILD_EXITED rc={rc}"
            break
        if t is not None and t >= a.abort_temp:
            reason = f"ABORT_TEMP {t}"
        elif m is not None and m < a.abort_mb:
            reason = f"ABORT_MEM {m}"
        elif any(not prot_now[p] for p in a.protect) or any(set(prot_now[p]) != set(prot0[p]) for p in a.protect):
            reason = f"PROTECTED_PROCESS_CHANGED {prot_now}"
        elif time.monotonic() - t_start > a.max_s:
            reason = "MAX_DURATION"
        if reason:
            status = reason
            targets = [d for d in desc if d in escaped and stat(d) and stat(d)["start"] == known.get(d)]
            try:
                os.killpg(ident["pgid"], signal.SIGINT)
            except ProcessLookupError:
                pass
            for d in targets:
                try:
                    os.kill(d, signal.SIGINT)
                except ProcessLookupError:
                    pass
            t_k = time.monotonic()
            while time.monotonic() - t_k < 10:
                desc2, _ = descendants(child.pid)
                if child.poll() is not None and not [d for d in desc2 + targets if stat(d)]:
                    break
                time.sleep(0.2)
            left = [d for d in targets + descendants(child.pid)[0] if stat(d) and stat(d)["start"] == known.get(d)]
            if child.poll() is None or left:
                try:
                    os.killpg(ident["pgid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
                for d in left:
                    try:
                        os.kill(d, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            child.wait()
            break
        time.sleep(0.5)
    def alive(p, s0):
        s = stat(p)
        if not s or s["start"] != s0:
            return False
        try:
            return open(f"/proc/{p}/stat").read().split(") ")[1][0] != "Z"     # a zombie is already dead
        except (OSError, IndexError):
            return False
    t_l = time.monotonic()
    leftovers = [p for p, s0 in known.items() if alive(p, s0)]
    while leftovers and time.monotonic() - t_l < 3.0:
        time.sleep(0.2)
        leftovers = [p for p, s0 in known.items() if alive(p, s0)]
    ev(kind="SUMMARY", status=status, child=ident, peak=peak, escaped=sorted(escaped), leftovers=leftovers,
       duration_s=round(time.monotonic() - t_start, 1))
    print(json.dumps(dict(status=status, peak=peak, escaped=sorted(escaped), leftovers=leftovers)))
    return 0 if not leftovers else 6


if __name__ == "__main__":
    sys.exit(main())
