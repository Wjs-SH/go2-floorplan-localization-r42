#!/usr/bin/env python3
"""live2 — operator shortcut for the LIVE-FIELD2 web console (8770). It manages ONLY the web console; research modules
(FAST-LIO, E2/E3/E4, SigLIP, E5–E7, pan-tilt, IMX477, HOME) are never started here — they start from the web
[실험 시작] under approval A. Every action goes through scripts/live2_ctl.sh (owner record = PID + starttime).

  live2 start | status | stop | restart | logs [-n N] [-f]
  live2 serve        (systemd user service live2-console.service: wait for Tailscale, keep the WEB console up)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

L2 = Path("/home/unitree/sci_ws/repairs/livefield2_20261018_151606")
CTL = L2 / "scripts" / "live2_ctl.sh"
SVC = "live2-console.service"
PORT = 8770
RUN_S = 43200                               # console cap per run (12 h); the service watcher starts the next one
STOP_FLAG = L2 / "runtime" / "ctl" / "serve.stop"
SERVE_LOG = L2 / "logs" / "live2_serve.log"
ACTIVE = ("STARTING", "RUNNING", "PARTIAL", "PAUSED", "STOPPING")
MODULES = ("fastlio", "lio", "lio2", "hub", "observe", "rec", "e2", "e34", "replay", "web")
os.environ["LIVE1_ROOT"] = str(L2)
sys.path.insert(0, str(L2 / "candidate" / "live1"))
import live1_common as C  # noqa: E402


def sh(cmd, timeout=120):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip()


def ctl(*args):
    return sh([str(CTL), *args])


def console_alive():
    rec = C.owner_record("console")
    return C.owner_alive(rec), rec


def tailscale_ip():
    try:
        return subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5).stdout.split()[0]
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return None


def busy():
    """Why the console must not be stopped now (None = safe)."""
    x = C.read_json(L2 / "runtime" / "ctl" / "experiment.json") or {}
    alive = [m for m in MODULES if C.owner_alive(C.owner_record(m))]
    if x.get("state") in ACTIVE:
        return f"실험 {x.get('exp_id')} 진행 중 ({x.get('state')}) — 태블릿에서 [실험 종료 · 기록 저장] 후 다시 하십시오"
    if alive:
        return f"연구 모듈 실행 중 {alive} — 실험 종료 후 다시 하십시오"
    return None


def svc(*args):
    return sh(["systemctl", "--user", *args, SVC], timeout=60)


def svc_installed():
    return (Path.home() / ".config" / "systemd" / "user" / SVC).exists()


def start_console():
    ok, rec = console_alive()
    if ok:
        return 0, f"웹 콘솔 이미 실행 중 (pid {rec['pid']}) — 중복 실행하지 않음"
    rc, out = ctl("start", "console", "--run-s", str(RUN_S))
    return rc, out.splitlines()[-1] if out else ""


def cmd_start():
    if STOP_FLAG.exists():
        STOP_FLAG.unlink()
    if svc_installed():
        rc, out = svc("start")
        time.sleep(2)
        ok, rec = console_alive()
        print(f"service start rc={rc} {out}".strip())
        print(f"웹 콘솔: {'실행 중 pid ' + str(rec['pid']) if ok else '아직 기동 전 (Tailscale 대기 중일 수 있음 — live2 logs)'}")
        return 0 if rc == 0 else rc
    rc, msg = start_console()
    print(msg)
    return rc


def cmd_stop():
    why = busy()
    if why:
        print("REFUSED:", why)
        return 3
    STOP_FLAG.parent.mkdir(parents=True, exist_ok=True)
    STOP_FLAG.write_text(C.now_wall())                           # tells the service watcher not to bring it back
    if svc_installed() and svc("is-active")[1] == "active":
        rc, out = svc("stop")
        print(f"service stopped rc={rc} {out}".strip())
    ok, _ = console_alive()
    if ok:
        rc, out = ctl("stop", "console")
        print(out.splitlines()[-1] if out else rc)
    print("웹 콘솔 정지됨 (연구 모듈 없음). 다시 켜기: live2 start")
    return 0


def cmd_restart():
    x = C.read_json(L2 / "runtime" / "ctl" / "experiment.json") or {}
    if x.get("state") in ("STARTING", "STOPPING"):
        print(f"REFUSED: 콘솔이 모듈을 켜거나 끄는 중 ({x.get('state')}) — 끝난 뒤 다시 하십시오")
        return 3
    ok, _ = console_alive()
    if ok:
        rc, out = ctl("stop", "console")
        print(out.splitlines()[-1] if out else rc)
    rc, msg = start_console()
    print(msg)
    if x.get("state") in ACTIVE:
        print(f"참고: 실험 {x.get('exp_id')} 는 그대로 진행 중입니다 (연구 모듈은 콘솔과 분리되어 영향 없음).")
    return rc


def cmd_status():
    ok, rec = console_alive()
    listen = any(f":{PORT} " in ln for ln in sh(["ss", "-ltn"])[1].splitlines())
    ap = C.approval_status("A")
    exp = ap.get("expires_epoch")
    x = C.read_json(L2 / "runtime" / "ctl" / "experiment.json") or {}
    alive = [m for m in MODULES if C.owner_alive(C.owner_record(m))]
    ts = tailscale_ip()
    r = C.resources()
    print(f"boot_id        {C.boot_id()}")
    print(f"웹 콘솔        {'실행 중 pid ' + str(rec['pid']) if ok else '정지'} · 포트 {PORT} {'LISTEN' if listen else '닫힘'}")
    if svc_installed():
        print(f"자동 실행      {SVC}: {svc('is-active')[1]} / {svc('is-enabled')[1]}")
    else:
        print(f"자동 실행      {SVC} 미설치")
    print(f"Tailscale      {ts or '주소 없음'}  →  접속 주소 http://{ts or '<ROBOT_TAILSCALE_IP>'}:{PORT}/")
    if ap.get("kind") == "STANDING":            # standing research authorization: no expiry (2026-10-20 crash: float(None))
        print(f"연구 실행 권한 {'상시 허용' if ap['valid'] else str(ap.get('state') or ap['why'])} · grant {ap.get('grant_id')} · "
              f"{','.join(ap.get('modules') or [])} · 실험당 최대 {round((ap.get('experiment_max_s') or 0) / 60)}분")
    else:
        left = '' if not exp else str(round((float(exp) - time.time()) / 60)) + '분 남음 · '
        print(f"승인 A         {'유효 · ' + left + ','.join(ap.get('modules') or []) if ap['valid'] else '없음/무효 (' + str(ap['why']) + ')'}")
    print(f"실험           {x.get('exp_id') or '-'} {x.get('state') or 'IDLE'}")
    print(f"연구 모듈      {alive or '실행 중인 모듈 없음'}")
    print(f"온도·메모리    {r['max_temp_c']:.1f} °C · MemAvailable {r['mem_available_mb']:.0f} MB")
    return 0


def cmd_logs(argv):
    n = int(argv[argv.index("-n") + 1]) if "-n" in argv else 30
    logs = sorted((L2 / "logs").glob("console_*.out"))
    if logs:
        print(f"== 웹 콘솔 로그 {logs[-1]}")
        print("\n".join(logs[-1].read_text(errors="replace").splitlines()[-n:]))
    if SERVE_LOG.exists():
        print(f"== 자동 실행 감시 로그 {SERVE_LOG}")
        print("\n".join(SERVE_LOG.read_text(errors="replace").splitlines()[-n:]))
    a = L2 / "runtime" / "ctl" / "audit.jsonl"
    if a.exists():
        print(f"== 웹 요청 기록 {a}")
        for ln in a.read_text(errors="replace").splitlines()[-n:]:
            try:
                d = json.loads(ln)
                print(f"{d.get('time')}  {d.get('ip')}  {d.get('action')}  {d.get('code')}  {d.get('why') or ''}")
            except ValueError:
                pass
    x = C.read_json(L2 / "runtime" / "ctl" / "experiment.json") or {}
    if x.get("timeline"):
        print(f"== 실험 {x.get('exp_id')} 사건")
        for e in x["timeline"][-n:]:
            print(f"{e.get('time')}  {e.get('kind')}  {json.dumps({k: v for k, v in e.items() if k not in ('wall', 'time', 'kind')}, ensure_ascii=False)[:120]}")
    if "-f" in argv and logs:
        os.execvp("tail", ["tail", "-n", "0", "-F", str(logs[-1]), str(a)])
    return 0


def log(msg):
    SERVE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SERVE_LOG, "a") as f:
        f.write(f"{C.now_wall()} {msg}\n")


def cmd_serve():
    """systemd user service body: wait for Tailscale (max 180 s), keep the WEB console up. Never starts a research module."""
    if STOP_FLAG.exists():
        STOP_FLAG.unlink()
    log(f"serve start boot={C.boot_id()}")
    t0 = time.time()
    while not tailscale_ip() and time.time() - t0 < 180:
        time.sleep(3)
    ts = tailscale_ip()
    log(f"tailscale {'ready ' + ts if ts else 'NOT ready after 180 s (console starts; tailnet access verified once it is up)'}")
    started_with_ts = bool(ts)
    while True:
        if STOP_FLAG.exists():
            log("stop flag -> serve exits")
            return 0
        ok, rec = console_alive()
        if not ok:
            rc, msg = start_console()
            started_with_ts = bool(tailscale_ip())
            log(f"console start rc={rc} {msg}")
        elif not started_with_ts and tailscale_ip() and not busy():
            log("tailscale came up after console start -> restart console once (idle) so tailnet devices are recognised")
            ctl("stop", "console")
            continue
        time.sleep(10)


def cmd_svc_stop():
    """ExecStop: stop the web console only when no experiment runs; otherwise leave it (KillMode=process keeps it)."""
    STOP_FLAG.parent.mkdir(parents=True, exist_ok=True)
    STOP_FLAG.write_text(C.now_wall())
    why = busy()
    if why:
        log(f"service stop: console LEFT RUNNING ({why})")
        return 0
    ok, _ = console_alive()
    if ok:
        ctl("stop", "console")
    log("service stop: console stopped")
    return 0


def main():
    a = sys.argv[1:] or ["status"]
    return dict(start=cmd_start, status=cmd_status, stop=cmd_stop, restart=cmd_restart, serve=cmd_serve,
                _svc_stop=cmd_svc_stop).get(a[0], lambda: cmd_logs(a) if a[0] == "logs" else (print(__doc__), 2)[1])()


if __name__ == "__main__":
    sys.exit(main() or 0)
