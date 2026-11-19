#!/usr/bin/env python3
"""console.py — LIVE2 operator console on the 8770 web: the LIVE1/LIVE2 read-only panel + an allow-listed control API.

What the operator can do from the browser (nothing else exists — there is no command / shell / path input anywhere):
  start experiment · pause · resume · end + archive · E2 initial run · E2 re-search · SigLIP on/off · GT clicks · detection labels
Every module start/stop goes through the SAME ctl.py the operator would run by hand (fixed argv, values from allow-lists),
so approval A (boot-bound, expiry, module list, per-module run limit), resource gates, single-instance and PID+starttime
ownership are enforced by ctl exactly as before. The console never sends a robot / pan-tilt / HOME / focus command, never
subscribes ROS and never touches ROPUS services.

Security (UI V2): account "recs" + operator-chosen password (scrypt, live2/auth.py), persistent device sessions (30 days
with "remember", server keeps only hashes) -> HttpOnly SameSite=Strict cookie + HMAC CSRF header; Origin / Host checks;
JSON bodies only; request_id de-duplication; one control job at a time; one experiment owned by one session (takeover
needs the password again); login / control only from loopback or the Tailscale network (LAN = read-only view); every
request audited. The legacy operator-token login stays until the new login is verified on the real tablet
(runtime/ctl/auth/legacy_retired.json disables it). A login never implies approval A.

Device mode (default, development): no login screen at all. The page and read-only data are served to loopback and to
Tailscale peers that `tailscale whois` identifies as nodes of the robot owner's tailnet; CONTROL is allowed only for
loopback and for allow-listed Tailscale nodes (runtime/ctl/auth/trusted_devices.json, by node StableID, e.g. tab-s8).
A control request must arrive on the robot's Tailscale address, the peer IP must belong to the whois node (WireGuard binds
the source address to the node key), the auto-issued device session must belong to that node, plus CSRF / Origin / Host /
request_id / single-owner checks. LAN (192.168.x) gets nothing. The account login (auth.py) is kept: mode.json {"mode":"account"}.

Idle rule (UI V2): while no experiment is active the live endpoints return NO_EXPERIMENT and no image; the previous run is
only reachable through /api/history (labelled HISTORICAL).

Pause = new ANALYSIS is held (E2 no new floor / pass, E3 no new keyframe, SigLIP stopped and restarted on resume).
FAST-LIO, LiDAR accumulation and the sensor hub keep running. Nothing is sent to the Go2.

GT clicks and detection labels go to <root>/evaluation/<exp_id>/ — an evaluation store that no E2..E7 process reads.
"""
import argparse
import hashlib
import ipaddress
import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = HERE.parent
sys.path.insert(0, str(CAND / "live1"))
sys.path.insert(0, str(HERE))
import live1_common as C  # noqa: E402
import live1_web as WEB  # noqa: E402
import depth_freshness as DF  # noqa: E402  (path added by live1_web)
import auth as AUTH  # noqa: E402
import exp_archive as EA  # noqa: E402

L = C.LROOT
CTL = CAND / "live1" / "ctl.py"
CTL_DIR = C.RUNTIME / "ctl"
EXP_FILE = CTL_DIR / "experiment.json"
AUDIT = CTL_DIR / "audit.jsonl"
EXPERIMENTS = L / "experiments"
EVAL = L / "evaluation"                         # GT + detection labels; never an input of E2..E7
PENDING_START = EVAL / "_pending" / "start_gt.json"   # start GT marked before [실험 시작]; moved into evaluation/<exp> at start
REQUIRE_START_GT = os.environ.get("LIVE2_REQUIRE_START_GT", "1") == "1"
HUB = (L / "runtime" / "fake_hub") if C.FAKE else Path("/dev/shm/live1_hub")
STORE = (L / "runtime" / "fake_store") if C.FAKE else Path("/dev/shm/live2_store")      # same as ctl.STORE_LIVE
SHM = C.RUNTIME / "e5_shm"
WEB.HUB = HUB
SEM_ROOT = Path.home() / "sci_ws" / "maps"
SEM_DIR = dict(kci=lambda f: SEM_ROOT / "kci_handoff" / f"{f}F", sw=lambda f: SEM_ROOT / f"S{f}F", gong=lambda f: SEM_ROOT / f"G{f}F")
BUILDINGS = ("kci", "sw", "gong")
START_ORDER = ("fastlio", "lio2", "hub", "rec", "observe", "e34", "e6", "e2")
STOP_ORDER = ("e6", "observe", "e2", "e34", "rec", "hub", "lio2", "fastlio")   # e6 first: HOME while odometry still runs
# R36: the two experiment modes, set once at the start and recorded. loc1 never starts E5/SigLIP, the E6 bridge or the OCR/VLM flow; it never asks the robot to stop and never turns into loc2 by itself.
# R39: one mode. loc1 / loc2 told the algorithm where it was; the robot now decides by itself (auto_fsm.py). A zone name may only label a finished run in the analysis.
AUTO_MODE = "AUTO_INITIAL_GLOBAL"
LOC_MODES = {AUTO_MODE: dict(name="auto", e5_auto_stop=True, pantilt=True, ocr_vlm=True, structure=True, skip=())}
CRITICAL = ("fastlio", "lio2")                  # without these nothing downstream is meaningful
APPROVAL_MODULE = dict(rec=None, e6=None)       # rec = file-only recorder (no approval); every other module needs approval A
# e6 (E8 automatic E6: student aim console + precision bridge) is gated by HARDWARE approval B ('pantilt', this boot) instead:
# without B it is OFF (observe-only E5 stops, no pan-tilt / focus), never a failure of the experiment.
LOGIN_FAILS, LOCKOUT_S = 5, 60
UI_V2 = HERE / "web_v2" / "index.html"
UI_LEGACY = CAND / "live1" / "web" / "index.html"
TAILNET = ipaddress.ip_network("100.64.0.0/10")
TAILNET6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")
COOKIE = "l2s"
MODE_FILE = CTL_DIR / "auth" / "mode.json"
TRUST_FILE = CTL_DIR / "auth" / "trusted_devices.json"


def auth_mode():
    return (C.read_json(MODE_FILE) or {}).get("mode", "device")


class DeviceGate:
    """Who is on the other end of a connection, without asking the user for anything (device mode)."""

    def __init__(self):
        self.cache, self.lock = {}, threading.Lock()
        self.self_ips, self.owner = set(), None
        self.refresh_self()

    def refresh_self(self):
        try:
            d = json.loads(subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True, timeout=5).stdout)
            self.self_ips = set(d["Self"].get("TailscaleIPs") or [])
            self.owner = ((d.get("User") or {}).get(str(d["Self"].get("UserID"))) or {}).get("LoginName") or \
                (d.get("CurrentTailnet") or {}).get("Name")
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
            self.self_ips, self.owner = set(), None

    def whois(self, ip):
        now = time.time()
        with self.lock:
            hit = self.cache.get(ip)
            if hit and now - hit[0] < 30:
                return hit[1]
        try:
            d = json.loads(subprocess.run(["tailscale", "whois", "--json", ip], capture_output=True, text=True, timeout=3).stdout)
            n, u = d.get("Node") or {}, d.get("UserProfile") or {}
            w = dict(stable_id=n.get("StableID"), name=(n.get("Name") or "").split(".")[0], fqdn=(n.get("Name") or "").rstrip("."),
                     user=u.get("LoginName"), tags=n.get("Tags") or [], addrs=[a.split("/")[0] for a in n.get("Addresses") or []])
        except (OSError, ValueError, subprocess.TimeoutExpired):
            w = None
        with self.lock:
            self.cache[ip] = (now, w)
        return w

    def trusted(self):
        return {d["stable_id"]: d for d in (C.read_json(TRUST_FILE) or {}).get("devices", []) if d.get("stable_id")}

    def identify(self, client_ip, local_ip):
        try:
            ip = ipaddress.ip_address(client_ip)
        except ValueError:
            return dict(kind="unknown", read=False, control=False, why="BAD_PEER_ADDRESS")
        if ip.is_loopback:
            return dict(kind="loopback", read=True, control=True, actor="dev:loopback", name="로봇 자체 (localhost)")
        if ip in TAILNET or ip in TAILNET6:
            if local_ip not in self.self_ips:
                return dict(kind="tailnet", read=False, control=False, why="NOT_RECEIVED_ON_TAILSCALE_ADDRESS")
            w = self.whois(client_ip)
            if not w or client_ip not in w["addrs"] or not w["stable_id"]:
                return dict(kind="tailnet", read=False, control=False, why="TAILSCALE_WHOIS_FAILED")
            same = bool(self.owner) and w["user"] == self.owner
            trust = w["stable_id"] in self.trusted()
            # PLATE_MERGE_BEGIN 추가 Tailscale 계정 허용 (학생 병합, 2026-11-15). 운영폴더 runtime/extra_tailscale_accounts 에 적힌 계정(한 줄에 하나)의
            # 기기는 로봇 소유 계정의 기기와 똑같이 다룬다: 보기는 허용, 조작은 허용 목록(trusted_devices.json)에 있는 기기만.
            # 파일이 없거나 비어 있으면 원래 동작과 같다.
            if not same and w["user"]:
                try:
                    _acc = {_l.strip().lower() for _l in (L / "runtime" / "extra_tailscale_accounts").read_text().splitlines()
                            if _l.strip() and not _l.lstrip().startswith("#")}
                except (OSError, ValueError):
                    _acc = set()
                same = str(w["user"]).strip().lower() in _acc
            # PLATE_MERGE_END
            return dict(kind="tailnet", read=same, control=same and trust, actor="dev:" + w["stable_id"], name=w["name"],
                        stable_id=w["stable_id"], user=w["user"],
                        why=None if same and trust else ("DEVICE_NOT_TRUSTED (제어 허용 장치 아님 — 보기 전용)" if same else "OTHER_TAILNET_USER"))
        return dict(kind="lan", read=False, control=False, why="NOT_TAILSCALE (Tailscale 주소로 접속하십시오)")
RID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
SNAP = re.compile(r"^S[0-9]{8}_[0-9]{6}_[0-9a-f]{6}$")
EXPID = re.compile(r"^(X[0-9]{8}_[0-9]{6}|NO_EXPERIMENT_[0-9]{8})$")
ACTIVE = ("STARTING", "RUNNING", "PARTIAL", "PAUSED", "STOPPING")


def jdump(o):
    return json.dumps(o, ensure_ascii=False, default=str)


def append_jsonl(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(jdump(rec) + "\n")


def sid_hash(sid):
    return None if not sid else hashlib.sha256(sid.encode()).hexdigest()[:12]


# ------------------------------------------------------------------ ctl bridge (fixed argv only)
def run_ctl(args, timeout=180):
    """ctl.py with a fixed argument list built from allow-listed values; returns (rc, tail of output)."""
    env = dict(os.environ, LIVE1_ROOT=str(L))
    try:
        r = subprocess.run(["/usr/bin/python3", "-I", "-B", str(CTL), *args], capture_output=True, text=True, timeout=timeout,
                           env=env, cwd=str(L), stdin=subprocess.DEVNULL)
        return r.returncode, (r.stdout + r.stderr)[-6000:]
    except subprocess.TimeoutExpired:
        return 124, f"ctl {' '.join(args[:2])} timed out after {timeout}s"


def refusal_reasons(out):
    """REFUSED / FAIL lines of a ctl run (shown to the operator as the reason)."""
    keep = [ln.replace("[live1_ctl] ", "") for ln in out.splitlines() if "REFUSED" in ln or " FAIL " in ln or "not alive" in ln]
    return keep[-6:] or [ln for ln in out.splitlines()[-3:]]


AIM_CONSOLE = "http://127.0.0.1:8088"           # student aim console (LIVE2 launcher binds loopback only)
E6_BEST = {}                                    # (observation_id, roi) -> best ROI sharpness seen through /api/e6/metrics
MANUAL_CMD_PHASES = dict(capture=("AIM_READY", "WAIT_MANUAL_FOCUS", "CAPTURE_FAILED"),
                         capture_anyway=("AIM_READY", "WAIT_MANUAL_FOCUS", "CAPTURE_FAILED"),
                         cancel=("AIM_READY", "WAIT_MANUAL_FOCUS", "CAPTURE_FAILED", "CAPTURE_REVIEW"),
                         abort=("WAIT_BODY_SETTLE", "AIMING", "WAIT_AIM_SETTLE", "AIM_READY", "WAIT_MANUAL_FOCUS", "CAPTURE_FAILED",
                                "CAPTURE_REVIEW"),
                         accept_capture=("CAPTURE_REVIEW",), retake=("CAPTURE_REVIEW",),     # 2026-11-01 post-capture review
                         reposition_done=("REPOSITION_NEEDED",), reposition_skip=("REPOSITION_NEEDED",))     # R36


def aim_console_get(path, timeout=2.0):
    """-> (status, body, headers) from the local aim console; (503, reason, {}) when it is not reachable."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(AIM_CONSOLE + path, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)
    except (urllib.error.URLError, OSError) as e:
        return 503, str(e).encode(), {}


def aim_console_alive():
    """Student aim console started for E6 (pid file written by e6_aim_console_ctl)."""
    try:
        os.kill(int((C.OWNED / "e6_aim_console.pid").read_text().split()[0]), 0)
        return True
    except (OSError, ValueError, IndexError):
        return False


def alive(m):
    return C.owner_alive(C.owner_record(m))


def fastlio_input_status():
    """FAST-LIO input mode of the running 'fastlio' module + (body mode) IMU bridge and L1-adapter health, read from its run dir."""
    rec = C.owner_record("fastlio") or {}
    inp = rec.get("input") or ("l1" if rec else None)
    out = dict(input=inp, alive=alive("fastlio") if rec else False)
    if inp == "body" and rec.get("body_run_dir"):
        rd = Path(rec["body_run_dir"])
        ad = C.read_json(rd / "adapter_status.json") or {}
        try:
            bh = json.loads((rd / "bridge_health.jsonl").read_text().strip().splitlines()[-1])
        except (OSError, ValueError, IndexError):
            bh = {}
        now = time.time()
        out.update(body_run_dir=str(rd), calib=rec.get("body_calib"), calib_sha256=rec.get("body_calib_sha256"),
                   bridge=dict(health=bh.get("health"), publish_hz=bh.get("publish_hz"), gaps=bh.get("gaps"),
                               age_s=round(now - bh["wall_ns"] / 1e9, 1) if bh.get("wall_ns") else None),
                   adapter=dict(gate_ok=ad.get("gate_ok"), odom_out=ad.get("odom_out"), cloud_out=ad.get("cloud_out"),
                                odom_gated=ad.get("odom_gated"), cloud_gated=ad.get("cloud_gated"), cloud_ms_max=ad.get("cloud_ms_max"),
                                tag=ad.get("tag"), age_s=round(now - ad["wall"], 1) if ad.get("wall") else None))
    return out


def e6_settle_view(X):
    """MANUAL_SETTLE panel (2026-10-30): the gate snapshot written by the E6 bridge, or — when the bridge is not running (no approval
    B) — the newest confirmed stop of THIS experiment from e6_state, so the tablet still shows the stop and why the pan-tilt cannot
    run. Read-only."""
    import e6_settle as ST
    mode = (X or {}).get("e6_mode") or "AUTO"
    apb = C.approval_status("B", "pantilt")
    snap = ST.read_snapshot(C.STATE) or {}
    exp = (X or {}).get("exp_id")
    if snap.get("experiment_id") not in (None, exp):
        snap = {}
    pend = None
    st = C.read_json(C.STATE / "e6_state.json") or {}
    for o in reversed(st.get("recent") or []):
        if o.get("experiment_id") in (None, exp) and o.get("state") in ("WAIT_STOP", "STOPPED_TARGET_READY"):
            pend = {k: o.get(k) for k in ("observation_id", "state", "created_timestamp", "stopped_wall", "detection_timestamp",
                                          "stop_check", "detect_to_stop_s", "detect_to_stop_path_m")}
            break
    now = time.time()
    return dict(mode=mode, e6_alive=alive("e6"), approval_b=dict(valid=apb["valid"], why=apb.get("why")),
                gate=snap or None, gate_age_s=None if not snap.get("wall") else round(now - float(snap["wall"]), 1),
                pending_stop=pend, imu=ST.imu_vibration(C.RUNTIME, now), message=ST.MSG_WAIT, unstable_message=ST.MSG_UNSTABLE)


def approval(module=None):
    ap = C.approval_status("A", module)
    exp = ap.get("expires_epoch")
    ap["remaining_s"] = None if not exp else round(float(exp) - time.time(), 1)
    return ap


def module_run_s(m, ap):
    """Per-module limit first, never past the approval expiry (ctl re-checks both). STANDING authorization has no expiry:
    the limit is the operator's per-experiment maximum (web setting, capped by standing_auth.DEV_CAP_S)."""
    lim = (ap.get("limits") or {}).get(m, {}).get("max_run_s")
    if ap.get("kind") == "STANDING":
        return float(lim if lim is not None else ap.get("experiment_max_s") or 0)
    rem = (ap.get("remaining_s") or 0) - 30.0
    return max(0.0, min(rem, float(lim)) if lim is not None else rem)


def run_authority(ap):
    """Plain-language authority block for the UI (standing or legacy approval A)."""
    if ap.get("kind") == "STANDING":
        return dict(kind="STANDING", valid=ap["valid"], state=ap.get("state"), why=ap.get("why"), grant_id=ap.get("grant_id"),
                    device=ap.get("device"), modules=ap.get("modules"), excluded=ap.get("excluded"),
                    experiment_max_s=ap.get("experiment_max_s"), boot_bound=False, expires=None,
                    hardware="NOT_ALLOWED (pan-tilt, IMX477, focus, HOME, serial, /cmd_vel, posture, stop)")
    return dict(kind="LEGACY_A", valid=ap["valid"], why=ap.get("why"), modules=ap.get("modules"), remaining_s=ap.get("remaining_s"),
                boot_bound=True)


# ------------------------------------------------------------------ experiment (single, persisted)
class Experiment:
    def __init__(self):
        self.lock = threading.RLock()
        self.d = C.read_json(EXP_FILE) or dict(state="IDLE")

    def save(self):
        with self.lock:
            C.atomic_json(EXP_FILE, self.d)

    def event(self, kind, **k):
        with self.lock:
            ev = dict(wall=time.time(), time=C.now_wall(), kind=kind, **k)
            self.d.setdefault("timeline", []).append(ev)
            self.d["timeline"] = self.d["timeline"][-400:]
            if self.d.get("record_dir"):
                append_jsonl(Path(self.d["record_dir"]) / "timeline.jsonl", ev)
            self.save()

    def mod(self, m, **k):
        with self.lock:
            self.d.setdefault("modules", {}).setdefault(m, {}).update(k, wall=time.time())
            self.save()

    @property
    def active(self):
        return self.d.get("state") in ACTIVE

    def eval_dir(self):
        e = self.d.get("exp_id") if self.active or self.d.get("state") == "ENDED" else None
        return EVAL / (e or f"NO_EXPERIMENT_{time.strftime('%Y%m%d')}")


# ------------------------------------------------------------------ controller
class Controller:
    def __init__(self):
        self.exp = Experiment()
        self.job_lock = threading.Lock()
        self.job = None
        self.fails, self.seen = {}, {}
        self.slock = threading.Lock()
        self.stop = False
        CTL_DIR.mkdir(parents=True, exist_ok=True)
        self.auth = AUTH.AuthStore(CTL_DIR / "auth", CTL_DIR / "operator_token")
        import safety_guard as SG
        self.guard = SG.SafetyGuard()

    # ---------------- auth
    def token(self):
        try:
            return (CTL_DIR / "operator_token").read_text().strip()
        except OSError:
            return None

    def legacy_allowed(self):
        return not (CTL_DIR / "auth" / "legacy_retired.json").exists()

    def login_legacy(self, ip, tok, ua):
        """Old operator-token login (kept until the account login is verified on the real tablet; never 'remembered')."""
        with self.slock:
            if not self.legacy_allowed():
                return None, "LEGACY_TOKEN_LOGIN_RETIRED (아이디·비밀번호로 로그인)"
            f = self.fails.get(ip, [0, 0.0])
            if f[0] >= LOGIN_FAILS and time.time() - f[1] < LOCKOUT_S:
                return None, f"LOCKED_OUT {LOCKOUT_S}s after {LOGIN_FAILS} wrong tokens"
            if not self.auth.setup_code_ok(tok):
                self.fails[ip] = [f[0] + 1, time.time()]
                return None, "WRONG_TOKEN"
            self.fails.pop(ip, None)
            return self.auth.create(AUTH.USER, False, ua, ip, kind="legacy_token"), None

    def session(self, sid):
        return self.auth.get(sid)

    def owns(self, sid):
        return bool(sid) and self.exp.d.get("owner") == sid_hash(sid)

    # ---------------- jobs (one at a time)
    def submit(self, name, fn, rid):
        # 2026-10-28 (test_console_fake 07, intermittent 409): the watchdog also takes job_lock for its 2-s periodic check while
        # an experiment runs; a request arriving in that moment was answered BUSY although no job ran. A RUNNING job still
        # refuses at once; otherwise wait (bounded) for the watchdog's short hold.
        j = self.job
        if j is not None and not j.get("done"):
            return None
        if not self.job_lock.acquire(timeout=2.5):
            return None
        job = self.job = dict(id=rid, action=name, started=time.time(), steps=[], done=False, result=None)

        def run():
            try:
                job["result"] = fn(job)
            except Exception as e:                     # recorded, shown to the operator; the console keeps running
                job["result"] = dict(ok=False, error=f"{type(e).__name__}: {e}")
                self.exp.event("job_error", action=name, error=str(e))
            finally:
                # 2026-10-22: 'done' was set BEFORE the lock was released -> a client acting right after seeing done got
                # 409 BUSY (intermittent test_console_fake failures). Release first, then publish done on THIS job.
                self.job_lock.release()
                job.update(done=True, finished=time.time())
        threading.Thread(target=run, daemon=True).start()
        return job

    def step(self, job, **k):
        job["steps"].append(dict(wall=time.time(), **k))

    # ---------------- module operations
    def start_module(self, job, m, building):
        if m == "e6":
            apb = C.approval_status("B", "pantilt")
            # PLATE_MERGE_BEGIN E6 = 명판 촬영 방식 (학생 병합, 2026-11-13; 11-15 사용자 결정 "지금 방식으로 E6 을 대체"). runtime/plate_case1.on 이 있으면
            # 동의하고 시작한 실험의 E6 는 명판 촬영 방식으로 한다: 예전 자동 촬영 프로그램(e6_precision_bridge)은 띄우지 않고, 팬틸트는 운용 탭의
            # 명판 촬영 카드(촬영 도구 8791)가 쓴다. 승인 B 기록은 그대로 남는다. 모듈 기록의 plate=True 와 사유 문구로 이 방식임을 남긴다.
            if apb["valid"] and (L / "runtime" / "plate_case1.on").exists():
                why = "E6 = 명판 촬영 방식 — 운용 탭 카드에서 조준·촬영 (예전 자동 촬영 프로그램은 실행하지 않음)"
                self.exp.mod(m, state="OFF", why=[why], plate=True)
                self.step(job, module=m, ok=True, why=why)
                self.exp.event("e6_plate", mode="CASE1")
                return True
            # PLATE_MERGE_END
            if not apb["valid"]:
                why = f"승인 B 없음({apb['why']}) — E6 팬틸트·초점·촬영 꺼짐, 정지는 관측 전용"
                self.exp.mod(m, state="OFF", why=[why])
                self.step(job, module=m, ok=True, why=why)
                self.exp.event("e6_off", why=apb["why"])
                return True
        ap = approval(APPROVAL_MODULE.get(m, m))
        if APPROVAL_MODULE.get(m, m) is not None and not ap["valid"]:
            self.exp.mod(m, state="REFUSED", why=[ap["why"]])
            self.step(job, module=m, ok=False, why=ap["why"])
            return False
        inp = (self.exp.d or {}).get("input") or "l1"            # FAST-LIO input mode of this experiment (l1 default | body)
        if m == "fastlio" and alive(m) and ((C.owner_record(m) or {}).get("input") or "l1") != inp:
            why = f"FAST-LIO already running with input {(C.owner_record(m) or {}).get('input') or 'l1'} (requested {inp}) — not reused"
            self.exp.mod(m, state="REFUSED", why=[why])
            self.step(job, module=m, ok=False, why=why)
            return False
        if alive(m):
            self.exp.mod(m, state="RUNNING", why=["already running (reused)"])
            self.step(job, module=m, ok=True, why="already running")
            return True
        args = ["start", m, "--run-s", f"{module_run_s(m, approval()):.0f}"]
        if m == "fastlio" and inp != "l1":
            args += ["--input", inp]
        e6m = (self.exp.d or {}).get("e6_mode") or "AUTO"     # E6 run mode of this experiment (AUTO default | MANUAL_SETTLE)
        if m == "e6" and e6m != "AUTO":
            args += ["--e6-mode", e6m]
        # R39: up to 4 children on 0,1,6,7 (nice 10; SigLIP keeps 0,1 at normal priority) + the repeat searches that make the floor comparison fair. auto_fsm lowers it to 2 while signs are in use (e2_budget.json). GPU E2 stays off.
        e2_args = ["--building", building, "--procs", "4", "--cores", "0,1,6,7", "--refresh-s", "90"]
        args += dict(lio2=["--accum"], observe=["--keep"], e34=["--building", building], e2=e2_args).get(m, [])
        self.exp.mod(m, state="STARTING", why=[])
        rc, out = run_ctl(args)
        if rc != 0:
            why = refusal_reasons(out)
            self.exp.mod(m, state="FAILED", rc=rc, why=why)
            self.step(job, module=m, ok=False, rc=rc, why=why)
            self.exp.event("module_start_failed", module=m, rc=rc, why=why)
            return False
        ok, why = self.wait_health(m)
        self.exp.mod(m, state="RUNNING" if ok else "FAILED", rc=rc, why=[why])
        self.step(job, module=m, ok=ok, why=why)
        self.exp.event("module_started" if ok else "module_unhealthy", module=m, why=why)
        return ok

    def wait_health(self, m, timeout=None):
        """Real health, not 'process exists': FAST-LIO map initialised, odometry flowing, hub frames, E3/E4 writing."""
        timeout = timeout or dict(fastlio=60, lio2=25, hub=20, observe=40, e34=20, e6=30, e2=90, rec=5).get(m, 10)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not alive(m):
                st = C.read_json(C.OWNED / f"{m}.sup_state.json") or {}
                return False, f"process ended: {st.get('end_reason') or st.get('status')}"
            if m == "fastlio":
                log = (C.owner_record(m) or {}).get("log")
                if log and Path(log).exists() and "Initialize the map kdtree" in Path(log).read_text(errors="replace"):
                    return True, "map kdtree initialised"
            elif m == "lio2":
                d = C.read_json(C.STATE / "lio_state.json") or {}
                if time.time() - float(d.get("written_wall", 0)) < 2.0 and ((d.get("odom") or {}).get("n") or 0) > 0:
                    return True, f"odometry {(d.get('odom') or {}).get('hz_10s')} Hz"
            elif m == "hub":
                d = C.read_json(HUB / "index.json") or {}
                if time.time() - float(d.get("wall", 0)) < 2.0:
                    return True, "hub index fresh"
            elif m == "observe":
                k = C.read_json(C.OWNED / "observe.keeper.json") or {}
                if k.get("siglip") == "RUNNING":
                    return True, "SigLIP keeper RUNNING"
            elif m == "e34":
                if (WEB.age_of(C.STATE / "e4_state.json") or 99) < 3.0:
                    return True, "E4 writing"
            elif m == "e2":
                if (WEB.age_of(C.STATE / "e2" / "e2_state.json") or 99) < 5.0:
                    return True, "E2 coordinator writing"
            elif m == "rec":
                if time.time() - t0 > 1.5:
                    return True, "recorder alive"
            elif m == "e6":
                b = C.read_json(C.STATE / "e6_bridge.json") or {}
                if b.get("state") == "REFUSED":
                    return False, f"E6 bridge refused: {b.get('why')}"
                if b.get("state") == "RUNNING" and time.time() - float(b.get("wall", 0)) < 3.0:
                    return True, f"E6 bridge RUNNING ({b.get('focus_mode')}, {b.get('af_strategy')})"
            time.sleep(0.5)
        return False, f"no health signal within {timeout}s (process still running)"

    def stop_module(self, job, m):
        if m == "e6" and not alive(m) and aim_console_alive():
            rc, out = run_ctl(["stop", "e6"], timeout=90)        # bridge gone: the aim console still goes HOME and exits
            self.exp.mod(m, state="STOPPED", why=refusal_reasons(out)[-2:])
            return True
        if not C.owner_record(m) or not alive(m):
            self.exp.mod(m, state="STOPPED", why=["not running"])
            return True
        rc, out = run_ctl(["stop", m], timeout=90 if m == "e6" else 60)
        ok = not alive(m)
        self.exp.mod(m, state="STOPPED" if ok else "STOP_FAILED", why=refusal_reasons(out)[-2:])
        self.step(job, module=m, ok=ok, stopped=True)
        self.exp.event("module_stopped" if ok else "module_stop_failed", module=m, rc=rc)
        return ok

    # ---------------- state archive (old data is never shown as LIVE)
    def archive_state(self, dest):
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        if C.STATE.exists():
            for p in list(C.STATE.iterdir()):
                shutil.move(str(p), str(dest / p.name))
                n += 1
        for p in (SHM / "status.json", SHM / "frame.jpg", SHM / "frame_inference.jpg"):
            if p.exists():
                (dest / "e5_shm").mkdir(exist_ok=True)
                shutil.move(str(p), str(dest / "e5_shm" / p.name))
        return n

    def set_pause(self, paused, rid):
        C.atomic_json(C.STATE / "control" / "pause.json", dict(paused=paused, since=time.time(), request_id=rid,
                                                               note="operator pause: analysis held; robot not commanded"))

    # ---------------- actions
    def a_start(self, job, building, siglip, rid, sid, input_mode="l1", e6_mode="AUTO", consent=None, loc_mode=AUTO_MODE):
        loc_mode, siglip = AUTO_MODE, True
        pol = LOC_MODES.get(loc_mode)
        if pol and not pol["e5_auto_stop"]:
            siglip, consent = False, None
        skip = (pol or {}).get("skip") or ()
        self.guard.reset()                                   # per experiment: no timer / odometry state carried over
        self.ready_last = None
        ap = approval()
        X = self.exp
        with X.lock:
            exp_id = time.strftime("X%Y%m%d_%H%M%S")
            rd = EXPERIMENTS / exp_id
            X.d = dict(exp_id=exp_id, state="STARTING", building=building, siglip=siglip, input=input_mode, e6_mode=e6_mode,
                       loc_mode=loc_mode, loc_policy={k: v for k, v in (pol or {}).items() if k != "skip"} or None,
                       created_wall=time.time(),
                       created=C.now_wall(), owner=sid_hash(sid), request_id=rid, modules={m: dict(state="PENDING") for m in START_ORDER
                                                                                           if (m != "observe" or siglip) and m not in skip},
                       record_dir=str(rd), eval_dir=str(EVAL / exp_id), approval=dict(file=ap.get("file"), expires=ap.get("expires_epoch"),
                                                                                 kind=ap.get("kind"), grant_id=ap.get("grant_id"),
                                                                                 experiment_max_s=ap.get("experiment_max_s")),
                       boot_id=C.boot_id(), paused=False, timeline=[])
            rd.mkdir(parents=True, exist_ok=True)
        moved = self.archive_state(rd / "pre_start_leftovers")
        sg = C.read_json(PENDING_START)
        if sg:                                   # evaluation store only: E2..E7 get the building name and nothing else
            sg = dict(sg, exp_id=exp_id, linked_at_start=C.now_wall())
            append_jsonl(EVAL / exp_id / "gt.jsonl", sg)
            PENDING_START.unlink()
            X.d["start_gt_id"] = sg["gt_id"]
        X.event("experiment_start", building=building, siglip=siglip, input=input_mode, e6_mode=e6_mode, loc_mode=loc_mode, archived_leftover_files=moved,
                start_gt=bool(sg))
        # PLATE_MERGE_BEGIN 학생 도구 시작 (학생 병합, 2026-11-13). runtime/student_start_hook 에 스크립트 경로가 적혀 있으면 실험을 시작할 때
        # 그 스크립트를 떼어서 돌린다(촬영 도구·전용 코어 도구). 파일이 없으면 아무 일도 하지 않는다. 실패해도 실험 시작은 그대로 간다.
        try:
            import subprocess as _sp
            _hook = (L / "runtime" / "student_start_hook").read_text().strip()
            if _hook and Path(_hook).is_file():
                _sp.Popen(["/bin/bash", _hook], stdin=_sp.DEVNULL, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, start_new_session=True, close_fds=True)
                X.event("student_start_hook", script=_hook)
        except (OSError, ValueError):
            pass
        # PLATE_MERGE_END
        if consent is not None:                  # BEFORE the modules: ctl 'start e6' and the bridge read approval B
            try:
                rec = write_pantilt_consent(exp_id, sid, consent, ap)
                X.d["pantilt_consent"] = dict(file=rec["file"], expires_epoch=rec["expires_epoch"], vlm=rec["limits"]["e6_vlm"])
                X.event("pantilt_consent", file=rec["file"], vlm=rec["limits"]["e6_vlm"])
            except OSError as ex:
                X.event("pantilt_consent_failed", error=f"{type(ex).__name__}: {ex}")
        self.set_pause(False, rid)
        failed, critical_failed = [], False
        for m in START_ORDER:
            if (m == "observe" and not siglip) or m in skip:
                continue
            if critical_failed:
                X.mod(m, state="SKIPPED", why=["critical module failed earlier"])
                continue
            if not self.start_module(job, m, building):
                failed.append(m)
                critical_failed = m in CRITICAL
        X.d["state"] = "RUNNING" if not failed else "PARTIAL"
        X.d["failed"] = failed
        X.event("experiment_running" if not failed else "experiment_partial", failed=failed)
        return dict(ok=not failed, state=X.d["state"], failed=failed)

    def a_end(self, job, reason):
        X = self.exp
        X.d["state"] = "STOPPING"
        X.event("experiment_stopping", reason=reason)
        bad = [m for m in STOP_ORDER if not self.stop_module(job, m)]
        if X.d.get("pantilt_consent"):          # the consent ends with the experiment (also bound to exp_id in approval_status)
            try:
                close_pantilt_consent(X.d["exp_id"], reason)
                X.event("pantilt_consent_closed", reason=reason)
            except OSError as ex:
                X.event("pantilt_consent_close_failed", error=f"{type(ex).__name__}: {ex}")
        rd = Path(X.d["record_dir"])
        self.set_pause(False, "end")
        n = self.archive_state(rd / "state")
        t_exp = float(X.d.get("created_wall") or 0)
        for m in STOP_ORDER:                     # 2026-10-30: only files of THIS experiment (an older owner record, e.g. e6 left
            rec = C.owner_record(m) or {}        # from 10-28 while e6 stayed OFF, must not be archived as this run's log)
            for k in ("sup_log", "log"):
                if rec.get(k) and Path(rec[k]).exists() and rec.get("started_wall") and Path(rec[k]).stat().st_mtime >= t_exp:
                    shutil.copy2(rec[k], rd / Path(rec[k]).name)
            st = C.read_json(C.OWNED / f"{m}.sup_state.json")
            stp = C.OWNED / f"{m}.sup_state.json"
            if st and stp.stat().st_mtime >= t_exp:
                C.atomic_json(rd / f"{m}.sup_state.json", st)
        X.d.update(state="ENDED" if not bad else "ENDED_WITH_LEFTOVERS", end_reason=reason, ended=C.now_wall(), ended_wall=time.time(),
                   stop_failed=bad, archived_files=n)
        X.event("experiment_ended", reason=reason, stop_failed=bad, archived_files=n)
        C.atomic_json(rd / "experiment.json", X.d)
        try:                                     # every writer has stopped: copy the run-reused volatile sources (shm store,
            ar = EA.archive_run(rd, EA.live_sources(L, STORE, HUB), X.d)   # FAST-LIO Log, recorder, E5 run) before any next run
            X.event("archive_verified" if ar["verified"] else "archive_failed", n_files=ar["n_files"], bytes=ar["bytes"],
                    missing=ar["missing"], took_s=ar["took_s"])
        except Exception as e:                   # noqa: BLE001  never breaks shutdown; the next start is then gated
            X.event("archive_failed", error=f"{type(e).__name__}: {e}")
        C.atomic_json(rd / "experiment.json", X.d)
        try:
            evaluate_experiment(X.d["exp_id"])
        except Exception as e:                   # evaluation must never break the shutdown path
            X.event("evaluation_failed", error=f"{type(e).__name__}: {e}")
        return dict(ok=not bad, state=X.d["state"], record_dir=str(rd), stop_failed=bad)

    def a_pause(self, job, rid):
        X = self.exp
        self.set_pause(True, rid)
        was = alive("observe")
        if was:
            self.stop_module(job, "observe")
        X.d.update(state="PAUSED", paused=True, siglip_was_on=was)
        X.event("paused", siglip_stopped=was)
        return dict(ok=True, state="PAUSED", siglip_stopped=was)

    def a_resume(self, job, rid):
        X = self.exp
        self.set_pause(False, rid)
        ok = True
        if X.d.get("siglip_was_on"):
            ok = self.start_module(job, "observe", X.d["building"])
        X.d.update(state="RUNNING" if ok else "PARTIAL", paused=False, siglip_was_on=False)
        X.event("resumed", siglip_restarted=ok)
        return dict(ok=ok, state=X.d["state"])

    def a_siglip(self, job, on):
        X = self.exp
        if on:
            ok = self.start_module(job, "observe", X.d["building"])
        else:
            ok = self.stop_module(job, "observe")
        X.d["siglip"] = on if ok else X.d.get("siglip")
        X.event("siglip_on" if on else "siglip_off", ok=ok)
        return dict(ok=ok)

    def a_e2_start(self, job):
        ok = self.start_module(job, "e2", self.exp.d["building"])
        self.exp.event("e2_start", ok=ok)
        return dict(ok=ok)

    def a_e2_research(self, rid):
        req = dict(why="OPERATOR_REQUEST", wall=time.time(), request_id=rid)
        C.atomic_json(C.STATE / "e2" / "e2_request.json", req)
        self.exp.event("e2_research_requested", request_id=rid)
        return dict(ok=True, note="E2 runs it at the next pass slot (min 120 s between passes, needs >= 2 m motion in the window)")

    def a_e6_operator_target(self, rid, body):
        """OPERATOR_ASSISTED E6 request: the operator tapped a target on the CURRENT live RGB frame. Depth = median of the
        aligned depth (mm) in a 9x9 window at that pixel; odometry = raw FAST-LIO interpolated at the frame stamp. Labelled
        OPERATOR_ASSISTED everywhere (never counted as automatic E5 detection or E8 automatic success)."""
        import numpy as np
        import det_pose as DP
        try:
            px, py = int(body["x"]), int(body["y"])
        except (KeyError, TypeError, ValueError):
            return dict(ok=False, why="X_Y_REQUIRED")
        idx = C.read_json(HUB / "index.json") or {}
        fr = idx.get("frames")
        fr = fr[-1] if isinstance(fr, list) and fr else fr
        if not fr or time.time() - float(idx.get("wall", 0)) > 2.0:
            return dict(ok=False, why="HUB_NOT_LIVE (no current RGB frame)")
        w, h = int(fr["w"]), int(fr["h"])
        if not (0 <= px < w and 0 <= py < h):
            return dict(ok=False, why=f"PIXEL_OUTSIDE_FRAME {w}x{h}")
        z = None
        ds_, gap_ = DF.frame_depth_state(fr)
        if ds_ != DF.DEPTH_OK:                        # R34: the slot file may hold an older frame's depth
            return dict(ok=False, why=f"DEPTH_UNAVAILABLE {ds_} (gap {gap_} s): 지금은 거리·조준을 계산할 수 없습니다")
        try:
            b = (HUB / f"slot_{int(fr['slot']):02d}.depth").read_bytes()
            if len(b) == w * h * 2:
                d = np.frombuffer(b, "<u2").reshape(h, w)[max(0, py - 4):py + 5, max(0, px - 4):px + 5]
                d = d[d > 0]
                z = round(float(np.median(d)) / 1000.0, 3) if len(d) >= 10 else None
        except OSError:
            z = None
        if z is None or not 0.3 <= z <= 10.0:
            return dict(ok=False, why=f"NO_VALID_DEPTH_AT_PIXEL ({z})")
        od = DP.odom_at([tuple(o[:4]) for o in idx.get("odom") or []], fr.get("stamp"))
        oid = time.strftime("OP_%H%M%S_") + rid[:6]
        rec = dict(observation_id=oid, source="OPERATOR_ASSISTED", basis="OPERATOR_PICKED_ON_LIVE_RGB", bbox=[px - 15, py - 15, px + 15, py + 15],
                   depth_m=z, detection_timestamp=fr.get("stamp"), image_frame_id=fr.get("seq"), odom_pose_at_detection=od,
                   request_id=rid, wall=time.time(), target_track_id=None, sim=None)
        with open(C.STATE / "e6_operator_requests.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.exp.event("e6_operator_target", observation_id=oid, pixel=[px, py], depth_m=z, odom=od.get("status"))
        return dict(ok=True, observation_id=oid, depth_m=z, odom=od.get("status"),
                    note="정지 요청 — 조이스틱으로 정지하면 E6 가 목표 좌표와 예상 팬틸트 각을 계산합니다 (구동 안 함)")

    def a_e6_manual(self, rid, body, sid):
        """Operator decision in the E6 manual-focus panel: capture / capture_anyway (low sharpness confirmed) / cancel /
        abort. Only queued for the CURRENT observation of THIS experiment while the bridge is alive in a matching phase;
        the precision bridge executes it (and answers in e6_manual.json). Nothing here touches a device."""
        import e6_manual as M
        act, oid = body.get("action"), body.get("observation_id")
        if act not in M.ACTIONS:
            return dict(ok=False, why="BAD_ACTION")
        snap = M.read_snapshot(C.STATE) or {}
        exp = self.exp.d.get("exp_id")
        if not snap or snap.get("observation_id") != oid or not isinstance(oid, str):
            return dict(ok=False, why=f"NOT_THE_CURRENT_OBSERVATION (current {snap.get('observation_id')})")
        if snap.get("experiment_id") not in (None, exp):
            return dict(ok=False, why=f"SESSION_MISMATCH ({snap.get('experiment_id')} != {exp})")
        if time.time() - float(snap.get("wall") or 0) > 5.0:
            return dict(ok=False, why="E6_BRIDGE_SILENT (manual state older than 5 s)")
        if snap.get("phase") not in MANUAL_CMD_PHASES[act]:
            return dict(ok=False, why=f"PHASE_{snap.get('phase')} (not accepted for {act})")
        roi = body.get("roi")
        if roi is not None and M.parse_roi(roi) is None:
            return dict(ok=False, why="BAD_ROI")
        verdict = body.get("verdict") if body.get("verdict") in ("sharp", "insufficient") else None
        cmd = dict(request_id=rid, action=act, observation_id=oid, experiment_id=exp, roi=roi, verdict=verdict,
                   hands_clear=body.get("hands_clear") is True,
                   best_seen=E6_BEST.get((oid, roi or "")), actor=sid_hash(sid), wall=time.time())
        M.append_command(C.STATE, cmd)
        self.exp.event("e6_manual_cmd", **{k: cmd[k] for k in ("request_id", "action", "observation_id", "roi", "verdict")})
        return dict(ok=True, queued=act, note="E6 브리지가 처리합니다 (결과는 관측 패널에 표시)")

    def a_e6_settle(self, rid, body, sid):
        """MANUAL_SETTLE operator decision: approve (진동 안정 확인 · 팬틸트 실행) / cancel (이번 의미정보 획득 취소). Pre-checked here
        (current observation of THIS experiment, bridge alive and waiting, approval B for approve); the E6 bridge re-checks stop,
        freshness, stability and approval B at the moment it reads the command and answers in e6_settle.json. Never touches a device."""
        import e6_settle as ST
        act, oid = body.get("action"), body.get("observation_id")
        if act not in ST.ACTIONS:
            return dict(ok=False, why="BAD_ACTION")
        exp = self.exp.d.get("exp_id")
        if (self.exp.d.get("e6_mode") or "AUTO") != "MANUAL_SETTLE":
            return dict(ok=False, why="NOT_MANUAL_SETTLE_MODE (이 실험은 AUTO E6)")
        apb = C.approval_status("B", "pantilt")
        if act == "approve" and not apb["valid"]:
            return dict(ok=False, why=f"APPROVAL_B_REQUIRED (하드웨어 승인 B {apb['why']}) — 팬틸트 구동 불가, 버튼은 승인 B 를 대신하지 않음")
        if not alive("e6"):
            return dict(ok=False, why="E6_NOT_RUNNING (E6 브리지가 실행 중이 아님 — 팬틸트 구동 불가)")
        snap = ST.read_snapshot(C.STATE) or {}
        if not snap or snap.get("observation_id") != oid or not isinstance(oid, str):
            return dict(ok=False, why=f"NOT_THE_CURRENT_OBSERVATION (current {snap.get('observation_id')})")
        if snap.get("experiment_id") not in (None, exp):
            return dict(ok=False, why=f"SESSION_MISMATCH ({snap.get('experiment_id')} != {exp})")
        if time.time() - float(snap.get("wall") or 0) > 5.0:
            return dict(ok=False, why="E6_BRIDGE_SILENT (대기 상태가 5 s 이상 갱신되지 않음)")
        if snap.get("state") != "WAITING_FOR_SETTLE_APPROVAL":
            return dict(ok=False, why=f"STATE_{snap.get('state')} (승인 대기 상태가 아님)")
        cmd = dict(request_id=rid, action=act, observation_id=oid, experiment_id=exp, actor=sid_hash(sid), wall=time.time())
        ST.append_command(C.STATE, cmd)
        self.exp.event("e6_settle_cmd", **{k: cmd[k] for k in ("request_id", "action", "observation_id")})
        return dict(ok=True, queued=act, note="E6 브리지가 정지·진동·승인 B 를 다시 확인한 뒤 처리합니다")

    def watchdog(self):
        """Approval expiry / boot change during an experiment -> orderly end (modules are capped anyway)."""
        while not self.stop:
            time.sleep(2.0)
            X = self.exp
            if X.d.get("state") in ("RUNNING", "PARTIAL", "PAUSED") and self.job_lock.acquire(blocking=False):
                job = None
                try:
                    self.check_modules()
                    ap = approval()
                    why = None if ap["valid"] else ap["why"]
                    res = self.guard.check(self.safety_inputs())         # sustained heat / throttling / odometry / faults
                    if res["trigger"] and not why:
                        why = f"SAFETY_{res['trigger']}"
                        X.event("safety_end", condition=res["trigger"], why=res["why"], snapshot=res["snapshot"])
                        try:
                            C.atomic_json(Path(X.d["record_dir"]) / "safety_end.json", dict(res, wall=time.time(), time=C.now_wall()))
                        except (OSError, KeyError):
                            pass
                    if ap.get("kind") == "STANDING" and ap.get("state") == "SUSPENDED":
                        why = None                     # suspend = no NEW experiment; the running one continues (revoke ends it)
                    if ap.get("kind") == "STANDING" and ap["valid"] and X.d.get("created_wall") and \
                            time.time() - float(X.d["created_wall"]) > float(ap.get("experiment_max_s") or 0) + 60:
                        why = "EXPERIMENT_MAX_TIME_REACHED"   # operator-set per-experiment maximum (modules are capped too)
                    if X.d.get("boot_id") and X.d["boot_id"] != C.boot_id():
                        why = "BOOT_ID_CHANGED"
                    if why:
                        job = self.job = dict(id="watchdog", action="end", started=time.time(), steps=[], done=False)
                        job["result"] = self.a_end(job, f"AUTO_END {why}")
                finally:
                    self.job_lock.release()               # release first, then publish done (same race as submit)
                    if job is not None:
                        job.update(done=True, finished=time.time())


    def ready_loop(self):
        """Own thread (2 s): the watchdog's timing (auto-end, module checks) is never delayed by the READY evaluation."""
        while not self.stop:
            time.sleep(2.0)
            X = self.exp
            if X.d.get("state") == "RUNNING" and not X.d.get("ready") and not (self.job and not self.job.get("done")):
                self.check_ready()

    def check_ready(self):
        """EXPERIMENT_READY_TO_MOVE (live2/ready_gate.py): measured readiness, latched once per experiment."""
        import ready_gate as RG
        panel = getattr(self, "panel", None)
        if panel is None or self.exp.d.get("ready"):
            return
        try:
            st = C.STATE
            e2 = C.read_json(st / "e2" / "e2_state.json") or {}
            snap = dict(lio=panel.lio(), rgb=panel.rgb(), siglip=panel.siglip(panel.e5()),       # only what READY needs
                        live2=dict(ages={k: WEB.age_of(p) for k, p in (("e2", st / "e2" / "e2_state.json"), ("e3", st / "e3_state.json"),
                                                                        ("e4", st / "e4_state.json"))}, e2=e2))
            br = panel.e6_bridge()
            e6 = dict(running=bool(br.get("alive")), state=br.get("state"), why=br.get("why"),
                      approval_b=C.approval_status("B", "pantilt").get("why") or "valid")
            self.ready_last = RG.evaluate(snap, self.exp.d, self.guard.last, C.boot_id(), e6=e6)
            with self.exp.lock:                            # the snapshot is slow: never write while an end/start job archives
                if self.exp.d.get("state") == "RUNNING" and not (self.job and not self.job.get("done")) \
                        and not self.job_lock.locked():
                    RG.latch(self.exp, self.ready_last, time.time())
        except Exception as e:                                                # noqa: BLE001  never breaks the watchdog
            self.ready_last = dict(ready=False, items=[], error=f"{type(e).__name__}: {e}")

    def safety_inputs(self):
        """Inputs of safety_guard (read-only). FAKE workspace: fake_inject.json 'safety' overrides individual inputs."""
        import safety_guard as SG
        X = self.exp
        rs = next((float(e["wall"]) for e in (X.d.get("timeline") or []) if e.get("kind") in ("experiment_running", "experiment_partial") and e.get("wall")), None)
        lio = C.read_json(C.STATE / "lio_state.json") or {}
        od = lio.get("odom") or {}
        fa = SG.file_age(C.STATE / "lio_state.json")
        age_lio = None if fa is None or od.get("last_rx_age_s") is None else round(fa + float(od["last_rx_age_s"]), 2)
        hub = C.read_json(HUB / "index.json") or {}
        age_hub = None if not hub.get("odom_last_rx_wall") else round(time.time() - float(hub["odom_last_rx_wall"]), 2)
        prot = {n: C.comm_count(n) for n in C.PROTECTED} if not C.FAKE else {}
        s = dict(state=X.d.get("state"), running_since=rs, temp_c=C.max_temp(), mem_mb=C.mem_available_mb(),
                 throttle=SG.throttle_states(), odom_age_lio=age_lio, odom_age_hub=age_hub,
                 odom_seen=bool(od.get("n")) or bool(hub.get("odom")),
                 modules={m: (v or {}).get("state") for m, v in (X.d.get("modules") or {}).items()},
                 sensors_ok=True if C.FAKE else all(prot.get(n) == c for n, c in C.PROTECTED.items()))
        if C.FAKE:
            s.update((C.read_json(L / "fake_inject.json") or {}).get("safety") or {})
        return s

    def check_modules(self):
        """A module that ended on its own (thermal / memory abort, run limit, crash) is shown as such -> experiment PARTIAL."""
        X = self.exp
        for m, v in list((X.d.get("modules") or {}).items()):
            if v.get("state") == "RUNNING" and not alive(m):
                st = C.read_json(C.OWNED / f"{m}.sup_state.json") or {}
                X.mod(m, state="ENDED_UNEXPECTEDLY", why=[f"{st.get('end_reason') or st.get('status')}"])
                if X.d.get("state") == "RUNNING":
                    X.d["state"] = "PARTIAL"
                X.d["failed"] = sorted(set((X.d.get("failed") or []) + [m]))
                X.event("module_ended_unexpectedly", module=m, end_reason=st.get("end_reason"))

    # ---------------- evaluation store
    def gt_save(self, b):
        bld, fl = b.get("building"), b.get("floor")
        if bld not in BUILDINGS or not isinstance(fl, int) or not floor_dir(bld, fl):
            return 400, dict(ok=False, why="BAD_FLOOR")
        kind = b.get("kind")
        if kind not in ("start", "checkpoint", "end"):
            return 400, dict(ok=False, why="BAD_KIND")
        if kind == "start" and self.exp.active:
            return 409, dict(ok=False, why="START_GT_BEFORE_START_ONLY (시작 위치는 실험 시작 전에 지정)")
        if kind == "start" and not isinstance(b.get("yaw_deg"), (int, float)):
            return 400, dict(ok=False, why="START_GT_NEEDS_DIRECTION (두 번째 탭으로 방향 지정)")
        if kind == "end" and self.exp.active:
            return 409, dict(ok=False, why="END_GT_AFTER_END_ONLY (실험 종료 후 지정)")
        if kind == "end" and not self.exp.d.get("exp_id"):
            return 409, dict(ok=False, why="NO_ENDED_EXPERIMENT")
        sem = semantic_meta(bld, fl)
        W_, H_ = sem["size"]
        px, py = b.get("px"), b.get("py")
        if not all(isinstance(v, (int, float)) for v in (px, py)) or not (0 <= px < W_ and 0 <= py < H_):
            return 400, dict(ok=False, why="CLICK_OUTSIDE_MAP")
        yaw = b.get("yaw_deg")
        if yaw is not None and not isinstance(yaw, (int, float)):
            return 400, dict(ok=False, why="BAD_YAW")
        res = sem.get("resolution_m_per_px")
        lio = C.read_json(C.STATE / "lio_state.json") or {}
        rec = dict(gt_id=f"GT{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}", kind=b["kind"], exp_id=self.exp.d.get("exp_id")
                   if self.exp.active else None, building=bld, floor=fl, px=round(float(px), 1), py=round(float(py), 1),
                   map_m=None if not res else [round(px * res, 3), round((H_ - py) * res, 3)], yaw_deg=yaw,
                   note=str(b.get("note") or "")[:300], wall=time.time(), time=C.now_wall(), map=sem["source"], map_sha256=sem["sha256"],
                   odom=dict(last_raw=(lio.get("odom") or {}).get("last_raw"), written_wall=lio.get("written_wall")),
                   source="HUMAN_CLICK", used_by_inference=False)
        if kind == "start":
            C.atomic_json(PENDING_START, rec)    # replaces an earlier pending start mark
            return 200, dict(ok=True, gt=rec, pending=True)
        if kind == "end":
            rec["exp_id"] = self.exp.d["exp_id"]
            append_jsonl(EVAL / self.exp.d["exp_id"] / "gt.jsonl", rec)
            ev = evaluate_experiment(self.exp.d["exp_id"])
            return 200, dict(ok=True, gt=rec, evaluation=ev)
        append_jsonl(self.exp.eval_dir() / "gt.jsonl", rec)
        return 200, dict(ok=True, gt=rec)

    def det_snapshot(self):
        """Freeze the current SigLIP inference frame + its status so a label always refers to one exact frame."""
        st, img = SHM / "status.json", SHM / "frame_inference.jpg"
        d = C.read_json(st)
        if not d or not img.exists():
            return 409, dict(ok=False, why="NO_SIGLIP_FRAME (observe not running)")
        snap = f"S{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"
        out = self.exp.eval_dir() / "snapshots"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out / f"{snap}.jpg")
        sc = d.get("scene") or {}
        stamp = sc.get("stamp")
        hub = C.read_json(HUB / "index.json") or {}
        seq = next((f.get("seq") for f in hub.get("frames") or [] if stamp is not None and f.get("stamp") is not None
                    and abs(float(f["stamp"]) - float(stamp)) < 1e-3), None)
        lio = C.read_json(C.STATE / "lio_state.json") or {}
        dt = os.stat(out / f"{snap}.jpg").st_mtime - os.stat(st).st_mtime
        meta = dict(snap_id=snap, exp_id=self.exp.d.get("exp_id") if self.exp.active else None, wall=time.time(), time=C.now_wall(),
                    run_id=d.get("run_id"), frame=dict(scene_stamp=stamp, n_scene=sc.get("n_scene"), hub_seq=seq,
                                                       hub_seq_note=None if seq is not None else "stamp not in the hub ring any more"),
                    image_status_mtime_diff_s=round(dt, 3), image_matches_status=abs(dt) < 0.5,
                    siglip=dict(threshold=sc.get("thr"), regions=sc.get("regions") or [], prompts=d.get("siglip_prompts")),
                    odom=dict(last_raw=(lio.get("odom") or {}).get("last_raw"), age_s=(lio.get("odom") or {}).get("last_rx_age_s")),
                    review="UNREVIEWED")
        C.atomic_json(out / f"{snap}.json", meta)
        return 200, dict(ok=True, snapshot=meta, image=f"/img/snap/{self.exp.eval_dir().name}/{snap}.jpg")

    def det_label(self, b):
        snap = b.get("snap_id")
        if not isinstance(snap, str) or not SNAP.match(snap):
            return 400, dict(ok=False, why="BAD_SNAP_ID")
        meta = C.read_json(self.exp.eval_dir() / "snapshots" / f"{snap}.json")
        if not meta:
            return 404, dict(ok=False, why="SNAPSHOT_NOT_FOUND")
        lab = b.get("label")
        if lab not in ("TP", "FP", "MISSED", "AMBIGUOUS"):
            return 400, dict(ok=False, why="BAD_LABEL")
        ri, pt = b.get("region_index"), b.get("point")
        regs = (meta.get("siglip") or {}).get("regions") or []
        if lab in ("TP", "FP") and not (isinstance(ri, int) and 0 <= ri < len(regs)):
            return 400, dict(ok=False, why="TP/FP need a SigLIP region of this frame")
        if lab == "MISSED" and not (isinstance(pt, list) and len(pt) == 2 and all(isinstance(v, (int, float)) for v in pt)):
            return 400, dict(ok=False, why="MISSED needs the clicked sign position")
        rec = dict(label_id=f"D{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}", snap_id=snap, label=lab,
                   region_index=ri if lab in ("TP", "FP", "AMBIGUOUS") and isinstance(ri, int) else None,
                   region=regs[ri] if isinstance(ri, int) and 0 <= ri < len(regs) else None,
                   point=pt if isinstance(pt, list) else None, sign_text=str(b.get("sign_text") or "")[:80], note=str(b.get("note") or "")[:300],
                   run_id=meta.get("run_id"), frame=meta.get("frame"), odom=meta.get("odom"), image_matches_status=meta.get("image_matches_status"),
                   wall=time.time(), time=C.now_wall(), review="HUMAN_LABELED", ground_truth_confirmed=lab != "AMBIGUOUS")
        append_jsonl(self.exp.eval_dir() / "detections.jsonl", rec)
        return 200, dict(ok=True, label=rec)

    # ---------------- state for the UI
    def e6_readiness(self):
        """What the E6 pan-tilt / IMX477 path needs on THIS boot, before [실험 시작] (no device contact; cached 3 s)."""
        now = time.time()
        if now - getattr(self, "_e6r_t", 0) < 3.0:
            return self._e6r
        import e6_aim_diag as AD
        import e6_precision_bridge as B
        try:
            holders = subprocess.run(["fuser", "/dev/ttyUSB0"], capture_output=True, text=True, timeout=3).stdout.split()
        except (OSError, subprocess.SubprocessError):
            holders = []
        if aim_console_alive():
            holders = []                                   # our own E6 console holds it during an experiment: expected
        self._e6r, self._e6r_t = AD.readiness(C.approval_status("B", "pantilt"), B.load_envelope(), holders=holders), now
        return self._e6r

    def state(self, sid):
        e6r = self.e6_readiness()                          # slow parts first: job / experiment are read back to back below
        job = self.job
        job = None if not job else {k: job.get(k) for k in ("id", "action", "started", "finished", "done", "result", "steps")}
        X = dict(self.exp.d)                                # job copied BEFORE the state: 'done' never pairs with an older state
        ap = approval()
        mods = {}
        for m in START_ORDER:
            rec = C.owner_record(m)
            st = C.read_json(C.OWNED / f"{m}.sup_state.json") or {}
            mods[m] = dict(alive=alive(m), pid=(rec or {}).get("pid"), sup=st.get("status"), end_reason=st.get("end_reason"),
                           exp=(X.get("modules") or {}).get(m),
                           approved=C.approval_status("B", "pantilt")["valid"] if m == "e6" else
                           (APPROVAL_MODULE.get(m, m) is None) or m in (ap.get("modules") or []),
                           max_run_s=((ap.get("limits") or {}).get(m) or {}).get("max_run_s"))
        res = C.resources()
        return dict(experiment={k: X.get(k) for k in ("exp_id", "state", "building", "siglip", "input", "e6_mode", "loc_mode", "created", "created_wall", "ended",
                                                     "end_reason", "failed", "paused", "record_dir", "eval_dir", "stop_failed")},
                    fastlio_input=fastlio_input_status(), e6_settle=e6_settle_view(X) if self.exp.active else None,
                    **live_location_view(X, self.exp.active),
                    timeline=(X.get("timeline") or [])[-12:], you_own=self.owns(sid), owned_by_other=bool(X.get("owner")) and not self.owns(sid)
                    and self.exp.active, logged_in=bool(self.session(sid)), modules=mods,
                    approval=dict(valid=ap["valid"], why=ap["why"], remaining_s=ap.get("remaining_s"), modules=ap.get("modules"),
                                  file=ap.get("file"), kind=ap.get("kind"), state=ap.get("state"),
                                  how="standing research authorization (ctl standing-grant)" if ap.get("kind") == "STANDING"
                                  else "approval is recorded by Claude Code (ctl record-approval) from the user's text"),
                    authority=run_authority(ap), preflight=preflight(),
                    safety_guard=self.guard.last if self.exp.active else None,
                    ready=dict(latched=X.get("ready"), now=getattr(self, "ready_last", None)) if self.exp.active else None,
                    e6_readiness=e6r,
                    safety=dict(boot_id=C.boot_id(), max_temp_c=res["max_temp_c"], mem_available_mb=round(res["mem_available_mb"]),
                                start_ok=res["max_temp_c"] is not None and res["max_temp_c"] < 72 and res["mem_available_mb"] >= 4300,
                                rule="module start: < 72 C and >= 4300 MB (ctl gate); running: supervisors abort at 80 C / < 2500 MB"),
                    job=job,
                    fake=C.FAKE, buildings=list(BUILDINGS), start_gt_required=REQUIRE_START_GT,
                    pending_start_gt=C.read_json(PENDING_START),
                    last_evaluation=C.read_json(EVAL / X["exp_id"] / "evaluation.json") if X.get("exp_id") else None,
                    web_approval=dict(boot_id=C.boot_id(), modules=list(WEB_APPROVAL_MODULES), max_minutes=WEB_APPROVAL_MAX_MIN,
                                      limits=WEB_APPROVAL_LIMITS,
                                      recorded_by=(C.read_json(ap["file"]) or {}).get("recorded_by") if ap.get("file") else None))


# ------------------------------------------------------------------ 2026-11-01 (r33) live clocks + the ONLY final-result trigger
def final_presentation(X, t_fix, now):
    """R39: GLOBAL_FIX_VERIFIED -> FINAL_PRESENTATION_PENDING -> FINAL_RESULT_VISIBLE. The FIX time is the verification time and never moves; only the popup waits until the pan-tilt work is over (hands off, HOME). Both times are recorded."""
    import auto_fsm as AF
    man, stl = C.read_json(C.STATE / "e6_manual.json") or {}, C.read_json(C.STATE / "e6_settle.json") or {}
    if man.get("experiment_id") not in (None, X.get("exp_id")) or now - float(man.get("wall") or 0) > 600:
        man = {}
    hw = AF.phase_to_hw(man.get("phase"), stl.get("state"))
    p = C.STATE / "final_presentation.json"
    rec = C.read_json(p) or {}
    if rec.get("t_fix") != t_fix:
        rec = dict(kind="LIVE2_FINAL_PRESENTATION", exp_id=X.get("exp_id"), t_fix=t_fix, hw_at_fix=hw, t_visible=None)
    busy = hw in AF.HW_BUSY or man.get("phase") == "HOME_NOT_CONFIRMED"
    if not busy and rec.get("t_visible") is None:
        rec["t_visible"] = now
    if not C.read_json(p) or (C.read_json(p) or {}).get("t_visible") != rec.get("t_visible") or (C.read_json(p) or {}).get("t_fix") != t_fix:
        try:
            tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(rec, default=str)); os.replace(tmp, p)
        except OSError:
            pass
    return dict(state="FINAL_RESULT_VISIBLE" if rec.get("t_visible") is not None else "FINAL_PRESENTATION_PENDING", hw=hw, phase=man.get("phase"), t_fix=t_fix, t_visible=rec.get("t_visible"))


def live_location_view(X, active):
    """-> dict(op_clock, final_fix) for the running experiment.
    op_clock : real elapsed time and the operator-normalized analysis clock (op_clock.py; settle -> 1 s, focus -> 3 s).
    final_fix: the backend's own FIX record — <state>/e7_state.json with global_decision GLOBAL_CONFIRMED written by e6e7_link
               after ALL of its gates (semantic: consistent pair + E2 rivals + FAST-LIO + same odometry session; structure: policy
               ENABLED) and written during THIS experiment. A SHADOW pass, an E7 score, a provisional candidate or a test file of
               another run never qualifies. The page shows the final result only when final_fix.valid is true."""
    if not active or not X.get("created_wall"):
        return dict(op_clock=None, final_fix=None, field_status=None)
    import op_clock as OC
    import status_view as SV
    try:
        fs = SV.build(X, C.STATE, C.LROOT / "logs", C.read_json(HUB / "index.json"))
    except Exception as e:                                       # noqa: BLE001  a status line never breaks the state
        fs = dict(chips=[dict(key="status", level="warn", code="STATUS_VIEW_ERROR", text=f"상태 표시 오류: {type(e).__name__}", detail=str(e)[:160], next=None)], observations=[])
    t0, now = float(X["created_wall"]), time.time()
    try:
        iv = OC.intervals(C.STATE, X.get("exp_id"))
        oc = OC.clock(iv, t0, now)
        oc.update(OC.ops_timer(C.STATE, X.get("exp_id"), t0, now))
    except Exception as e:                                       # noqa: BLE001  a display clock never breaks the state
        iv, oc = [], dict(error=f"{type(e).__name__}: {e}")
    e7 = C.read_json(C.STATE / "e7_state.json") or {}
    ff = dict(valid=False, why="no E7 GLOBAL_CONFIRMED in this experiment")
    if e7.get("global_decision") == "GLOBAL_CONFIRMED":
        tw = float(e7.get("written_wall") or 0)
        if tw < t0:
            ff = dict(valid=False, why="E7 record older than this experiment")
        else:
            fo = C.read_json(C.STATE / "fix_overlay.json") or {}
            fo_ok = abs(float(fo.get("e7_written_wall") or 0) - tw) < 1e-3 and not fo.get("why")
            at = OC.clock(iv, t0, tw) if iv is not None else None
            ff = dict(valid=True, t_fix=tw, path=e7.get("path") or ("SEMANTIC_PAIR" if e7.get("used_observation_ids") else e7.get("source")),
                      hypothesis=e7.get("semantic_hypothesis"), used_observation_ids=e7.get("used_observation_ids") or [],
                      why=e7.get("why"), source=e7.get("source"), TTF_raw_s=(at or {}).get("raw_s"), TTF_normalized_s=(at or {}).get("normalized_s"),
                      normalized_note=OC.NOTE, fix_overlay_ok=fo_ok, fix_overlay_why=None if fo_ok else (fo.get("why") or "E7 점군 스냅샷 없음"))
            ff["presentation"] = final_presentation(X, tw, now)
            first = ((C.read_json(C.STATE / "stage_milestones.json") or {}).get("first") or {}).get("PROVISIONAL") or {}
            ff["first_candidate"] = dict(floor=first.get("floor"), leader=first.get("leader"), t_s=round(float(first["wall"]) - t0, 1)) if first.get("wall") else None
    return dict(op_clock=oc, final_fix=ff, field_status=fs)


# ------------------------------------------------------------------ start readiness shown before [실험 시작]
_pf_cache = [0.0, None]


def preflight():
    """Cheap per-boot readiness (cached 3 s). The authoritative checks are ctl gates at each module start."""
    now = time.time()
    if _pf_cache[1] is not None and now - _pf_cache[0] < 3.0:
        return _pf_cache[1]
    res = C.resources()
    free_gb = shutil.disk_usage(str(C.LROOT)).free / 2 ** 30
    prot = {n: C.comm_count(n) for n in C.PROTECTED} if not C.FAKE else {}
    items = dict(
        boot_id=dict(ok=True, value=C.boot_id()[:8], note="per-boot state is checked fresh; nothing reused from an earlier boot"),
        temperature=dict(ok=res["max_temp_c"] is not None and res["max_temp_c"] < 72.0, value=res["max_temp_c"], rule="< 72 C"),
        memory=dict(ok=res["mem_available_mb"] >= 4300, value=round(res["mem_available_mb"]), rule=">= 4300 MB"),
        storage=dict(ok=free_gb >= 5.0, value=round(free_gb, 1), rule=">= 5 GB"),
        gpu=dict(ok=True, value=res["gpu_load_pct"], rule="shown only (SigLIP worker owns the GPU)"),
        sensors=dict(ok=C.FAKE or all(prot.get(n) == c for n, c in C.PROTECTED.items()), value=prot, rule="ROPUS sensor services running"),
        fastlio_duplicate=dict(ok=C.FAKE or not C.find_pids("fastlio_mapping") or alive("fastlio"),
                               rule="no FAST-LIO outside LIVE2 control"))
    if C.FAKE:                       # FAKE workspace: robot resources are fixtures, as in ctl fake_gates (2026-10-22: a hot SoC
        inj = (C.read_json(L / "fake_inject.json") or {}).get("preflight_fail") or []   # after heavy offline jobs made the
        for k in ("temperature", "memory", "storage"):                                   # console tests fail intermittently)
            items[k] = dict(items[k], ok=k not in inj, fake=True, real_value=items[k]["value"])
    out = dict(ok=all(v["ok"] for v in items.values()), items=items, checked=now)
    _pf_cache[:] = [now, out]
    return out


# ------------------------------------------------------------------ web approval A (fixed scope, explicit operator act)
WEB_APPROVAL_MODULES = ("fastlio", "lio2", "hub", "observe", "e2", "e34", "rec")   # never e2_gpu / frame_probe / pan-tilt
WEB_APPROVAL_MAX_MIN = 60
WEB_APPROVAL_LIMITS = {"fastlio": {"max_run_s": 1800}, "lio2": {"max_run_s": 1800, "accum_cloud_registered": True},
                       "hub": {"max_run_s": 1800}, "observe": {"max_run_s": 1800, "gpu_workers": 1, "mode": "SHADOW", "start_temp_c": 72,
                                                               "start_mem_mb": 4300, "stop_temp_c": 80, "stop_mem_mb": 2500},
                       "e2": {"max_run_s": 1800, "device": "cpu"}, "e34": {"max_run_s": 1800}}


def approval_text(boot, modules, minutes, device):
    lim = ", ".join(f"{m} {WEB_APPROVAL_LIMITS.get(m, {}).get('max_run_s', '승인 기간 내')}{'초' if m in WEB_APPROVAL_LIMITS else ''}" for m in modules)
    return (f"[SCI LIVE-FIELD2 실험 실행 승인 A — 웹 승인]\n\n"
            f"현재 boot_id {boot} 에 한하여, 기록 시점부터 최대 {minutes}분 동안 다음 모듈의 실행을 승인한다.\n"
            f"- 모듈 / 최대 실행: {lim}\n- e2 는 CPU 모드 (e2_gpu 미승인), observe 는 SigLIP GPU 작업자 1개 SHADOW\n"
            f"- 자원 보호: 시작 72°C 미만·MemAvailable 4300MB 이상, 80°C 이상 또는 2500MB 미만이면 중단\n"
            f"- 승인하지 않음: frame_probe, 팬틸트, 초점, IMX477, HOME, /cmd_vel 및 Go2 이동·자세·정지 명령, ROPUS 변경,"
            f" /lidar_points 추가 구독, 승인 시간 자동 연장\n- 로봇 이동은 운용자가 조이스틱으로 직접 수행한다.\n\n"
            f"웹 승인 기기: {device} · 시각 {C.now_wall()} · 운용자가 화면에서 범위를 확인하고 명시적으로 승인함\n")


def record_web_approval(boot, modules, minutes, device):
    d = CTL_DIR / "web_approvals"
    d.mkdir(parents=True, exist_ok=True)
    t = d / f"A_web_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    t.write_text(approval_text(boot, modules, minutes, device))
    lim = {m: v for m, v in WEB_APPROVAL_LIMITS.items() if m in modules}
    return run_ctl(["record-approval", "A", "--text-file", str(t), "--minutes", str(minutes), "--modules", ",".join(modules),
                    "--limits", jdump(lim), "--recorded-by", f"web:{device}"], timeout=30)


# ------------------------------------------------------------------ post-run evaluation (GT is never an inference input)
def _hyp_m(h, res, H_):
    return None if not h.get("pose_px") or not res else (h["pose_px"][0] * res, (H_ - h["pose_px"][1]) * res)


# ------------------------------------------------------------------ 2026-10-31 per-experiment pan-tilt consent (tablet)
PANTILT_CONSENT_ACKS = ("operator_supervises", "manual_button_starts_aim", "motion_blocks_drive", "manual_focus_no_motor")
PANTILT_CONSENT_MAX_S = 3 * 3600.0


def write_pantilt_consent(exp_id, sid, consent, ap):
    """The operator's explicit tablet consent for THIS experiment -> a hardware approval B record (kind B, module 'pantilt') of
    this boot, bound to exp_id (live1_common.legacy_approval_status rejects it for any other / ended experiment) and closed
    at [실험 종료]. It replaces only the boot-file paperwork: the bridge still re-checks B per command, the stop, the
    servo envelope, staleness, single execution, the operator buttons and HOME; focus is manual (no focus motor)."""
    now = time.time()
    acks = consent.get("acks") or {}
    vlm = bool(consent.get("vlm"))
    text = (f"[8770 태블릿 · 실험별 팬틸트 사용 동의] 실험 {exp_id} · 운영자 세션 {sid_hash(sid)} · {C.now_wall()} · 부팅 {C.boot_id()}\n"
            "- 이번 실험에서 팬틸트 카메라를 사용한다: 예\n"
            f"- 운영자가 로봇을 직접 감독한다: {'확인' if acks.get('operator_supervises') else '-'}\n"
            f"- 정지 후 [① 진동 안정 확인 · 팬틸트 조준 시작] 버튼으로만 조준이 시작된다: {'확인' if acks.get('manual_button_starts_aim') else '-'}\n"
            f"- 로봇이 움직이면 팬틸트 구동이 차단된다: {'확인' if acks.get('motion_blocks_drive') else '-'}\n"
            f"- 초점은 수동(초점 모터 사용 안 함), 촬영은 [② 초점 확인 완료 · 사진 촬영] 버튼으로만: {'확인' if acks.get('manual_focus_no_motor') else '-'}\n"
            f"- VLM 사용(메모리 여유가 있을 때만): {'예' if vlm else '아니오 (OCR만)'}")
    exp_max = float(ap.get("experiment_max_s") or PANTILT_CONSENT_MAX_S)
    rec = dict(kind="B", user_text=text, user_text_sha256=hashlib.sha256(text.encode()).hexdigest(), boot_id=C.boot_id(),
               granted_wall=C.now_wall(), granted_epoch=now, expires_epoch=now + min(exp_max, PANTILT_CONSENT_MAX_S), modules=["pantilt"],
               limits=dict(source="TABLET_EXPERIMENT_CONSENT", exp_id=exp_id, focus_mode="MANUAL_FIXED_NO_MOTOR", e6_vlm=vlm,
                           envelope_source="operator-verified envelope file (bridge load_envelope), unchanged"),
               recorded_by="8770 console (operator tablet consent at experiment start)", operator=sid_hash(sid))
    C.APPROVALS.mkdir(parents=True, exist_ok=True)
    f = C.APPROVALS / f"B_{time.strftime('%Y%m%d_%H%M%S')}_{exp_id}.json"
    C.atomic_json(f, rec)
    return dict(rec, file=str(f))


def close_pantilt_consent(exp_id, reason):
    """Newest B record = an expired closing record for this experiment (approval_status reads the newest)."""
    now = time.time()
    text = f"[8770 태블릿 · 팬틸트 사용 동의 종료] 실험 {exp_id} · {C.now_wall()} · {reason}"
    rec = dict(kind="B", user_text=text, user_text_sha256=hashlib.sha256(text.encode()).hexdigest(), boot_id=C.boot_id(),
               granted_wall=C.now_wall(), granted_epoch=now, expires_epoch=now - 1.0, modules=["pantilt"],
               limits=dict(source="TABLET_EXPERIMENT_CONSENT_CLOSED", exp_id=exp_id), recorded_by="8770 console (experiment end)")
    time.sleep(1.05)                             # a later file name than the consent record of the same second
    C.atomic_json(C.APPROVALS / f"B_{time.strftime('%Y%m%d_%H%M%S')}_{exp_id}_END.json", rec)


def evaluate_experiment(exp_id):
    """evaluation/<exp>/evaluation.json from the ARCHIVED run state + the human GT marks. Keeps apart: start GT, odometry
    end reference (start GT + relative FAST-LIO motion; contains drift; the odom-axis-to-heading relation is not verified),
    user end GT, and the algorithm's final candidates. Nothing here is fed back to E2..E7."""
    ed, rd = EVAL / exp_id, EXPERIMENTS / exp_id
    gts = [json.loads(x) for x in open(ed / "gt.jsonl")] if (ed / "gt.jsonl").exists() else []
    start = next((g for g in gts if g["kind"] == "start"), None)
    end = next((g for g in reversed(gts) if g["kind"] == "end"), None)
    lio = C.read_json(rd / "state" / "lio_state.json") or {}
    odom = lio.get("odom") or {}
    last = odom.get("last_raw") or {}
    e3 = C.read_json(rd / "state" / "e3_state.json") or {}
    e4 = C.read_json(rd / "state" / "e4_state.json") or {}
    e2 = C.read_json(rd / "state" / "e2" / "e2_state.json") or {}
    out = dict(exp_id=exp_id, written=C.now_wall(), used_by_inference=False,
               start_gt=start, end_gt=end, odometry=dict(start_raw=[0.0, 0.0, 0.0], end_raw=[last.get("x"), last.get("y"), last.get("yaw_deg")],
                                                          path_m=odom.get("path_m"), frame=[odom.get("frame_id"), odom.get("child_frame_id")],
                                                          note="FAST-LIO starts at its own origin where the robot stood still at start"),
               e2=dict(status=e2.get("status"), floors_done=e2.get("floors_done"), floors_total=e2.get("floors_total")),
               e4=dict(state=e4.get("state"), reasons=e4.get("reasons")))
    if last.get("x") is not None:
        out["odometry"]["net_displacement_m"] = round(math.hypot(last["x"], last["y"]), 2)   # axis-independent
    if start and start.get("map_m") and last.get("x") is not None and start.get("yaw_deg") is not None:
        th = math.radians(start["yaw_deg"])     # ASSUMPTION delta=0: odom x axis = robot heading at start (no frame contract)
        ref = (start["map_m"][0] + math.cos(th) * last["x"] - math.sin(th) * last["y"],
               start["map_m"][1] + math.sin(th) * last["x"] + math.cos(th) * last["y"])
        out["odom_end_reference"] = dict(floor=start["floor"], map_m=[round(v, 2) for v in ref], status="UNVERIFIED_FRAME_CONTRACT",
                                         why="start GT + relative odometry; heading offset odom->body assumed 0; includes drift")
    if start and end and start.get("map_m") and end.get("map_m"):
        gt_disp = math.hypot(end["map_m"][0] - start["map_m"][0], end["map_m"][1] - start["map_m"][1])
        out["gt_net_displacement_m"] = round(gt_disp, 2)
        if out["odometry"].get("net_displacement_m") is not None:
            out["odometry_vs_gt_displacement_error_m"] = round(abs(out["odometry"]["net_displacement_m"] - gt_disp), 2)   # axis-independent
        if out.get("odom_end_reference") and end["floor"] == start["floor"]:
            r = out["odom_end_reference"]["map_m"]
            out["odom_end_reference"]["error_vs_end_gt_m"] = round(math.hypot(r[0] - end["map_m"][0], r[1] - end["map_m"][1]), 2)
    H = sorted(e3.get("hypotheses") or [], key=lambda h: h.get("score", 9e9))
    ref_gt = end or start
    if ref_gt:
        sem = semantic_meta(ref_gt["building"], ref_gt["floor"])
        res, H_ = sem.get("resolution_m_per_px"), sem["size"][1]
        top = []
        for i, h in enumerate(H[:10]):
            m = _hyp_m(h, res, H_)
            top.append(dict(rank=i + 1, id=h.get("id"), floor=h.get("floor"), status=h.get("status"), score=h.get("score"),
                            map_m=None if not m else [round(m[0], 2), round(m[1], 2)],
                            error_vs_end_gt_m=None if not (end and m and h.get("floor") == end["floor"] and end.get("map_m"))
                            else round(math.hypot(m[0] - end["map_m"][0], m[1] - end["map_m"][1]), 2)))
        gfl = ref_gt["floor"]
        out["final_candidates_top10"] = top
        out["correct_floor_rank"] = next((t["rank"] for t in top if t["floor"] == gfl), None)
        out["topk_contains_correct_floor"] = {k: any(t["floor"] == gfl for t in top[:k]) for k in (1, 3, 5)}
        out["floor_reference"] = "end GT" if end else "start GT (end GT not marked yet)"
    try:                                         # 2026-10-30: TTF_raw / manual waits / TTF_adjusted (record only; FIX-less runs get no TTF)
        import e6_settle as ST
        tm = ST.time_metrics(rd)
        C.atomic_json(EVAL / exp_id / "time_metrics.json", tm)
        out["time_metrics"] = {k: tm.get(k) for k in ("e6_mode", "TTF_raw_s", "TTF_adjusted_s", "sum_manual_wait_s", "why")}
    except Exception as e:                       # noqa: BLE001  evaluation must never fail because of the timing add-on
        out["time_metrics"] = dict(error=f"{type(e).__name__}: {e}")
    try:                                         # 2026-10-30: E8 per-run summary (E8_SUMMARY.json / .md) — record only
        import e8_summary as E8S
        e8 = E8S.summarize(rd, EVAL / exp_id, out)
        out["e8"] = {k: e8.get(k) for k in ("result_class", "time_to_fix_s", "manual_wait_s", "fix", "ground_truth", "stages", "replay")}
        out["e8"].update(e2_all_floors_s=(e8.get("e2") or {}).get("all_floors_exact_s"), e4_first_confident_s=(e8.get("e4") or {}).get("first_confident_s"),
                         e7_structure=((e8.get("e7") or {}).get("structure_path") or {}).get("status"),
                         semantic_recovery=(e8.get("semantic_recovery") or {}).get("status"))
    except Exception as e:                       # noqa: BLE001  evaluation must never fail because of the summary
        out["e8"] = dict(error=f"{type(e).__name__}: {e}")
    try:                                         # R35: one row for the paper (PAPER_METRICS.json) — record only, after E8_SUMMARY and time_metrics
        import paper_metrics as PM
        out["paper_metrics"] = PM.build(rd, EVAL / exp_id).get("class_")
    except Exception as e:                       # noqa: BLE001  evaluation must never fail because of this row
        out["paper_metrics"] = f"{type(e).__name__}: {e}"
    C.atomic_json(EVAL / exp_id / "evaluation.json", out)
    return out


# ------------------------------------------------------------------ idle view + history (old runs are never shown as live)
_HOSTS = [0.0, None]
_host_addresses = C.host_addresses


def cached_host_addresses():
    """`ip addr` once a minute instead of on every 1 Hz poll (less CPU / heat on the robot)."""
    if _HOSTS[1] is None or time.time() - _HOSTS[0] > 60:
        _HOSTS[:] = [time.time(), _host_addresses()]
    return _HOSTS[1]


C.host_addresses = cached_host_addresses


def idle_snapshot(panel, X):
    """Same keys as Panel.snapshot(), but no live block carries data: nothing from a previous run looks current."""
    na = dict(status="NO_EXPERIMENT", why="실행 중인 실험 없음")
    return dict(rgb=dict(na), siglip=dict(na), session={}, localization=dict(status="GLOBAL_UNKNOWN", why="실험 없음"),
                live2=dict(ages={}, final_pose="UNKNOWN", idle=True),
                meta=dict(kind="LIVE1_STATE", test_only=panel.test_only, trial_id=None, server_wall=time.time(), server_time=C.now_wall(),
                          boot_id=C.boot_id(), host=C.host_addresses()),
                approvals={k: C.approval_status(k) for k in ("A", "B")}, contract=panel.contract(), components=panel.components(),
                lio=dict(na, odom={}, accumulation={}, track_raw=[]), e5=dict(na, scene={}, images={}, regions=[]),
                resources=C.resources(),
                research={k: dict(na) for k in ("E2", "E3", "E4", "E5_SEES_E4", "E5", "OCR_VLM", "SEMANTIC_MAP", "E6", "E7")},
                last_experiment=next(iter(history_list()), None))


def history_list(limit=30):
    out = []
    for d in sorted(EXPERIMENTS.glob("X*_*"), reverse=True)[:limit]:
        x = C.read_json(d / "experiment.json") or ((C.read_json(EXP_FILE) or {}) if (C.read_json(EXP_FILE) or {}).get("exp_id") == d.name else {})
        out.append(dict(exp_id=d.name, created=x.get("created"), ended=x.get("ended"), state=x.get("state") or "UNKNOWN",
                        end_reason=x.get("end_reason"), building=x.get("building"), siglip=x.get("siglip"), failed=x.get("failed") or [],
                        archived=(d / "state").is_dir(), historical=True))
    return out


def history_detail(e):
    d = EXPERIMENTS / e
    x = C.read_json(d / "experiment.json") or {}
    st = d / "state"
    e2, e3, e4 = (C.read_json(st / n) or {} for n in ("e2/e2_state.json", "e3_state.json", "e4_state.json"))
    lio = (C.read_json(st / "lio_state.json") or {}).get("odom") or {}
    tl = []
    if (d / "timeline.jsonl").exists():
        tl = [json.loads(ln) for ln in open(d / "timeline.jsonl") if ln.strip()][-80:]
    return dict(exp_id=e, historical=True, summary={k: x.get(k) for k in ("created", "ended", "state", "end_reason", "building", "siglip",
                                                                          "failed", "stop_failed", "modules")},
                e2=dict(status=e2.get("status"), pass_id=e2.get("pass_id"), floors_done=e2.get("floors_done"),
                        floors_failed=e2.get("floors_failed"), n_candidates=len(e2.get("candidates") or [])),
                e3=dict(alive=e3.get("n_alive"), active=e3.get("n_active"), keyframes=e3.get("n_keyframes")),
                e4=dict(state=e4.get("state"), reasons=e4.get("reasons"), best_floor=(e4.get("metrics") or {}).get("best_floor")),
                odom=dict(n=lio.get("n"), hz_last=lio.get("hz_10s")), timeline=tl,
                gt=(EVAL / e / "gt.jsonl").exists(), detections=(EVAL / e / "detections.jsonl").exists(),
                evaluation=C.read_json(EVAL / e / "evaluation.json"))


# ------------------------------------------------------------------ semantic maps for GT (read-only)
def floor_dir(b, f):
    d = SEM_DIR[b](f)
    if (d / "base_map.png").exists():
        return d
    loc = WEB.MAPS_ROOT / WEB.MAP_DIRS[b] / f"{f}F" / "maps"
    return loc if (loc / "base_map.png").exists() else None


_SEM = {}


def semantic_meta(b, f):
    k = (b, f)
    if k not in _SEM:
        d = floor_dir(b, f)
        js = d / f"{d.name}_semantic_map.json"
        from PIL import Image
        size = Image.open(d / "base_map.png").size
        meta = dict(source=str(d / "base_map.png"), size=size, semantic=js.exists(), resolution_m_per_px=None, labels=[],
                    sha256=C.sha256(d / "base_map.png"))
        if js.exists():
            j = json.loads(js.read_text())
            meta["resolution_m_per_px"] = (j.get("map_meta") or {}).get("resolution_m_per_px")
            meta["labels"] = [dict(t=o.get("value") or o.get("display_name"), px=o.get("px")) for o in j.get("semantic_objects") or []
                              if o.get("px") and o.get("type") in ("room_label",)][:400]
            meta["sha256"] = C.sha256(js)
            meta["semantic_json"] = str(js)
        _SEM[k] = meta
    return _SEM[k]


_STH = {}


def sem_thumb(b, f, max_w=1400):
    k = (b, f)
    if k not in _STH:
        from PIL import Image
        import io
        d = floor_dir(b, f)
        im = Image.open(d / "base_map.png").convert("L")
        fac = min(1.0, max_w / im.size[0])
        th = im.resize((max(1, int(im.size[0] * fac)), max(1, int(im.size[1] * fac))))
        bio = io.BytesIO()
        th.save(bio, "PNG")
        _STH[k] = (bio.getvalue(), fac)
    return _STH[k]


def floors_of(b):
    root = WEB.MAPS_ROOT / WEB.MAP_DIRS[b]
    fl = sorted(int(p.name[:-1]) for p in root.iterdir() if p.name.endswith("F") and p.name[:-1].isdigit()) if root.exists() else []
    return [dict(floor=f, semantic=(SEM_DIR[b](f) / f"{SEM_DIR[b](f).name}_semantic_map.json").exists()) for f in fl]


# ------------------------------------------------------------------ HTTP
def make_handler(ctrl, panel, allowed_hosts):
    read_only = WEB.make_handler(panel)
    views = {}
    gate = DeviceGate()

    class H(read_only):
        server_version = "LIVE2Console/1"

        def _json(self, code, obj, cookie=None):
            body = jdump(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if cookie is not None:
                self.send_header("Set-Cookie", cookie)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _sid(self):
            for part in (self.headers.get("Cookie") or "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == COOKIE:
                    return v
            return None

        def _net_ok(self):
            """Login and control only from the robot itself or the Tailscale network (WireGuard-encrypted). A LAN
            client (e.g. 192.168.x over plain HTTP) gets the read-only view; no password or cookie crosses the LAN."""
            try:
                ip = ipaddress.ip_address(self.client_address[0])
            except ValueError:
                return False
            return ip.is_loopback or ip in TAILNET

        def _cookie(self, sess):
            c = f"{COOKIE}={sess['sid']}; HttpOnly; SameSite=Strict; Path=/"
            return c + (f"; Max-Age={int(sess['max_age'])}" if sess.get("max_age") else "")

        def _ident(self):
            if not hasattr(self, "_id_cache"):
                try:
                    local = self.connection.getsockname()[0]
                except OSError:
                    local = None
                self._id_cache = gate.identify(self.client_address[0], local)
            return self._id_cache

        def _device_session(self):
            """Device mode: a valid device session cookie bound to THIS node (auto-issued; nothing typed by the user)."""
            idn, sid = self._ident(), self._sid()
            s = ctrl.session(sid)
            if s and s.get("kind") == "device" and s.get("device") == idn.get("actor"):
                return sid, s, None
            if not idn.get("control"):
                return None, None, None
            sess = ctrl.auth.create(idn["name"], True, self.headers.get("User-Agent") or "", self.client_address[0], kind="device",
                                    device=idn["actor"])
            return sess["sid"], ctrl.session(sess["sid"]), self._cookie(sess)

        def _session_info(self):
            if auth_mode() == "device":
                idn = self._ident()
                sid, s, cookie = self._device_session()
                ap = approval()
                info = dict(mode="device", logged_in=bool(s), read_ok=idn["read"], control_ok=idn["control"], device=idn.get("name"),
                            device_kind=idn["kind"], why=idn.get("why"), csrf=ctrl.auth.csrf(sid) if s else None,
                            trusted_devices=[d.get("name") for d in gate.trusted().values()], control_network_ok=idn["read"],
                            approval_valid=ap["valid"], transport="http")
                return info, cookie
            sid = self._sid()
            s = ctrl.session(sid)
            return dict(mode="account", logged_in=bool(s), user=(s or {}).get("user"), csrf=ctrl.auth.csrf(sid) if s else None,
                        remember=(s or {}).get("remember"), expires=(s or {}).get("expires"), kind=(s or {}).get("kind"),
                        needs_enrollment=ctrl.auth.needs_enrollment(), legacy_token_login=ctrl.legacy_allowed(),
                        control_network_ok=self._net_ok(), transport="http",
                        setup_code_file=str(CTL_DIR / "operator_token"), user_id=AUTH.USER), None

        def _host_ok(self):
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return host in allowed_hosts

        def _origin_ok(self):
            o = self.headers.get("Origin")
            sfs = self.headers.get("Sec-Fetch-Site")
            if sfs and sfs not in ("same-origin", "none"):
                return False
            return o is None or o.split("://", 1)[-1] == self.headers.get("Host")

        def _seen(self, kind, **k):
            """Access diagnostics (page loads, refused hosts): at most one audit row per client IP and kind per minute."""
            key, now = (self.client_address[0], kind), time.time()
            if now - views.get(key, 0) >= 60:
                views[key] = now
                append_jsonl(AUDIT, dict(wall=now, time=C.now_wall(), ip=self.client_address[0], action=kind,
                                         host=self.headers.get("Host"), ua=(self.headers.get("User-Agent") or "")[:120], **k))

        def do_GET(self):
            if not self._host_ok():
                self._seen("host_refused", code=421)
                return self._json(421, dict(ok=False, why="HOST_NOT_ALLOWED", host=self.headers.get("Host"),
                                            use="http://<robot IP>:8770/"))
            path = self.path.split("?")[0]
            q = dict(x.split("=", 1) for x in self.path.split("?", 1)[1].split("&") if "=" in x) if "?" in self.path else {}
            public = ("/", "/index.html", "/legacy", "/api/session")
            if auth_mode() == "device":
                idn = self._ident()
                if path not in public and not idn["read"]:
                    self._seen("read_refused", code=403, why=idn.get("why"))
                    return self._json(403, dict(ok=False, why=idn.get("why") or "NOT_ALLOWED"))
            elif path not in public and not ctrl.session(self._sid()):
                return self._json(401, dict(ok=False, why="LOGIN_REQUIRED"))   # camera images and state only after login
            if path in ("/", "/index.html", "/legacy"):
                self._seen("page_view", code=200, ui="legacy" if path == "/legacy" else "v2")
                page = UI_LEGACY if path == "/legacy" or not UI_V2.exists() else UI_V2
                return self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/state":
                X = ctrl.exp.d
                snap = panel.snapshot() if ctrl.exp.active else idle_snapshot(panel, X)
                snap["meta"].update(experiment_state=X.get("state") or "IDLE", experiment_id=X.get("exp_id") if ctrl.exp.active else None,
                                    live=ctrl.exp.active and not C.FAKE, fake=C.FAKE, idle=not ctrl.exp.active,
                                    note="console: read-only panel + allow-listed control API (no command input)")
                return self._json(200, snap)
            if path in ("/img/live.jpg", "/img/inference.jpg", "/img/map.png", "/img/rgb.jpg") and not ctrl.exp.active:
                return self._send(404, b"NO_EXPERIMENT", "text/plain")          # no old picture while idle
            if path == "/api/ctl/state":
                idn = self._ident()
                actor = idn.get("actor") if auth_mode() == "device" and idn.get("control") else self._sid()
                st = ctrl.state(actor)
                if auth_mode() == "device":
                    st.update(logged_in=True, control_ok=idn["control"], device=idn.get("name"))
                return self._json(200, st)
            if path == "/api/session":
                info, cookie = self._session_info()
                return self._json(200, info, cookie)
            if path == "/api/auth/devices":
                sid = self._sid()
                s = ctrl.session(sid)
                if not s:
                    return self._json(401, dict(ok=False, why="LOGIN_REQUIRED"))
                return self._json(200, dict(ok=True, devices=ctrl.auth.devices(s["user"], s["id"])))
            if path == "/api/history":
                return self._json(200, dict(ok=True, experiments=history_list()))
            if path.startswith("/api/history/"):
                e = path.rsplit("/", 1)[1]
                if not re.match(r"^X[0-9]{8}_[0-9]{6}$", e) or not (EXPERIMENTS / e).is_dir():
                    return self._json(404, dict(ok=False, why="NO_SUCH_EXPERIMENT"))
                return self._json(200, dict(ok=True, historical=True, experiment=history_detail(e)))
            if path == "/api/gt/floors" and q.get("b") in BUILDINGS:
                return self._json(200, dict(building=q["b"], floors=floors_of(q["b"])))
            if path == "/api/gt/meta" and q.get("b") in BUILDINGS and q.get("f", "").isdigit():
                b, f = q["b"], int(q["f"])
                if not floor_dir(b, f):
                    return self._json(404, dict(ok=False))
                m = semantic_meta(b, f)
                return self._json(200, dict(m, factor=sem_thumb(b, f)[1]))
            if path.startswith("/img/sem/"):
                parts = path.split("/")
                if len(parts) == 5 and parts[3] in BUILDINGS and parts[4][:-4].isdigit() and floor_dir(parts[3], int(parts[4][:-4])):
                    return self._send(200, sem_thumb(parts[3], int(parts[4][:-4]))[0], "image/png")
                return self._send(404, b"NO_MAP", "text/plain")
            if path.startswith("/img/snap/"):
                parts = path.split("/")
                if len(parts) == 5 and EXPID.match(parts[3]) and parts[4].endswith(".jpg") and SNAP.match(parts[4][:-4]):
                    p = EVAL / parts[3] / "snapshots" / parts[4]
                    if p.exists():
                        return self._send(200, p.read_bytes(), "image/jpeg")
                return self._send(404, b"NO_SNAPSHOT", "text/plain")
            if path in ("/img/e6live.jpg", "/api/e6/metrics"):   # IMX477 manual-focus view (read-only, loopback aim console)
                import urllib.parse as up
                import e6_manual as M
                if not ctrl.exp.active or not aim_console_alive():
                    return self._send(404, b"NO_E6_CONSOLE", "text/plain")
                if path == "/img/e6live.jpg":
                    w = max(320, min(1920, int(q.get("w", "960")) if q.get("w", "960").isdigit() else 960))
                    code, body, hd = aim_console_get(f"/live2/preview.jpg?w={w}")
                    if code != 200:
                        return self._send(503, b"NO_FRAME", "text/plain")
                    self.send_response(200)
                    for k, v in [("Content-Type", "image/jpeg"), ("Cache-Control", "no-store"), ("Content-Length", str(len(body)))] + \
                            [(k, hd[k]) for k in ("X-Seq", "X-Frame-Age-S", "X-Frame-Size") if k in hd]:
                        self.send_header(k, v)
                    self.end_headers()
                    self.wfile.write(body)
                    return None
                roi = up.unquote(q.get("roi", ""))
                if roi and M.parse_roi(roi) is None:
                    return self._json(400, dict(ok=False, why="BAD_ROI"))
                code, body, _ = aim_console_get("/live2/metrics?" + up.urlencode(dict(roi=roi, auto="1")), timeout=3.0)
                try:
                    m = json.loads(body.decode())
                except (ValueError, UnicodeDecodeError):
                    return self._json(503, dict(ok=False, why="NO_METRICS"))
                oid = (M.read_snapshot(C.STATE) or {}).get("observation_id")
                v = (m.get("sharp") or {}).get("roi")
                if oid and v is not None:
                    k = (oid, roi)
                    E6_BEST[k] = max(E6_BEST.get(k) or 0.0, float(v))
                    m["best_roi"] = E6_BEST[k]
                    if len(E6_BEST) > 200:
                        E6_BEST.pop(next(iter(E6_BEST)))
                return self._json(code, m)
            if path in ("/api/gt/list", "/api/det/list"):
                fn = "gt.jsonl" if path == "/api/gt/list" else "detections.jsonl"
                p = ctrl.exp.eval_dir() / fn
                rows = [json.loads(x) for x in open(p)][-30:] if p.exists() else []
                return self._json(200, dict(rows=rows, store=str(p)))
            return super().do_GET()

        def do_POST(self):
            ip = self.client_address[0]
            path = self.path.split("?")[0]
            sid = self._sid()

            def done(code, obj, action, body=None, cookie=None):
                append_jsonl(AUDIT, dict(wall=time.time(), time=C.now_wall(), ip=ip, sid=sid_hash(sid), action=action, code=code,
                                         request_id=(body or {}).get("request_id"), ok=obj.get("ok"), why=obj.get("why"),
                                         params={k: v for k, v in (body or {}).items() if k not in ("token", "request_id", "password", "current", "new",
                                                                                        "setup_code", "samples")}))
                return self._json(code, obj, cookie)

            if not self._host_ok():
                return done(421, dict(ok=False, why="HOST_NOT_ALLOWED"), path)
            if not self._origin_ok():
                return done(403, dict(ok=False, why="CROSS_ORIGIN_REFUSED"), path)
            if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
                return done(415, dict(ok=False, why="JSON_ONLY"), path)
            n = int(self.headers.get("Content-Length") or 0)
            if n > 65536:
                return done(413, dict(ok=False, why="BODY_TOO_LARGE"), path)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                assert isinstance(body, dict)
            except (ValueError, AssertionError):
                return done(400, dict(ok=False, why="BAD_JSON"), path)
            ua = self.headers.get("User-Agent") or ""
            actor = sid
            if auth_mode() == "device":
                idn = self._ident()
                if not idn["control"]:
                    return done(403, dict(ok=False, why=idn.get("why") or "DEVICE_NOT_TRUSTED"), path, body)
                if path.startswith("/api/auth/") or path in ("/api/login", "/api/logout"):
                    return done(404, dict(ok=False, why="ACCOUNT_LOGIN_DISABLED (device mode: 로그인 없음)"), path, body)
                s = ctrl.session(sid)
                if not s or s.get("kind") != "device" or s.get("device") != idn["actor"]:
                    return done(401, dict(ok=False, why="DEVICE_SESSION_REQUIRED (페이지를 새로 여십시오)"), path, body)
                if not ctrl.auth.csrf_ok(sid, self.headers.get("X-CSRF-Token")):
                    return done(403, dict(ok=False, why="CSRF_TOKEN_MISMATCH"), path, body)
                return self._control(path, body, idn["actor"], done, ip)
            if not self._net_ok():
                return done(403, dict(ok=False, why="CONTROL_ONLY_VIA_TAILSCALE (이 주소는 보기 전용 — Tailscale 주소로 접속)"), path, body)
            if path == "/api/auth/enroll":
                lk = ctrl.auth.locked(ip, "enroll")
                if lk:
                    return done(429, dict(ok=False, why=lk), "enroll", body)
                why = ctrl.auth.enroll(body.get("user"), body.get("password"), body.get("setup_code"))
                if why:
                    if why == "SETUP_CODE_WRONG":
                        ctrl.auth.note_fail(ip, "enroll")
                    return done(400 if why.startswith("WEAK") else 403, dict(ok=False, why=why), "enroll", body)
                sess = ctrl.auth.create(AUTH.USER, bool(body.get("remember")), ua, ip)
                return done(200, dict(ok=True, csrf=sess["csrf"], user=AUTH.USER), "enroll", body, cookie=self._cookie(sess))
            if path in ("/api/auth/login", "/api/login"):
                if body.get("token") is not None and body.get("password") is None:
                    sess, why = ctrl.login_legacy(ip, body.get("token"), ua)
                else:
                    sess, why = ctrl.auth.login(ip, body.get("user"), body.get("password"), body.get("remember"), ua)
                if not sess:
                    return done(429 if (why or "").startswith("LOCKED") else 401, dict(ok=False, why=why), "login", body)
                return done(200, dict(ok=True, csrf=sess["csrf"], user=sess["user"], remember=sess["remember"]), "login", body,
                            cookie=self._cookie(sess))
            s = ctrl.session(sid)
            if not s:
                return done(401, dict(ok=False, why="LOGIN_REQUIRED"), path, body)
            if not ctrl.auth.csrf_ok(sid, self.headers.get("X-CSRF-Token")):
                return done(403, dict(ok=False, why="CSRF_TOKEN_MISMATCH"), path, body)
            gone = f"{COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"
            if path in ("/api/auth/logout", "/api/logout"):
                ctrl.auth.logout(sid)
                return done(200, dict(ok=True), "logout", body, cookie=gone)
            if path == "/api/auth/logout_all":
                ctrl.auth.logout_all(s["user"])
                return done(200, dict(ok=True), "logout_all", body, cookie=gone)
            if path == "/api/auth/password":
                why = ctrl.auth.change_password(s["user"], body.get("current"), body.get("new"))
                if why:
                    return done(400, dict(ok=False, why=why), "password_change", body)
                sess = ctrl.auth.create(s["user"], bool(s.get("remember")), ua, ip)
                return done(200, dict(ok=True, csrf=sess["csrf"], note="다른 모든 기기는 로그아웃되었습니다"), "password_change", body,
                            cookie=self._cookie(sess))
            if path == "/api/auth/revoke":
                n = ctrl.auth.revoke(s["user"], str(body.get("id") or "")[:12]) if len(str(body.get("id") or "")) >= 8 else 0
                return done(200, dict(ok=bool(n), revoked=n), "revoke_device", body)
            return self._control(path, body, actor, done, ip)

        def _control(self, path, body, actor, done, ip):
            """Common control path after authentication; `actor` = owner identity (device node or account session)."""
            if path == "/api/ui/telemetry":
                rows = [r for r in (body.get("samples") or [])[:50] if isinstance(r, dict)]
                for r in rows:
                    append_jsonl(CTL_DIR / "ui_latency.jsonl", dict(wall=time.time(), ip=ip, **{k: r.get(k) for k in
                                 ("kind", "age_ms", "fetch_ms", "render_ms", "stamp_age_ms", "viewport")}))
                return self._json(200, dict(ok=True, n=len(rows)))
            if not path.startswith("/api/ctl/"):
                return done(404, dict(ok=False, why="NO_SUCH_ENDPOINT"), path, body)
            action = path[len("/api/ctl/"):]
            rid = body.get("request_id")
            if not isinstance(rid, str) or not RID.match(rid):
                return done(400, dict(ok=False, why="REQUEST_ID_REQUIRED"), action, body)
            key = f"{sid_hash(actor)}:{rid}"                                 # per operator: no answer leaks to another one
            with ctrl.slock:
                if key in ctrl.seen:
                    prev = ctrl.seen[key]
                    return done(200, dict(prev, duplicate=True, why="DUPLICATE_REQUEST_ID (not executed again)"), action, body)
                ctrl.seen[key] = dict(ok=False, why="IN_PROGRESS")          # reserved: a concurrent copy is refused too
            code, obj = self.dispatch(action, body, rid, actor)
            with ctrl.slock:
                ctrl.seen[key] = obj
                if len(ctrl.seen) > 2000:
                    for k in list(ctrl.seen)[:500]:
                        ctrl.seen.pop(k, None)
            return done(code, obj, action, body)

        def dispatch(self, action, body, rid, sid):
            X = ctrl.exp
            if action == "approval_grant":
                return self._approval_grant(body)
            if action in ("standing_suspend", "standing_resume", "standing_revoke", "standing_exp_max"):
                return self._standing(action, body)
            if action == "approval_revoke":
                if X.active:
                    X.event("approval_revoked_from_web")
                rc, out = run_ctl(["record-approval", "A", "--text-file", str(self._revoke_text()), "--minutes", "0", "--modules", "",
                                   "--recorded-by", f"web:{self._ident().get('name')}"], timeout=30)
                return (200, dict(ok=True, note="승인 취소됨 (실행 중인 실험은 감시기가 정리)")) if rc == 0 else (500, dict(ok=False, why=out[-200:]))
            if action == "gt_save":
                return ctrl.gt_save(body)
            if action == "det_snapshot":
                return ctrl.det_snapshot()
            if action == "det_label":
                return ctrl.det_label(body)
            if action == "takeover":
                if auth_mode() == "device":
                    if body.get("confirm") is not True:
                        return 400, dict(ok=False, why="CONFIRM_REQUIRED (제어권 인수 확인)")
                elif not ctrl.auth.check_password(AUTH.USER, body.get("password")):
                    return 401, dict(ok=False, why="PASSWORD_REQUIRED_FOR_TAKEOVER")
                X.d["owner"] = sid_hash(sid)
                X.event("takeover", by=sid_hash(sid))
                return 200, dict(ok=True)
            control = ("start", "end", "pause", "resume", "siglip_on", "siglip_off", "e2_start", "e2_research", "start_missing",
                       "e6_operator_target", "e6_manual", "e6_settle")
            if action not in control:
                return 404, dict(ok=False, why="ACTION_NOT_ALLOWED")
            if action == "start":
                if X.active:
                    return 409, dict(ok=False, why=f"EXPERIMENT_ALREADY_ACTIVE {X.d.get('exp_id')} ({X.d.get('state')})")
                b = body.get("building")
                if b not in BUILDINGS:
                    return 400, dict(ok=False, why="BAD_BUILDING")
                ap = approval()
                if not ap["valid"]:
                    return 403, dict(ok=False, why=f"APPROVAL_INVALID {ap['why']}")
                pf = preflight()
                if not pf["ok"]:
                    bad = [k for k, v in pf["items"].items() if not v["ok"]]
                    return 409, dict(ok=False, why=f"PREFLIGHT_NOT_READY {bad} (자원 보호로 실행 대기)")
                missing = [m for m in START_ORDER if APPROVAL_MODULE.get(m, m) and m not in ap.get("modules", [])
                           and (m != "observe" or body.get("siglip"))]
                if missing:
                    return 403, dict(ok=False, why=f"MODULES_NOT_APPROVED {missing}")
                ok_ar, why_ar, _ = EA.start_gate(EXPERIMENTS, EA.live_sources(L, STORE, HUB))
                if not ok_ar:                    # the previous run's volatile records would be overwritten by this start
                    return 409, dict(ok=False, why=f"ARCHIVE_REQUIRED {why_ar} (이전 실험 기록 보존 실패 — 시작 보류)")
                if REQUIRE_START_GT and not C.read_json(PENDING_START):
                    return 409, dict(ok=False, why="START_GT_REQUIRED (평가용 시작 위치·방향을 먼저 지정)")
                siglip = True                    # R39: SigLIP is part of the one mode; whether a sign may stop the robot is E4's choice (auto_fsm), not the operator's
                loc_mode = body.get("loc_mode") or AUTO_MODE
                if loc_mode not in LOC_MODES:
                    return 400, dict(ok=False, why=f"LOC_MODE_REMOVED {loc_mode!r} (구역 선택은 없어졌습니다: 로봇이 구조적 모호성을 스스로 판단합니다)")
                inp = body.get("input") or "l1"
                if inp not in ("l1", "body"):
                    return 400, dict(ok=False, why=f"BAD_INPUT {inp!r} (l1 | body)")
                e6m = body.get("e6_mode") or "AUTO"
                if e6m not in ("AUTO", "MANUAL_SETTLE"):
                    return 400, dict(ok=False, why=f"BAD_E6_MODE {e6m!r} (AUTO | MANUAL_SETTLE)")
                consent = body.get("pantilt_consent") or None   # 2026-10-31: per-experiment pan-tilt use (operator, this tablet)
                if consent is not None:
                    acks = consent.get("acks") or {}
                    need = [k for k in PANTILT_CONSENT_ACKS if acks.get(k) is not True]
                    if consent.get("use") is not True or need:
                        return 400, dict(ok=False, why=f"PANTILT_CONSENT_INCOMPLETE {need} (모든 확인 항목을 체크해야 사용 동의가 성립)")
                j = ctrl.submit("start", lambda job: ctrl.a_start(job, b, siglip, rid, sid, inp, e6m, consent, loc_mode), rid)
                return (202, dict(ok=True, job=rid)) if j else (409, dict(ok=False, why="BUSY (another control job is running)"))
            if not X.active:
                return 409, dict(ok=False, why="NO_ACTIVE_EXPERIMENT")
            if not ctrl.owns(sid):
                return 423, dict(ok=False, why="EXPERIMENT_OWNED_BY_ANOTHER_SESSION (takeover needs the password)")
            st = X.d.get("state")
            if action == "pause" and st not in ("RUNNING", "PARTIAL"):
                return 409, dict(ok=False, why=f"CANNOT_PAUSE_IN_{st}")
            if action == "resume" and st != "PAUSED":
                return 409, dict(ok=False, why=f"CANNOT_RESUME_IN_{st}")
            if action in ("siglip_on", "e2_start", "start_missing", "e2_research") and st == "PAUSED":
                return 409, dict(ok=False, why="PAUSED (resume first)")
            if action in ("siglip_on", "e2_start", "start_missing"):
                ap = approval()
                if not ap["valid"]:
                    return 403, dict(ok=False, why=f"APPROVAL_INVALID {ap['why']}")
            if action == "e2_research":
                return 200, ctrl.a_e2_research(rid)
            if action == "e6_operator_target":
                r = ctrl.a_e6_operator_target(rid, body)
                return (200 if r["ok"] else 409), r
            if action == "e6_manual":
                r = ctrl.a_e6_manual(rid, body, sid)
                return (202 if r["ok"] else 409), r
            if action == "e6_settle":
                r = ctrl.a_e6_settle(rid, body, sid)
                return (202 if r["ok"] else 409), r
            fn = dict(end=lambda job: ctrl.a_end(job, "OPERATOR_END"), pause=lambda job: ctrl.a_pause(job, rid),
                      resume=lambda job: ctrl.a_resume(job, rid), siglip_on=lambda job: ctrl.a_siglip(job, True),
                      siglip_off=lambda job: ctrl.a_siglip(job, False), e2_start=lambda job: ctrl.a_e2_start(job),
                      start_missing=lambda job: ctrl.a_start_missing(job))[action]
            j = ctrl.submit(action, fn, rid)
            return (202, dict(ok=True, job=rid)) if j else (409, dict(ok=False, why="BUSY (another control job is running)"))

        def _approval_grant(self, b):
            """Web approval A: only from a control-capable device (trusted tablet / robot itself), fixed scope, explicit ack."""
            idn = self._ident() if auth_mode() == "device" else dict(control=True, name="account-session")
            if not idn.get("control"):
                return 403, dict(ok=False, why="DEVICE_NOT_TRUSTED")
            if ctrl.exp.active:
                return 409, dict(ok=False, why="EXPERIMENT_ACTIVE (실험 중에는 승인 범위·시간을 바꾸지 않습니다)")
            if b.get("ack") is not True:
                return 400, dict(ok=False, why="ACK_REQUIRED (범위 확인 체크가 필요합니다)")
            if b.get("boot_id") != C.boot_id():
                return 400, dict(ok=False, why=f"BOOT_ID_MISMATCH (현재 {C.boot_id()[:8]})")
            mods = b.get("modules") or list(WEB_APPROVAL_MODULES)
            if not isinstance(mods, list) or not mods or any(m not in WEB_APPROVAL_MODULES for m in mods):
                return 400, dict(ok=False, why=f"MODULES_NOT_ALLOWED_FROM_WEB (허용: {list(WEB_APPROVAL_MODULES)})")
            mins = b.get("minutes")
            if not isinstance(mins, int) or not 5 <= mins <= WEB_APPROVAL_MAX_MIN:
                return 400, dict(ok=False, why=f"MINUTES_OUT_OF_RANGE (5–{WEB_APPROVAL_MAX_MIN})")
            dev = idn.get("name") + (f" ({idn.get('stable_id')})" if idn.get("stable_id") else "")
            rc, out = record_web_approval(C.boot_id(), mods, mins, dev)
            if rc != 0:
                return 500, dict(ok=False, why=out[-300:])
            ap = approval()
            return 200, dict(ok=ap["valid"], approval=dict(valid=ap["valid"], remaining_s=ap.get("remaining_s"), modules=ap.get("modules"),
                                                          file=ap.get("file")), device=dev)

        def _standing(self, action, b):
            """Standing authorization from the web: suspend / resume (suspended only) / revoke / per-experiment maximum.
            Never grant or widen (CLI only). Control-capable device only; revoke needs an explicit ack and ends a running run."""
            import standing_auth as SA
            idn = self._ident() if auth_mode() == "device" else dict(control=True, name="account-session")
            if not idn.get("control"):
                return 403, dict(ok=False, why="DEVICE_NOT_TRUSTED")
            who = f"web:{idn.get('name')}"
            try:
                if action == "standing_exp_max":
                    mins = b.get("minutes")
                    if not isinstance(mins, int):
                        return 400, dict(ok=False, why="MINUTES_INT_REQUIRED")
                    s = C.standing().set_exp_max_s(mins * 60, who)
                    ctrl.exp.event("standing_experiment_max_set", seconds=s, by=who)
                    return 200, dict(ok=True, experiment_max_s=s, applies="next experiment (and the running one's watchdog)")
                if action == "standing_revoke" and b.get("ack") is not True:
                    return 400, dict(ok=False, why="ACK_REQUIRED (철회 후 재허용은 로봇에서 ctl standing-grant 로만 가능)")
                state = dict(standing_suspend="SUSPENDED", standing_resume="ACTIVE", standing_revoke="REVOKED")[action]
                r = C.standing().set_state(state, who, str(b.get("why") or "")[:200])
            except (PermissionError, ValueError) as e:
                return 409, dict(ok=False, why=str(e))
            ctrl.exp.event("standing_" + state.lower(), by=who, grant_id=r["grant_id"])
            return 200, dict(ok=True, state=state, grant_id=r["grant_id"], dev_cap_s=SA.DEV_CAP_S)

        def _revoke_text(self):
            d = CTL_DIR / "web_approvals"
            d.mkdir(parents=True, exist_ok=True)
            t = d / f"A_web_revoke_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            t.write_text(f"[LIVE-FIELD2 승인 A 취소 — 웹]\n기기 {self._ident().get('name')} 가 {C.now_wall()} 에 현재 승인 A 를 취소함 (모듈 없음, 즉시 만료).\n")
            return t

        do_PUT = do_DELETE = do_PATCH = lambda self: self._json(405, dict(ok=False, why="METHOD_NOT_ALLOWED"))

    return H


def _start_missing(self, job):
    """Retry the modules that are not running (after a PARTIAL start); same gates as a fresh start."""
    X = self.exp
    failed = []
    for m in START_ORDER:
        if (m == "observe" and not X.d.get("siglip")) or m in ((LOC_MODES.get(X.d.get("loc_mode")) or {}).get("skip") or ()):
            continue
        if not alive(m) and not self.start_module(job, m, X.d["building"]):
            failed.append(m)
            if m in CRITICAL:
                break
    X.d["state"] = "RUNNING" if not failed else "PARTIAL"
    X.d["failed"] = failed
    X.event("start_missing", failed=failed)
    return dict(ok=not failed, failed=failed)


Controller.a_start_missing = _start_missing


def tailscale_names():
    """MagicDNS names of this machine (e.g. ubuntu.tailXXXX.ts.net) so a tablet may use them; read-only query."""
    try:
        d = json.loads(subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True, timeout=5).stdout)
        dn = ((d.get("Self") or {}).get("DNSName") or "").rstrip(".")
        return {dn, dn.split(".")[0]} - {""}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--max-s", type=float, required=True)
    ap.add_argument("--root", default=None, help="must equal LIVE1_ROOT (sanity check)")
    a = ap.parse_args()
    if a.root and Path(a.root).resolve() != L.resolve():
        sys.exit(f"REFUSED: --root {a.root} != LIVE1_ROOT {L}")
    ctrl = Controller()
    if ctrl.exp.d.get("state") in ("STARTING", "STOPPING"):           # console restarted mid-job: show it honestly
        ctrl.exp.d["state"] = "PARTIAL"
        ctrl.exp.event("console_restarted_during_job")
    panel = WEB.Panel(C.STATE, SHM, test_only=C.FAKE, trial_id=ctrl.exp.d.get("exp_id"))
    ctrl.panel = panel                                                 # READY gate reads the same state as the page
    hosts = {"localhost", "127.0.0.1", "::1", os.uname().nodename} | {v for k, v in C.host_addresses().items() if k != "hostname"}
    hosts |= tailscale_names()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(ctrl, panel, hosts))
    srv.daemon_threads = True
    threading.Thread(target=ctrl.watchdog, daemon=True).start()
    threading.Thread(target=ctrl.ready_loop, daemon=True).start()
    t_end = time.time() + a.max_s

    def end(*_):
        ctrl.stop = True
        threading.Thread(target=srv.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, end)
    signal.signal(signal.SIGINT, end)
    threading.Thread(target=lambda: (time.sleep(max(0.0, t_end - time.time())), end()), daemon=True).start()
    print(f"LIVE2_CONSOLE listening {a.host}:{a.port} fake={C.FAKE} max_s={a.max_s:.0f} hosts={sorted(hosts)}", flush=True)
    srv.serve_forever(poll_interval=0.5)
    srv.server_close()
    print("LIVE2_CONSOLE stopped (experiment modules are NOT stopped by the console exit; they keep their own limits)", flush=True)


if __name__ == "__main__":
    main()
