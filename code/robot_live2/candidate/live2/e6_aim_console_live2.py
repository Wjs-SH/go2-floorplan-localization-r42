#!/usr/bin/env python3
"""e6_aim_console_live2.py — runs the UNCHANGED student aim_console with in-memory focus adapters: by default the student
KCI stepwise itself behind the LIVE2 guards (e6_af_student), or explicitly the LIVE2 Focus V2 search (e6_focus_v2). Pan-tilt calibration, aim, HOME, capture, the r1 HTTP
contract and the single DeviceOwner worker are the student's. Same command-line arguments as aim_console.py.

Starting this opens /dev/ttyUSB0 like the student console: DTR resets the Nano (boot pose 95.4/157.2, then HOME).
Start it only through e6_aim_console_ctl.py (single instance, port free, approval B) — never next to another console.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KC = HERE.parent / "external" / "sci_event" / "code"
sys.path.insert(0, str(KC))
sys.path.insert(0, str(HERE))

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import aim_console as AC  # noqa: E402  student module, unchanged
import e6_focus_v2 as F2  # noqa: E402

# LIVE2_AF_STRATEGY (set by e6_aim_console_ctl.py):
#   student_stepwise (default, 2026-10-22 directive) — the student KCI stepwise, unchanged, behind e6_af_student guards
#   focus_v2                                          — the LIVE2 Focus V2 search (alternative, explicit choice only)
STRATEGIES = ("student_stepwise", "focus_v2")

# 2026-10-22 final audit: the student console serves 0.0.0.0:8088 and its own routes (/aim /focus /home and the diagnostics
# /goto /trim /tilt_sweep /pin_test /focus_profile /focus_speed_test /aim_marker) move hardware with NO approval check — the
# approval B gate lived only in the precision bridge. Under LIVE2 (in memory, student files unchanged):
#   * the HTTP server binds 127.0.0.1 only (the bridge and e6_aim_console_ctl are local);
#   * the diagnostic / manual-motion routes answer 403;
#   * R1Hardware.move / home need a valid approval B ('pantilt', this boot) at that moment, R1Hardware.focus additionally the
#     bridge's focus_allowed (AUTOFOCUS_V2, mechanical check PASSED, profile id, verification flag, fault cleared).
#     A refusal raises -> the student DeviceOwner marks the device UNKNOWN (no further command until an operator check).
DIAG_ROUTES = ("/goto", "/trim", "/tilt_sweep", "/pin_test", "/focus_profile", "/focus_speed_test", "/aim_marker")


class Live2Refused(RuntimeError):
    pass


def approval_gate(kind):
    """-> (ok, why) for a hardware action of `kind` ('move' | 'home' | 'focus') at this moment."""
    sys.path.insert(0, str(HERE.parent / "live1"))
    import live1_common as C
    import e6_precision_bridge as B
    if kind == "focus" and os.environ.get("LIVE2_FOCUS_MODE") == "MANUAL_FIXED_NO_MOTOR":
        return False, "MANUAL_FIXED_NO_MOTOR: the DC focus motor is excluded from this console (lens set by hand)"
    ap = C.approval_status("B", "pantilt")
    if not ap.get("valid"):
        return False, f"approval B {ap.get('why')}"
    if kind == "focus":
        return B.focus_allowed(ap)
    return True, None


def install_guards(AC, gate=approval_gate, audit=None):
    audit = audit or (lambda *a, **k: None)
    cc = AC.cc
    if not getattr(cc.Server, "_live2_local", False):
        base = cc.Server

        class LocalServer(base):
            _live2_local = True

            def __init__(self, addr, handler, *a, **k):
                super().__init__(("127.0.0.1", addr[1]), handler, *a, **k)
        cc.Server = LocalServer
    H = AC.Handler
    if not getattr(H.do_GET, "_live2", False):
        orig_get = H.do_GET

        def do_GET(self):
            import urllib.parse
            if urllib.parse.urlparse(self.path).path in DIAG_ROUTES:
                audit("REFUSED_DIAG_ROUTE", path=self.path)
                return self.reply(b"LIVE2: diagnostic / manual motion route disabled (use the E6 bridge)", code=403)
            return orig_get(self)
        do_GET._live2 = True
        H.do_GET = do_GET
    R = AC.R1Hardware
    for name, kind in (("move", "move"), ("home", "home"), ("focus", "focus")):
        orig = getattr(R, name)
        if getattr(orig, "_live2", False):
            continue

        def guarded(self, *a, _orig=orig, _kind=kind, **k):
            ok, why = gate(_kind)
            if not ok:
                audit("REFUSED_AT_CONSOLE", kind=_kind, why=why)
                raise Live2Refused(f"LIVE2: {_kind} refused at the console: {why}")
            return _orig(self, *a, **k)
        guarded._live2 = True
        guarded.__wrapped__ = orig
        setattr(R, name, guarded)


# ------------------------------------------------------------------ FOCUS_DIAGNOSTIC (2026-10-22)
# The general AF stays behind focus_allowed + the persistent fault lock. A latched lock must not make the CAUSE undiagnosable,
# so this is a separate, much smaller grant: approval B (this boot) with limits.focus_diagnostic = {...}. It never enables AF
# (focus_mode stays whatever B says), never clears the lock, never aims or HOMEs. Everything is enforced HERE, in the console
# process that owns the serial port, through the student's single DeviceOwner worker (serialised with aim / focus / HOME):
#   hard caps (approval may only lower): PWM <= 160, pulse <= 200 ms, total <= 800 ms, <= 4 pulses, <= 25 s, 1 run per
#   approval record (persistent marker) and per console process; fixed plan +,+,-,- (both directions, reversal lash seen);
#   before / after every pulse a FRESH IMX477 frame (sign ROI + centre + full-frame sharpness, motion-image change), the raw
#   Nano reply lines, send time; a missing / ERR Nano reply or any exception -> F0, stop, no further pulse; F0 always at the end.
DIAG_HARD = dict(pwm_max=160, pulse_ms_max=200, total_ms_max=800, pulses_max=4, wall_s_max=25.0)
DIAG_RUNS = HERE.parents[1] / "runtime" / "e6_focus_diag_runs"
DIAG_STATE = dict(runs=0)


def diag_limits(ap):
    """approval B -> (limits, None) or (None, why). Caps = min(hard caps, approval)."""
    if not ap.get("valid"):
        return None, f"approval B {ap.get('why')}"
    g = (ap.get("limits") or {}).get("focus_diagnostic")
    if not isinstance(g, dict):
        return None, "approval B has no focus_diagnostic grant"
    lim = {k: min(float(v), float(g.get(k, v))) for k, v in DIAG_HARD.items()}
    if DIAG_STATE["runs"] >= 1:
        return None, "focus diagnostic already run in this console process"
    marker = DIAG_RUNS / (Path(ap["file"]).stem + ".json")
    if marker.exists():
        return None, f"focus diagnostic already run under {Path(ap['file']).name}"
    return dict(lim, marker=str(marker), approval_file=ap["file"]), None


def diag_plan(lim):
    ms = int(min(lim["pulse_ms_max"], lim["total_ms_max"] // 4))
    plan = [(1, ms), (1, ms), (-1, ms), (-1, ms)][:int(lim["pulses_max"])]
    assert sum(m for _, m in plan) <= lim["total_ms_max"] and all(m <= lim["pulse_ms_max"] for _, m in plan)
    return plan


def _sharp_set(frame, roi):
    import cv2
    import numpy as np
    import e6_af_student as AFS
    f = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    lv = lambda im: float(cv2.Laplacian(im, cv2.CV_64F).var())                    # noqa: E731  student formula
    s_roi, small, luma, _ = AFS._measure_roi(frame, roi)
    return dict(roi=round(s_roi, 2), centre=round(lv(g[180:540, 320:960]), 2), full=round(lv(g), 2), luma=round(luma, 1)), \
        small, cv2.resize(g, (160, 120)).astype(np.float32)


def run_focus_diag(a, lim, tok, clock=time, log=lambda *_: None):
    """Executed by the student DeviceOwner worker. -> result dict (also written to the marker file)."""
    import numpy as np
    import e6_focus_v2 as F2
    t0 = clock.time()
    out = dict(kind="LIVE2_FOCUS_DIAGNOSTIC", started=time.strftime("%F %T"), limits={k: lim[k] for k in DIAG_HARD},
               approval_file=lim["approval_file"], plan=diag_plan(lim), pulses=[], aborted=None, fault_lock_present=F2.FAULT_FILE.exists())
    Path(lim["marker"]).parent.mkdir(parents=True, exist_ok=True)
    Path(lim["marker"]).write_text(json.dumps(dict(out, state="RUNNING"), default=str))       # 1 run per approval, even on a crash
    DIAG_STATE["runs"] += 1
    st, nano = a.stats, a.nano
    first = st.fresh_frame(st.get()[0], 1.0)
    if first is None:
        out["aborted"] = "NO_FRESH_IMX477_FRAME"
    else:
        roi, src = F2.sign_roi(first[1])
        out.update(roi=list(roi), roi_source=src)
        prev_seq, (prev_s, prev_small, prev_full) = first[0], _sharp_set(first[1], roi)
        out["sharp_start"], cum = prev_s, 0
        for i, (d, ms) in enumerate(out["plan"]):
            tok.check()
            if clock.time() - t0 > lim["wall_s_max"]:
                out["aborted"] = "WALL_TIME_LIMIT"
                break
            if cum + ms > lim["total_ms_max"]:
                out["aborted"] = "TOTAL_DRIVE_LIMIT"
                break
            ser, lines = getattr(nano, "ser", None), []
            orig_rl = getattr(ser, "readline", None)
            if orig_rl is not None:
                def _tap(*a_, **k_):
                    b_ = orig_rl(*a_, **k_)
                    lines.append(b_.decode(errors="replace").strip() if isinstance(b_, bytes) else str(b_))
                    return b_
                ser.readline = _tap
            cmd = f"F{int(lim['pwm_max']) * d},{ms}"
            t_send = clock.time()
            try:
                nano.cmd(cmd)
            finally:
                if orig_rl is not None:
                    try:
                        del ser.readline
                    except AttributeError:
                        ser.readline = orig_rl
            cum += ms
            clock.sleep(ms / 1000.0 + 0.35)
            fr = st.fresh_frame(prev_seq, 1.0)
            rec = dict(n=i + 1, cmd=cmd, dir=d, pwm=int(lim["pwm_max"]), ms=ms, cum_ms=cum, t_send=round(t_send, 3),
                       nano_reply=[x for x in lines if x][:3], nano_ok=any(x and not x.startswith("ERR") for x in lines),
                       frame_seq_before=prev_seq, sharp_before=prev_s)
            if fr is None:
                rec["frame"] = "STALE"
                out["pulses"].append(rec)
                out["aborted"] = "NO_FRESH_IMX477_FRAME"
                break
            s, small, full = _sharp_set(fr[1], roi)
            rec.update(frame_seq_after=fr[0], frame_rx_mono_ns=fr[2], sharp_after=s,
                       moved_roi=round(float(np.mean(np.abs(small - prev_small))), 2),
                       moved_full=round(float(np.mean(np.abs(full - prev_full))), 2))
            out["pulses"].append(rec)
            log(f"  LIVE2 초점 진단 {rec['n']}: {cmd} 응답 {rec['nano_reply'][:1]} ROI {prev_s['roi']}->{s['roi']} "
                f"변화 {rec['moved_roi']}/{rec['moved_full']}")
            if not rec["nano_ok"]:
                out["aborted"] = "NANO_NO_REPLY_OR_ERR"
                break
            prev_seq, prev_s, prev_small, prev_full = fr[0], s, small, full
    try:
        nano.cmd("F0")
        out["f0_sent"] = True
    except Exception as e:                                                   # noqa: BLE001
        out["f0_sent"] = f"{type(e).__name__}: {e}"
    ps = out["pulses"]
    resp = [p for p in ps if p.get("moved_roi", 0) >= 2.0 or p.get("moved_full", 0) >= 2.0 or
            (p.get("sharp_after") and abs(p["sharp_after"]["roi"] - p["sharp_before"]["roi"]) >= 0.15 * max(p["sharp_before"]["roi"], 1e-6))]
    nano_all = bool(ps) and all(p.get("nano_ok") for p in ps)
    out.update(drive_ms=sum(p["ms"] for p in ps), wall_s=round(clock.time() - t0, 2), image_response_pulses=[p["n"] for p in resp],
               nano_replied_all=nano_all,
               evidence=("A_NANO_NOT_REPLYING" if ps and not nano_all else
                         "E_IMAGE_FOCUS_CHANGES" if resp else
                         "B_C_or_D_IMAGE_STATIC (Nano replied; shaft / gear / ring not observable in the IMX477 view)" if ps else
                         "NOT_RUN"),
               state="DONE")
    Path(lim["marker"]).write_text(json.dumps(out, ensure_ascii=False, default=str))
    return out


def install_diag_route(AC, gate=None):
    """GET /live2/focus_diag?req=&owner= (127.0.0.1 only) -> submitted to the student DeviceOwner as a 'diag' job."""
    H = AC.Handler
    if getattr(H.do_GET, "_live2_diag", False):
        return
    inner = H.do_GET

    def do_GET(self):
        import urllib.parse
        url = urllib.parse.urlparse(self.path)
        if url.path != "/live2/focus_diag":
            return inner(self)
        sys.path.insert(0, str(HERE.parent / "live1"))
        import live1_common as C
        ap = gate() if gate else C.approval_status("B", "pantilt")
        lim, why = diag_limits(ap)
        if lim is None:
            _audit("REFUSED_FOCUS_DIAG", why=why)
            return self.reply(json.dumps(dict(status="REFUSED", reason_code=why)).encode(), "application/json", 403)
        q = urllib.parse.parse_qs(url.query)
        rid, own = (q.get("req") or [None])[0], (q.get("owner") or [None])[0]
        if not rid or not own:
            rid, own = AC.OWNER["o"].manual_id("live2_focus_diag")
        _audit("SENT_FOCUS_DIAG", limits={k: lim[k] for k in DIAG_HARD}, approval_file=lim["approval_file"])
        a = AC.AIMER["a"]
        res = AC.OWNER["o"].submit("diag", rid, own, dict(name="live2_focus_diag"),
                                   fn=lambda tok: run_focus_diag(a, lim, tok, log=AC.log))
        return self.reply(json.dumps(res, ensure_ascii=False, default=str).encode(), "application/json")
    do_GET._live2_diag = True
    do_GET._live2 = True
    H.do_GET = do_GET


# ------------------------------------------------------------------ MANUAL FOCUS (2026-10-26, E8 path without the DC motor)
# 127.0.0.1 only (LocalServer). Read-only: /live2/preview.jpg (newest IMX477 frame, resized) and /live2/metrics (student
# sharpness on ROI / centre / full). Owner jobs (serialised with aim / HOME by the student DeviceOwner, no motion at all):
# /live2/manual_capture (one fresh full-resolution frame -> capture dir + sha256) and /live2/pose (Nano '?': commanded target).
def manual_capture_job(a, owner, rid, roi, timeout_s=2.0):
    import cv2
    import e6_manual as M
    seq0 = a.stats.get()[0]
    fr = a.stats.fresh_frame(seq0, timeout_s)
    if fr is None:                     # a refusal is a RESULT (the student owner marks UNKNOWN only on real errors)
        return dict(capture=None, why="NO_FRESH_FRAME (IMX477 stalled)")
    seq, frame, rx = fr
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return dict(capture=None, why="JPEG_ENCODE_FAILED")
    data = buf.tobytes()
    cid = f"cap-{rid}-s{seq}".replace(":", "_").replace("/", "_")
    path = os.path.join(owner.capture_dir, f"{cid}.jpg")
    if os.path.exists(path):
        return dict(capture=None, why="CAPTURE_ID_COLLISION")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    m = M.measure(frame, roi, auto=True)
    cap = dict(capture_id=cid, request_id=rid, path=path, frame_seq=seq, frame_rx_mono_ns=rx, saved_wall=time.time(),
               sha256=M.sha256_bytes(data), width=frame.shape[1], height=frame.shape[0], jpeg_quality=95, metrics=m,
               kind="LIVE2_MANUAL_FOCUS_CAPTURE")
    with open(os.path.join(owner.capture_dir, f"{cid}.json"), "w", encoding="utf-8") as f:
        json.dump(cap, f, ensure_ascii=False, indent=1)
    return dict(capture=cap)


def shake(stats, gap=0.15, size=(320, 180)):
    """2026-10-28 field (operator: the head keeps shaking 2-7 s after the stop): global image shift between two FRESH IMX477
    frames `gap` s apart, phase correlation on a Hanning-windowed gray 320x180 (whole-frame camera motion; a moving person or
    glare biases it — recorded with the correlation response). -> dict(shift_px (full-res px), response, dt_s) | dict(stale)."""
    import cv2
    import numpy as np
    f1 = stats.fresh_frame(stats.get()[0], 1.0)
    if f1 is None:
        return dict(stale=True, why="no fresh IMX477 frame")
    time.sleep(gap)
    f2 = stats.fresh_frame(f1[0], 1.0)
    if f2 is None:
        return dict(stale=True, why="no second fresh frame")
    g = [np.float32(cv2.cvtColor(cv2.resize(f[1], size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)) for f in (f1, f2)]
    win = cv2.createHanningWindow(size, cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(g[0], g[1], win)
    k = f1[1].shape[1] / size[0]
    return dict(stale=False, shift_px=round(float(np.hypot(dx, dy)) * k, 1), dx_px=round(dx * k, 1), dy_px=round(dy * k, 1),
                response=round(float(resp), 3), dt_s=round((f2[2] - f1[2]) / 1e9, 3), seq=[f1[0], f2[0]])


def install_manual_routes(AC):
    H = AC.Handler
    if getattr(H.do_GET, "_live2_manual", False):
        return
    inner = H.do_GET

    def do_GET(self):
        import urllib.parse
        import cv2
        import e6_manual as M
        url = urllib.parse.urlparse(self.path)
        if not url.path.startswith("/live2/") or url.path == "/live2/focus_diag":
            return inner(self)
        q = urllib.parse.parse_qs(url.query)
        js = lambda o, code=200: self.reply(json.dumps(o, ensure_ascii=False, default=str).encode(), "application/json", code)  # noqa: E731
        if url.path == "/live2/camera":
            return js(camera_route(q) if getattr(AC.cv2, "__class__", None).__name__ == "Cv2Proxy"
                      else dict(on_demand=False, open=True, why="IMX477 always on (not in on-demand mode)"))
        a = AC.AIMER["a"]
        if a is None:
            return js(dict(status="NOT_READY"), 503)
        if url.path in ("/live2/preview.jpg", "/live2/metrics"):
            with a.stats.lock:
                seq, frame, rx = a.stats.seq, a.stats.frame, a.stats.rx_mono_ns
            if frame is None:
                return js(dict(status="NO_FRAME"), 503)
            age = (time.monotonic_ns() - rx) / 1e9 if rx else None
            if url.path == "/live2/preview.jpg":
                w = max(320, min(1920, int((q.get("w") or ["960"])[0])))
                img = frame if w >= frame.shape[1] else cv2.resize(frame, (w, int(w * frame.shape[0] / frame.shape[1])),
                                                                    interpolation=cv2.INTER_AREA)
                body = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Seq", str(seq))
                self.send_header("X-Frame-Age-S", f"{age:.3f}" if age is not None else "")
                self.send_header("X-Frame-Size", f"{frame.shape[1]}x{frame.shape[0]}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return None
            roi = M.parse_roi((q.get("roi") or [""])[0])
            m = M.measure(frame, roi, auto=(q.get("auto") or ["1"])[0] == "1")
            return js(dict(m, seq=seq, frame_age_s=None if age is None else round(age, 3),
                           resolution=[frame.shape[1], frame.shape[0]], wall=time.time()))
        if url.path == "/live2/shake":                  # camera shake: image shift between two fresh frames (no motion)
            return js(shake(a.stats, float((q.get("gap") or ["0.15"])[0])))
        o = AC.OWNER["o"]
        rid, own = (q.get("req") or [None])[0], (q.get("owner") or [None])[0]
        if not rid or not own:
            return js(dict(protocol="r1-aim-1", status="PROTOCOL_ERROR", reason_code="request_id_and_owner_required"), 400)
        if url.path == "/live2/manual_capture":
            if os.environ.get("LIVE2_FOCUS_MODE") != "MANUAL_FIXED_NO_MOTOR":
                return js(dict(status="REFUSED", reason_code="manual capture route only in MANUAL_FIXED_NO_MOTOR mode"), 403)
            roi = M.parse_roi((q.get("roi") or [""])[0])
            _audit("SENT_MANUAL_CAPTURE", request_id=rid, note="no motion: one IMX477 frame")
            return js(o.submit("diag", rid, own, dict(name="live2_manual_capture"),
                               fn=lambda tok: manual_capture_job(a, o, rid, roi)))
        if url.path == "/live2/pose":
            def pose(tok):
                st = a.nano.cmd("?") or {}
                return dict(pan=st.get("pan"), tilt=st.get("tilt"), angle_kind="COMMANDED target reported by the Nano (no sensor)")
            return js(o.submit("diag", rid, own, dict(name="live2_pose"), fn=pose))
        return js(dict(status="NOT_FOUND"), 404)
    do_GET._live2_manual = True
    do_GET._live2_diag = getattr(inner, "_live2_diag", False)
    do_GET._live2 = True
    H.do_GET = do_GET


# ------------------------------------------------------------------ IMX477 ON DEMAND (2026-10-27 field: camera only per event)
# The student main loop opens the IMX477 once and reads it for the whole run (1080p MJPG, USB2). In MANUAL_FIXED_NO_MOTOR mode
# the capture object of THAT loop is wrapped (student file unchanged): the device is opened only while ARMED — the bridge arms
# it when a SigLIP STOP request appears (before the robot stops; pan-tilt still does not move) and keeps it armed through
# aim / manual focus / capture; it is released HOLD_S after the last arm while the device owner is idle. Opening goes through
# the stable by-id link (the /dev/videoN number changes after a USB re-plug). Measured 2026-10-27 on this camera: open -> first
# 1920x1080 frame 1.0-1.1 s (5/5), ~31 fps; field detection -> stop took 3-12 s.
CAM_BYID = "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_IMX477_HQ_Camera_UC517-video-index0"
CAM = dict(armed_until=0.0, opens=0, releases=0, last_open_s=None, last_open_wall=None, last_error=None, open=False,
           device=None, why=None)


class OnDemandCapture:
    def __init__(self, cv2, src, api, busy=lambda: False, clock=time.time, byid=CAM_BYID):
        self.cv2, self.src, self.api, self.busy, self.clock, self.byid = cv2, src, api, busy, clock, byid
        self.props, self.cap = [], None
        self.probe_ok = self._open() is not None          # same start check as the student (exit if the camera is missing)
        self._close("start probe")

    def _dev(self):
        return os.path.realpath(self.byid) if os.path.exists(self.byid) else self.src

    def _open(self):
        t0 = self.clock()
        cap = self.cv2.VideoCapture(self._dev(), self.api)
        if not cap.isOpened():
            CAM.update(last_error=f"cannot open {self._dev()}")
            return None
        for p, v in self.props:
            cap.set(p, v)
        self.cap = cap
        CAM.update(opens=CAM["opens"] + 1, open=True, device=self._dev(), last_open_wall=self.clock(), t_open=t0, first_frame=False)
        return cap

    def _close(self, why):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            CAM.update(releases=CAM["releases"] + 1, open=False, why=why)

    def _ensure(self):
        now = self.clock()
        if now < CAM["armed_until"] or self.busy():
            if self.cap is None:
                self._open()
        elif self.cap is not None:
            self._close("idle: not armed")
        return self.cap

    def isOpened(self):
        return self.probe_ok

    def set(self, prop, val):
        self.props.append((prop, val))
        return True if self.cap is None else self.cap.set(prop, val)

    def _mark(self, ok):
        if ok and not CAM.get("first_frame"):
            CAM.update(first_frame=True, last_open_s=round(self.clock() - CAM.get("t_open", self.clock()), 2))
        return ok

    def read(self):
        if self._ensure() is None:
            time.sleep(0.05)
            return False, None
        ok, f = self.cap.read()
        self._mark(ok)
        return ok, f

    def grab(self):
        if self._ensure() is None:
            time.sleep(0.05)
            return False
        return self._mark(self.cap.grab())

    def retrieve(self):
        return (False, None) if self.cap is None else self.cap.retrieve()

    def release(self):
        self._close("console exit")


def install_no_realsense_images():
    """R34: the student console's RealSense colour / depth subscriptions are not created under LIVE2 (camera_info stays: /aim needs K).
    11-01 field + read-only reproduction: any second process reading these image topics eventually stops aligned depth for EVERY reader.
    LIVE2 gives /aim its target pixel and distance; the operator views come from the sensor hub. Student file unchanged."""
    from rclpy.node import Node
    orig = Node.create_subscription

    def create_subscription(self, msg_type, topic, *a, **k):
        if getattr(msg_type, "__name__", "") == "Image" and str(topic).startswith("/camera/"):
            print(f"LIVE2_E6 no subscription to {topic}: the sensor hub is the only image reader", flush=True)
            return None
        return orig(self, msg_type, topic, *a, **k)
    Node.create_subscription = create_subscription


def install_camera_byid(AC, byid=CAM_BYID):
    """2026-10-28 05:35 field: the student find_camera('Arducam') (sysfs name) found nothing once and the console exited; the
    stable by-id link is used when it exists, the student lookup otherwise (student file unchanged)."""
    orig = AC.find_camera

    def find_camera(name=None, *a, **k):
        if os.path.exists(byid):
            return os.path.realpath(byid)
        return orig(name, *a, **k) if name is not None else orig(*a, **k)
    AC.find_camera = find_camera


def install_camera_on_demand(AC):
    """Replace the cv2 name INSIDE the student module only by a proxy whose VideoCapture is OnDemandCapture."""
    import cv2

    class Cv2Proxy:
        def __getattr__(self, n):
            return getattr(cv2, n)

        @staticmethod
        def VideoCapture(src, api=None):
            busy = lambda: (AC.OWNER["o"] is not None and AC.OWNER["o"].state().get("device_state") != "IDLE")  # noqa: E731
            return OnDemandCapture(cv2, src, api if api is not None else cv2.CAP_ANY, busy=busy)
    AC.cv2 = Cv2Proxy()


def camera_route(q):
    """/live2/camera?arm=1&hold=<s> -> arm for hold s (max 600); without arm: status only."""
    if (q.get("arm") or ["0"])[0] == "1":
        hold = max(5.0, min(600.0, float((q.get("hold") or ["120"])[0])))
        until = time.time() + hold
        CAM["armed_until"] = until if (q.get("set") or ["0"])[0] == "1" else max(CAM["armed_until"], until)
    return dict(CAM, armed=time.time() < CAM["armed_until"], armed_left_s=round(max(0.0, CAM["armed_until"] - time.time()), 1),
                on_demand=True)


def _audit(event, **kw):
    try:
        import e6_precision_bridge as B
        B.audit(event, tool="e6_aim_console_live2", **kw)
    except Exception:                                                        # noqa: BLE001
        pass


if __name__ == "__main__":
    install_guards(AC, audit=_audit)
    install_diag_route(AC)
    install_manual_routes(AC)
    install_camera_byid(AC)
    install_no_realsense_images()
    if os.environ.get("LIVE2_FOCUS_MODE") == "MANUAL_FIXED_NO_MOTOR" and os.environ.get("LIVE2_IMX477_ON_DEMAND", "1") == "1":
        install_camera_on_demand(AC)
    strategy = os.environ.get("LIVE2_AF_STRATEGY", "student_stepwise")
    if strategy not in STRATEGIES:
        raise SystemExit(f"LIVE2_AF_STRATEGY={strategy} not in {STRATEGIES}")
    if strategy == "focus_v2":
        F2.install(AC)
    else:
        import e6_af_student as AFS
        AFS.install(AC, F2)
    raise SystemExit(AC.main())
