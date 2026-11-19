"""e6_settle.py — LIVE2 E6 run mode MANUAL_SETTLE (2026-10-30 directive): after the E5 STOP request and the confirmed stop, the
pan-tilt waits until the OPERATOR, who watches the body shake die down, presses [진동 안정 확인 · 팬틸트 실행] on the 8770 page.
AUTO (default) keeps the existing path unchanged. This module never commands the robot or the pan-tilt; it only decides whether
the existing E6 pipeline may start for ONE observation.

Flow (states written to <state>/e6_settle.json, every change appended to <state>/e6_settle_events.jsonl):
  WAITING_FOR_SETTLE_APPROVAL -> E6_APPROVED (pipeline continues: aim / focus / capture / OCR / HOME as before)
                              -> USER_CANCELLED | ROBOT_MOVED | SETTLE_APPROVAL_EXPIRED | EXPERIMENT_ENDED   (no E6 for it)
Operator commands: <state>/control/e6_settle_cmd.jsonl, appended by the 8770 console (auth + CSRF + request_id there). Only lines
written after this bridge started are read; a command for another observation / experiment is answered STALE_EVENT.
[approve] is accepted only when ALL hold at that moment (otherwise refused with the reason; the wait continues):
  1 the command names the CURRENT waiting observation of THIS experiment     5 software stability (stability() below)
  2 the stop request of this observation exists and the stop was confirmed   6 no E6 already running for it (synchronous gate)
  3 the robot is still STOPPED on fresh FAST-LIO odometry (e6_core rule)      7 not approved before (duplicate -> refused)
  4 odometry and, when available, the body-IMU health are fresh               8 hardware approval B valid now ('pantilt')
The button never replaces approval B, and a wait timeout NEVER starts E6 (it invalidates the observation instead).

Software stability (PROVISIONAL, from real records — reports/body_imu_field_loc12_analysis_20261030_1135/data/settle_analysis.json):
  2026-10-30 dynfield /lowstate walk, 5 walk->stop transitions: walking gyro RMS 29 deg/s (p5 15); within 0.5-2 s after the stop
  the 1 s gyro RMS falls to a plateau of 2.7-3.3 deg/s (peak 4-12 deg/s, accel-norm std 0.05-0.19 m/s^2); standing still
  1.74 deg/s. Lines chosen between the settled plateau and walking, so a normal settled stop passes and walking / shaking does not:
  gyro RMS <= 5 deg/s, gyro peak <= 15 deg/s, accel-norm std <= 0.30 m/s^2 (1 s window from the body-IMU bridge health).
  Camera shake (aim console /live2/shake, IMX477 whole-frame shift) <= 8 px when measurable (existing e6_pipeline SETTLE line).
  In L1 input mode no body-IMU bridge runs: the IMU line is reported NOT_AVAILABLE and the odometry + camera checks remain.
"""
import json
import math
import time
from pathlib import Path

MODES = ("AUTO", "MANUAL_SETTLE")
GYRO_RMS_MAX_DPS, GYRO_PEAK_MAX_DPS, ACC_STD_MAX = 5.0, 15.0, 0.30
IMU_FRESH_S = 2.5
SHAKE_OK_PX = 8.0
WAIT_MAX_S = 300.0          # no approval within this -> observation invalid (never auto-run)
POLL_S = 0.25
ACTIONS = ("approve", "cancel")
MSG_UNSTABLE = "로봇 움직임 또는 진동이 감지됩니다. 안정된 뒤 다시 승인하세요."
MSG_WAIT = "로봇 정지 완료 — 진동이 안정되면 팬틸트 실행을 승인하세요."
MSG_MOVED = "로봇이 움직였습니다. 완전히 멈추면 현재 자세로 표지판 위치를 다시 계산합니다."
END_STATES = ("E6_APPROVED", "USER_CANCELLED", "ROBOT_MOVED", "SETTLE_APPROVAL_EXPIRED", "EXPERIMENT_ENDED")


def _now_pair(clock=time):
    return dict(wall=round(clock.time(), 3), mono=round(time.monotonic(), 3))


def body_run_dir(runtime):
    """runtime/owned/fastlio.json -> body_run_dir (body input mode) or None (L1 / not running)."""
    try:
        rec = json.loads((Path(runtime) / "owned" / "fastlio.json").read_text())
    except (OSError, ValueError):
        return None
    try:                                                 # a record of a stopped / earlier-boot FAST-LIO is not the live input
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = None
    if rec.get("boot_id") != boot or not Path(f"/proc/{rec.get('pid')}").exists():
        return None
    ex = rec.get("extra") if isinstance(rec.get("extra"), dict) else rec
    if (ex.get("input") or rec.get("input")) != "body":
        return None
    d = ex.get("body_run_dir") or rec.get("body_run_dir")
    return Path(d) if d else None


def _last_line(p, max_bytes=4096):
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - max_bytes))
            lines = f.read().decode(errors="replace").strip().splitlines()
        return json.loads(lines[-1]) if lines else None
    except (OSError, ValueError, IndexError):
        return None


def imu_vibration(runtime, now_wall=None):
    """Newest 1 s window of the body-IMU bridge health. -> dict(available, ...). Never raises."""
    d = body_run_dir(runtime)
    if d is None:
        return dict(available=False, why="NOT_AVAILABLE (L1 입력 모드 또는 FAST-LIO 정지: 본체 IMU 브리지 없음)")
    h = _last_line(d / "bridge_health.jsonl")
    if not h:
        return dict(available=False, why="NO_BRIDGE_HEALTH")
    now_wall = time.time() if now_wall is None else now_wall
    age = now_wall - h.get("wall_ns", 0) / 1e9
    out = dict(available=True, health=h.get("health"), age_s=round(age, 2), gyro_rms_dps=h.get("gyro_rms_dps"),
               gyro_peak_dps=h.get("gyro_peak_dps"), acc_std=h.get("acc_norm_std"), publish_hz=h.get("publish_hz"))
    if out["gyro_rms_dps"] is None:
        out.update(available=False, why="BRIDGE_WITHOUT_VIBRATION_FIELDS (old bridge build)")
    return out


def stability(stop, imu, shake):
    """-> (ok, reasons[], detail). stop = e6_core.stop_state dict; imu = imu_vibration(); shake = /live2/shake reply or None."""
    bad = []
    if not (stop or {}).get("stopped"):
        bad.append(f"정지 아님 ({(stop or {}).get('why')})")
    if imu and imu.get("available"):
        if imu["age_s"] > IMU_FRESH_S or imu.get("health") != "READY":
            bad.append(f"본체 IMU 갱신 안 됨 ({imu.get('health')}, {imu['age_s']} s 전)")
        else:
            if imu["gyro_rms_dps"] > GYRO_RMS_MAX_DPS:
                bad.append(f"자이로 RMS {imu['gyro_rms_dps']:.1f} > {GYRO_RMS_MAX_DPS} deg/s")
            if imu["gyro_peak_dps"] > GYRO_PEAK_MAX_DPS:
                bad.append(f"자이로 최대 {imu['gyro_peak_dps']:.1f} > {GYRO_PEAK_MAX_DPS} deg/s")
            if imu["acc_std"] > ACC_STD_MAX:
                bad.append(f"가속도 변동 {imu['acc_std']:.2f} > {ACC_STD_MAX} m/s²")
    px = None if not shake or shake.get("stale") or shake.get("_error") else shake.get("shift_px")
    if px is not None and px > SHAKE_OK_PX:
        bad.append(f"카메라 영상 흔들림 {px:.1f} > {SHAKE_OK_PX} px")
    detail = dict(stop={k: (stop or {}).get(k) for k in ("stopped", "speed_mps", "yaw_rate_dps", "why", "odom_age_s")},
                  imu=imu, camera_shake_px=px, camera_shake="MEASURED" if px is not None else "NOT_MEASURABLE",
                  lines=dict(gyro_rms_dps=GYRO_RMS_MAX_DPS, gyro_peak_dps=GYRO_PEAK_MAX_DPS, acc_std=ACC_STD_MAX, shake_px=SHAKE_OK_PX))
    return not bad, bad, detail


class SettleGate:
    def __init__(self, state_dir, runtime, clock=time, sleep=time.sleep, wait_max_s=WAIT_MAX_S):
        self.state, self.runtime, self.clock, self.sleep, self.wait_max_s = Path(state_dir), Path(runtime), clock, sleep, wait_max_s
        self.cmd_file = self.state / "control" / "e6_settle_cmd.jsonl"
        try:
            self.off = self.cmd_file.stat().st_size          # commands from before this bridge are never executed
        except OSError:
            self.off = 0
        self.d = {}
        self.approved = set()

    # ---- records
    def _write(self):
        self.d["wall"] = self.clock.time()
        self.state.mkdir(parents=True, exist_ok=True)
        tmp = self.state / f".e6_settle.json.{id(self)}.tmp"
        tmp.write_text(json.dumps(self.d, ensure_ascii=False, default=str))
        tmp.replace(self.state / "e6_settle.json")

    def event(self, kind, **kw):
        r = dict(kind=kind, observation_id=self.d.get("observation_id"), experiment_id=self.d.get("experiment_id"), **_now_pair(self.clock), **kw)
        with open(self.state / "e6_settle_events.jsonl", "a") as f:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        return r

    def commands(self):
        out = []
        try:
            with open(self.cmd_file) as f:
                f.seek(self.off)
                for line in f:
                    if not line.endswith("\n"):
                        break
                    self.off += len(line.encode())
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            pass
        return out

    def _reply(self, cmd, accepted, why):
        r = dict(request_id=cmd.get("request_id"), action=cmd.get("action"), accepted=accepted, why=why, **_now_pair(self.clock))
        self.d.setdefault("replies", []).append(r)
        self.d["replies"] = self.d["replies"][-10:]
        self.event("COMMAND", **{k: v for k, v in r.items() if k not in ("wall", "mono")}, actor=cmd.get("actor"))
        return r

    def _state(self, st, why=None, **kw):
        self.d.update(state=st, why=why, state_since=self.clock.time(), **kw)
        self._write()
        self.event("STATE", state=st, why=why)

    # ---- the gate
    def wait(self, obs, experiment_id, stop_check, shake_fn, approval_check, moved_check, ended=lambda: False, heartbeat=None, rebase=None):
        """Blocks until approved / cancelled / invalid. -> (outcome, why, record). outcome == 'E6_APPROVED' lets E6 continue."""
        oid = obs["observation_id"]
        t0 = self.clock.time()
        self.d = dict(kind="LIVE2_E6_SETTLE", mode="MANUAL_SETTLE", observation_id=oid, experiment_id=experiment_id,
                      source=obs.get("source"), message=MSG_WAIT,
                      times=dict(t_e5_trigger=obs.get("detection_timestamp"), t_stop_request=obs.get("created_timestamp"),
                                 t_stop_confirmed=obs.get("stopped_wall"), t_wait_start=round(t0, 3), t_wait_start_mono=round(time.monotonic(), 3),
                                 t_settle_ready=None, t_manual_approval=None, t_end=None),
                      e4_at_request=obs.get("e4_at_request"), odom_pose_at_stop=obs.get("odom_pose_at_stop"), replies=[])
        self._state("WAITING_FOR_SETTLE_APPROVAL", why=MSG_WAIT)
        while True:
            now = self.clock.time()
            stop = stop_check()
            imu = imu_vibration(self.runtime, now)
            try:
                shake = shake_fn()
            except Exception as e:                                       # noqa: BLE001
                shake = dict(stale=True, why=f"{type(e).__name__}: {e}")
            ok, bad, det = stability(stop, imu, shake)
            apb = approval_check()
            self.d.update(live=dict(det, stable=ok, unstable_reasons=bad, checked_wall=round(now, 3),
                                    elapsed_since_stop_request_s=None if not obs.get("created_timestamp") else round(now - obs["created_timestamp"], 1)),
                          approval_b=dict(valid=bool(apb.get("valid")), why=apb.get("why")))
            if ok and self.d["times"]["t_settle_ready"] is None:
                self.d["times"]["t_settle_ready"] = round(now, 3)
                self.event("SETTLE_READY_SOFTWARE", detail=det)
            if not ok and self.d["times"]["t_settle_ready"] is not None and not (stop or {}).get("stopped"):
                self.d["times"]["t_settle_ready"] = None                 # motion again: the software-ready time restarts
            moved, why_m = moved_check(obs)
            if moved:
                # R40 (X20261103_000434 O005: 0.04 m / 6.2 deg; X20261103_001810 O003: 0.30 m): nothing has moved the head yet, so a small move only changes where the plate is. The target is fixed in odometry: new stop, new prediction.
                nb = None
                if rebase is not None:
                    if not (stop or {}).get("stopped"):
                        self.d.update(message=MSG_MOVED)
                        self._state("WAITING_FOR_SETTLE_APPROVAL", why=MSG_MOVED)
                        self.d["times"]["t_settle_ready"] = None
                        if ended():
                            return self._finish("EXPERIMENT_ENDED", "실험 종료 — 팬틸트 실행 안 함")
                        if now - t0 > self.wait_max_s:
                            return self._finish("SETTLE_APPROVAL_EXPIRED", f"{self.wait_max_s:.0f} s 동안 승인 없음 — 이번 관측 무효 (자동 실행 안 함)")
                        self._write()
                        if heartbeat is not None:
                            heartbeat()
                        self.sleep(POLL_S)
                        continue
                    nb = rebase(obs)
                if not nb:
                    return self._finish("ROBOT_MOVED", f"정지 위치에서 이동함 ({why_m}) — 이번 관측 무효, 팬틸트 실행 안 함")
                obs.update(nb)
                self.d.update(message=MSG_WAIT, odom_pose_at_stop=obs.get("odom_pose_at_stop"))
                self.event("REBASED_AFTER_MOVE", why=why_m)
                self._state("WAITING_FOR_SETTLE_APPROVAL", why=MSG_WAIT)
                continue
            if ended():
                return self._finish("EXPERIMENT_ENDED", "실험 종료 — 팬틸트 실행 안 함")
            if now - t0 > self.wait_max_s:
                return self._finish("SETTLE_APPROVAL_EXPIRED", f"{self.wait_max_s:.0f} s 동안 승인 없음 — 이번 관측 무효 (자동 실행 안 함)")
            batch = self.commands()
            for i, c in enumerate(batch):
                act = c.get("action")
                if c.get("observation_id") != oid or c.get("experiment_id") not in (None, experiment_id):
                    self._reply(c, False, f"STALE_EVENT (현재 관측 {oid})")
                    continue
                if act not in ACTIONS:
                    self._reply(c, False, "BAD_ACTION")
                    continue
                if act == "cancel":
                    self._reply(c, True, "이번 의미정보 획득 취소")
                    self._answer_rest(batch[i + 1:])
                    return self._finish("USER_CANCELLED", "운영자 취소 — 팬틸트 실행 안 함")
                if oid in self.approved:
                    self._reply(c, False, "DUPLICATE (이미 승인됨)")
                    continue
                if not apb.get("valid"):
                    self._reply(c, False, f"APPROVAL_B_REQUIRED (하드웨어 승인 B {apb.get('why')}) — 팬틸트 구동 불가")
                    continue
                if not obs.get("created_timestamp") or not obs.get("stopped_wall"):
                    self._reply(c, False, "NO_STOP_REQUEST_OR_STOP (정지 요청·정지 확인 기록 없음)")
                    continue
                if not ok:
                    self._reply(c, False, MSG_UNSTABLE + " [" + "; ".join(bad) + "]")
                    continue
                self.approved.add(oid)
                self.d["times"]["t_manual_approval"] = round(self.clock.time(), 3)
                self.d["times"]["t_manual_approval_mono"] = round(time.monotonic(), 3)
                self.d["approval_check"] = dict(det, approval_b=self.d["approval_b"], request_id=c.get("request_id"), actor=c.get("actor"))
                self._reply(c, True, "승인 — 팬틸트 실행")
                self._answer_rest(batch[i + 1:])
                return self._finish("E6_APPROVED", "운영자 승인 + 정지·안정·승인 B 확인 — 기존 E6 경로 실행")
            self._write()
            if heartbeat is not None:
                heartbeat()
            self.sleep(POLL_S)

    def _answer_rest(self, rest):
        """commands that arrived in the same poll after the deciding one: answered, never executed (double tap)."""
        for c in rest:
            self._reply(c, False, "DUPLICATE (이 관측은 이미 결정됨)")

    def _finish(self, outcome, why):
        self.d["times"]["t_end"] = round(self.clock.time(), 3)
        self._state(outcome, why=why)
        return outcome, why, dict(self.d)

    def mark(self, **times):
        """pipeline milestones after the approval (t_e6_start, t_image_capture, t_semantic_complete) — record only."""
        self.d.setdefault("times", {}).update({k: round(v, 3) for k, v in times.items() if v is not None})
        self._write()
        self.event("MILESTONE", **{k: round(v, 3) for k, v in times.items() if v is not None})


def append_command(state_dir, cmd):
    """8770 console side: one operator command line (validated there)."""
    p = Path(state_dir) / "control" / "e6_settle_cmd.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(cmd, ensure_ascii=False) + "\n")
    return p


def read_snapshot(state_dir):
    try:
        return json.loads((Path(state_dir) / "e6_settle.json").read_text())
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ time-to-fix metrics (post-run, record only)
def _jl(p):
    try:
        return [json.loads(x) for x in open(p) if x.strip()]
    except OSError:
        return []


def _union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def time_metrics(exp_dir):
    """Archived experiment dir -> dict. Raw timestamps are copied, never modified. All times = host wall clock (one clock for every
    process of the run); monotonic values are kept beside them where the gate recorded them.
      TTF_raw       = t_fix - t_run_start                                  (only when an E7 GLOBAL_CONFIRMED exists)
      T_manual_wait = t_manual_approval - t_stop_request  per APPROVED observation (includes the robot's own stopping time and the
                      body settling the operator waited for)
      TTF_adjusted  = TTF_raw - |union of the manual-wait intervals that end before t_fix|   (a procedure-corrected index, NOT the
                      time-to-fix of a fully automatic system)
      cancelled / expired / moved waits are listed but never subtracted (no approval time exists)."""
    exp_dir = Path(exp_dir)
    ex = json.loads((exp_dir / "experiment.json").read_text())
    st = exp_dir / "state" if (exp_dir / "state").exists() else exp_dir
    t_run = ex.get("created_wall")
    ev = _jl(st / "e6_settle_events.jsonl")
    obs = {}
    for o in _jl(st / "e6_observations.jsonl"):
        obs[o["observation_id"]] = o
    prec = {}
    for r in _jl(st / "e6_precision_results.jsonl"):
        prec[r.get("observation_id")] = r
    by = {}
    for e in ev:
        by.setdefault(e.get("observation_id"), []).append(e)
    rows, waits = [], []
    for oid, es in by.items():
        o = obs.get(oid) or {}
        endst = [e for e in es if e["kind"] == "STATE" and e.get("state") in END_STATES]
        out = endst[-1]["state"] if endst else "NOT_OBSERVED"
        appr = next((e for e in es if e["kind"] == "COMMAND" and e.get("action") == "approve" and e.get("accepted")), None)
        ready = next((e for e in es if e["kind"] == "SETTLE_READY_SOFTWARE"), None)
        ms = {}
        for e in es:
            if e["kind"] == "MILESTONE":
                ms.update({k: v for k, v in e.items() if k.startswith("t_")})
        p = prec.get(oid) or {}
        t_req = o.get("created_timestamp")
        row = dict(observation_id=oid, outcome=out, t_e5_trigger=o.get("detection_timestamp"), t_stop_request=t_req,
                   t_stop_confirmed=o.get("stopped_wall"), t_settle_ready=None if not ready else ready["wall"],
                   t_manual_approval=None if not appr else appr["wall"], t_manual_approval_mono=None if not appr else appr["mono"],
                   t_e6_start=ms.get("t_e6_start"), t_image_capture=ms.get("t_image_capture") or p.get("capture_wall"),
                   t_semantic_complete=ms.get("t_semantic_complete"), e6_status=p.get("status"), e6_timings_ms=p.get("e6_timings_ms"))
        if appr and t_req:
            row["T_manual_wait_s"] = round(appr["wall"] - t_req, 3)
            row["T_stop_part_s"] = None if not o.get("stopped_wall") else round(o["stopped_wall"] - t_req, 3)
            row["T_settle_part_s"] = None if not o.get("stopped_wall") else round(appr["wall"] - o["stopped_wall"], 3)
            waits.append((t_req, appr["wall"]))
        elif t_req:
            t_end = next((e["wall"] for e in endst), None)
            row["T_wait_not_subtracted_s"] = None if t_end is None else round(t_end - t_req, 3)
        if row["t_e6_start"] and row["t_semantic_complete"]:
            row["T_E6_s"] = round(row["t_semantic_complete"] - row["t_e6_start"], 3)
        rows.append(row)
    e7 = None
    try:
        e7 = json.loads((st / "e7_state.json").read_text())
    except (OSError, ValueError):
        pass
    fix = e7 if e7 and e7.get("global_decision") == "GLOBAL_CONFIRMED" else None
    res = dict(kind="LIVE2_TIME_METRICS", exp_id=ex.get("exp_id"), clock="host wall clock (all processes); mono kept where recorded",
               t_run_start=t_run, e6_mode=ex.get("e6_mode", "AUTO"), observations=rows,
               manual_wait_intervals=[[round(a, 3), round(b, 3)] for a, b in _union(waits)])
    # 2026-11-01 (r33): operator-normalized clock (settle -> 1 s, focus -> 3 s); an analysis index kept BESIDE the raw values
    try:
        import op_clock as OC
        t_fix_ = fix.get("written_wall") if fix and (fix.get("written_wall") or 0) >= (t_run or 0) else None
        oc = OC.summary(st, t_run, ex.get("ended_wall") or time.time(), t_fix_, ex.get("exp_id"))
        res.update(operator_clock=oc, T_end_raw_s=(oc.get("at_end") or {}).get("raw_s"), T_end_normalized_s=(oc.get("at_end") or {}).get("normalized_s"),
                   TTF_normalized_s=oc.get("TTF_normalized_s"), normalized_note=OC.NOTE)
    except Exception as e:                                       # noqa: BLE001  never breaks the raw record
        res.update(operator_clock=dict(error=f"{type(e).__name__}: {e}"))
    if not fix:
        res.update(t_fix=None, TTF_raw_s=None, TTF_adjusted_s=None, TTF_normalized_s=None,
                   why="NO_FIX (no E7 GLOBAL_CONFIRMED) — no time-to-fix value")
        return res
    t_fix = fix.get("written_wall")
    raw = t_fix - t_run
    sub = sum(max(0.0, min(b, t_fix) - a) for a, b in _union(waits) if a < t_fix)
    res.update(t_fix=t_fix, fix_path=fix.get("path"), TTF_raw_s=round(raw, 3), sum_manual_wait_s=round(sub, 3),
               TTF_adjusted_s=round(raw - sub, 3),
               note="TTF_adjusted removes only the identified operator waits of the manual procedure; it is not an automatic-mode time")
    return res
