#!/usr/bin/env python3
"""LIVE1 web panel: READ-ONLY view of what the LIVE1 components actually wrote. GET only, no command endpoint.

Data sources (files only; the web never subscribes ROS, never loads a model, never opens a device):
  --state   : lio_state.json / lio_map.png (lio_monitor), supervisor state files, baseline json
  --e5-shm  : status.json / frame.jpg / frame_inference.jpg (e5_onboard SHADOW via observe_node)
Every value is shown with its source age; missing = DATA_UNAVAILABLE, old = STALE (never green).
--test-only marks every response TEST_ONLY (fixture data, e.g. a recorded SEM1 status).
"""
import argparse
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sci_sensor_bridge"))
import depth_freshness as DF  # noqa: E402
import live1_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
LIO_STALE_S, E5_STALE_S, ODOM_STALE_S, INFER_OLD_S, STOP_UI_S, RGB_STALE_S, SIGLIP_STALE_S = 2.0, 3.0, 0.5, 1.0, 6.0, 1.0, 5.0
STOP_UI_VISUAL_S = 20.0          # operator needs time to find a safe place; the request stays visible this long
STOP_UI_WAIT_MAX_S, STOP_UI_DONE_S = 120.0, 30.0   # 2026-10-21 EXP1: a request was not noticed. While its E6 record still waits for the
#   stop (WAIT_STOP) the request stays up (max 120 s); after STOP_CONFIRMED a calm "stop confirmed" note stays 30 s. Display only.
HUB = Path("/dev/shm/live1_hub")
MAPS_ROOT = Path.home() / "sci_ws" / "external" / "sci_live" / "floor_id" / "maps"        # read-only floor plans
MAP_DIRS = dict(kci="kci_maps_perfloor", sw="sw_maps_perfloor", gong="gong_maps_perfloor")
E2_STALE_S, E3_STALE_S, E4_STALE_S = 5.0, 3.0, 3.0
_THUMBS = {}


def floor_thumb(building, floor, max_w=900):
    """Down-scaled floor plan PNG (+ scale factor) for the candidate view. Cached; source maps are only read."""
    k = (building, floor)
    if k not in _THUMBS:
        from PIL import Image
        import io
        im = Image.open(MAPS_ROOT / MAP_DIRS[building] / f"{floor}F" / "maps" / "base_map.png").convert("L")
        f = min(1.0, max_w / im.size[0])
        th = im.resize((max(1, int(im.size[0] * f)), max(1, int(im.size[1] * f))))
        b = io.BytesIO(); th.save(b, "PNG")
        _THUMBS[k] = (b.getvalue(), f, im.size)
    return _THUMBS[k]


def age_of(path):
    try:
        return time.time() - os.stat(path).st_mtime
    except OSError:
        return None


def _health(ok, stale, why):
    return "OK" if ok else ("STALE" if stale else why)


class Panel:
    def __init__(self, state, e5_shm, test_only=False, trial_id=None):
        self.state, self.shm, self.test_only, self.trial_id = Path(state), Path(e5_shm), test_only, trial_id
        C.cpu_pct()

    def lio(self):
        p = self.state / "lio_state.json"
        d, a = C.read_json(p), age_of(p)
        if d is None:
            return dict(status="DATA_UNAVAILABLE", why="lio_monitor not running (FAST-LIO health unknown)")
        od = d.get("odom") or {}
        stale = a is None or a > LIO_STALE_S
        hz, lra = od.get("hz_10s"), od.get("last_rx_age_s")
        base = C.read_json(self.state / "baseline_A1.json")
        if stale:
            st = "STALE"
        elif od.get("n", 0) == 0 or lra is None:
            st = "NO_ODOMETRY"
        elif lra > ODOM_STALE_S:
            st = "ODOM_STALE"
        elif base and base.get("hz_p50") and hz is not None and hz < 0.95 * base["hz_p50"]:
            st = "BELOW_BASELINE_95PCT"
        elif hz is not None and hz < 8.0:
            st = "LOW_RATE"
        else:
            st = "OK"
        acc = d.get("accumulation") or {}
        return dict(status=st, file_age_s=round(a, 2) if a is not None else None, odom=od, accumulation=acc, baseline=base,
                    track_raw=d.get("track_raw", [])[-1500:], frame_note=d.get("frame_note"),
                    map_age_s=None if age_of(self.state / "lio_map.png") is None else round(age_of(self.state / "lio_map.png"), 2))

    def e5(self):
        p = self.shm / "status.json"
        d, a = C.read_json(p), age_of(p)
        if d is None:
            return dict(status="DATA_UNAVAILABLE", why="observe (E5 shadow) not running")
        stale = a is None or a > E5_STALE_S
        now_m, now_w = time.monotonic(), time.time()
        sc = d.get("scene") or {}
        sh = d.get("source_health") or {}
        ws = (d.get("sem1_would_stop_private") or {})
        stop_ui = [r for r in (ws.get("last") or []) if r.get("kind") == "WOULD_STOP" and now_m - float(r.get("mono", -1e9)) <= STOP_UI_S]
        vs = d.get("live2_stop_request") or {}                 # LIVE2 verified visual stop (no frame contract), see observe_node
        e6 = {}                                          # observation_id -> (state, stopped_wall) from the E6 tracker record
        for o in ((C.read_json(self.state / "e6_state.json") or {}).get("recent") or []):
            if o.get("observation_id"):
                e6[o["observation_id"]] = (o.get("state"), o.get("stopped_wall"))
        done = []
        for r in vs.get("last") or []:
            age = now_w - float(r.get("wall", -1e18))
            st, sw = e6.get(r.get("observation_id"), (None, None))
            r = dict(r, e6_state=st, age_s=round(age, 1))
            if st == "STOPPED_TARGET_READY":
                if sw and now_w - float(sw) <= STOP_UI_DONE_S:
                    done.append(dict(r, since_confirm_s=round(now_w - float(sw), 1)))
            elif age <= STOP_UI_VISUAL_S or (st == "WAIT_STOP" and age <= STOP_UI_WAIT_MAX_S):
                stop_ui.append(r)
        infer_age = None if sc.get("stamp") is None else now_w - float(sc["stamp"])
        regions = []
        for r in sc.get("regions") or []:
            regions.append(dict(box=[round(v) for v in r.get("box", [])], spot=r.get("spot"), sim=r.get("sim"), display=r.get("display"),
                                state=r.get("state"), reasons=(r.get("reasons") or [])[:4], z=r.get("z")))
        tm = sc.get("timing") or {}
        return dict(status="STALE" if stale else ("SIGLIP_" + str(d.get("siglip"))), file_age_s=round(a, 2) if a is not None else None,
                    run_id=d.get("run_id"), boot_id=d.get("boot_id"), siglip=d.get("siglip"), source=d.get("source"),
                    mode=(d.get("live1") or {}).get("mode", "E5_STANDALONE"), prompts=d.get("siglip_prompts"),
                    threshold=sc.get("thr"), threshold_mode=(d.get("threshold_mode") or {}).get("mode"),
                    scene=dict(n=sc.get("n_scene"), skipped=sc.get("n_skipped"), t=sc.get("t"), stamp=sc.get("stamp"),
                               source_age_s=None if infer_age is None else round(infer_age, 2),
                               source_old=infer_age is None or infer_age > INFER_OLD_S,
                               scene_ms=tm.get("scene_ms"), siglip_infer_ms=tm.get("siglip_infer_ms"), proposals_ms=tm.get("proposals_ms"),
                               n_tiles=tm.get("siglip_n_tiles"), crop_budget=tm.get("crop_budget"), admission=tm.get("e5rt2_admission")),
                    regions=regions, p2_tracks=(d.get("p2_tracks") or [])[:8], p2_defer=d.get("p2_defer"),
                    would_stop=dict(counts=ws.get("counts"), last=(ws.get("last") or [])[-3:]),
                    obs_flow=self.obs_flow(now_w),
                    stop_request_ui=dict(active=bool(stop_ui), items=stop_ui, n_visual_total=vs.get("n"), confirmed_recent=done,
                                         note="STOP_REQUEST_UI = sign for the OPERATOR only; nothing is sent to the robot"),
                    gate=d.get("e5rt_gate"), tracker=d.get("e5rt_tracker"), source_health=sh,
                    geometry=dict(pose_mode=sh.get("pose_mode"), geometry_status=sh.get("geometry_status"), odom=sh.get("odom")),
                    pantilt=dict(console=d.get("pantilt_console"), execute_enabled=d.get("pantilt_execute_enabled"),
                                 device=d.get("device"), home_last=d.get("home_last")),
                    events=(d.get("events") or [])[-5:], faults=(d.get("faults") or [])[-5:], e4=d.get("e4"),
                    e4_link=d.get("live2_e4_link"), live2_siglip_gate=d.get("live2_siglip_gate"),
                    resources=d.get("resources"),
                    images=dict(live_age_s=_r(age_of(self.shm / "frame.jpg")), inference_age_s=_r(age_of(self.shm / "frame_inference.jpg"))))

    def obs_flow(self, now):
        """Operator state of the newest observation (STOP_REQUESTED .. RESUME_READY / HOLD_ERROR), see live2/obs_flow.py.
        Safety-guard pending conditions are applied by the page (it also holds that state) -> HOLD_ERROR there."""
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live2"))
            import obs_flow as OF
            e6 = C.read_json(self.state / "e6_state.json") or {}
            prec, upd = {}, {}
            for name, d, key in (("e6_precision_results.jsonl", prec, None), ("e6_hypothesis_updates.jsonl", upd, "rows")):
                p = self.state / name
                if p.exists():
                    for line in p.read_text().splitlines()[-400:]:
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        if key:
                            d.setdefault(r.get("observation_id"), []).append(r)
                        else:
                            d[r.get("observation_id")] = r
            lio = self.lio()
            od = lio.get("odom") or {}
            lio_ok = lio.get("status") in ("OK", "LOW_RATE", "BELOW_BASELINE_95PCT") and (od.get("last_rx_age_s") or 9) < 0.5
            br = self.e6_bridge()
            return OF.flow(e6.get("recent"), prec, upd, C.read_json(self.state / "e7_state.json"),
                           br["precision_mode"], lio_ok, False, now, e6_blocked=br.get("blocked"),
                           pair=C.read_json(self.state / "e8_pair_state.json"),         # E8 pair policy (e34) when present
                           manual=C.read_json(self.state / "e6_manual.json"))           # manual focus phases (2026-10-26)
        except Exception as e:                   # noqa: BLE001  display helper must never break the status endpoint
            return dict(state="UNKNOWN", why=f"{type(e).__name__}: {e}")

    def rgb(self):
        """RGB LIVE only if the hub index is fresh AND a new color frame arrived within RGB_STALE_S (process alive is not enough)."""
        d = C.read_json(HUB / "index.json")
        if d is None:
            return dict(status="DATA_UNAVAILABLE", why="sensor hub not running")
        idx_age = time.time() - float(d.get("wall", 0))
        lc = d.get("last_color_wall")
        frame_age = None if lc is None else time.time() - float(lc)
        h = d.get("receiver_health") or {}
        fr = (d.get("frames") or [{}])[-1]
        st = "STALE" if idx_age > 2.0 or frame_age is None or frame_age > RGB_STALE_S else "LIVE"
        ds_ = DF.hub_depth_state(d) if st == "LIVE" else dict(state=DF.DEPTH_MISSING, why="no live colour frame", gap_s=None)
        return dict(status=st, depth_state=ds_["state"], depth_gap_s=ds_.get("gap_s"), depth_why=ds_.get("why"), index_age_s=_r(idx_age), last_frame_age_s=_r(frame_age), last_seq=fr.get("seq"), last_stamp=fr.get("stamp"),
                    header_age_s=None if fr.get("stamp") is None else _r(time.time() - fr["stamp"]),
                    color=h.get("color"), depth=h.get("depth"), odom=h.get("odom"), pose_mode=h.get("pose_mode"),
                    writer=d.get("writer"), live_jpeg_age_s=_r(age_of(HUB / "live.jpg")))

    def siglip(self, e5):
        k = C.read_json(C.RUNTIME / "owned" / "observe.keeper.json")
        ka = age_of(C.RUNTIME / "owned" / "observe.keeper.json")
        sc = (e5 or {}).get("scene") or {}
        e5_age = (e5 or {}).get("file_age_s")
        fresh = e5_age is not None and e5_age <= SIGLIP_STALE_S and (e5 or {}).get("siglip") == "OK"
        if k and ka is not None and ka < 5.0:
            ks = k.get("siglip")
            if ks == "RUNNING":
                st = "LIVE" if fresh else "STARTING_OR_STALLED"
            elif ks in ("PAUSED_THERMAL", "PAUSED_MEMORY", "WAITING_RESOURCES", "WAITING_HUB"):
                st = "PAUSED"
            elif ks in ("ERROR",):
                st = "ERROR"
            else:
                st = ks
            return dict(status=st, keeper=ks, why=k.get("why"), runs=len(k.get("runs", [])), last_run=(k.get("runs") or [None])[-1],
                        scene_age_s=e5_age)
        if (e5 or {}).get("status") == "DATA_UNAVAILABLE":
            return dict(status="NOT_RUNNING", why="observe not started")
        return dict(status="LIVE" if fresh else "STALE", why=None if fresh else "no keeper; status.json not fresh", scene_age_s=e5_age)

    def localization(self, lio, l2=None):
        last = ((lio or {}).get("odom") or {}).get("last_raw")
        e4 = (l2 or {}).get("e4") or {}
        return dict(status="GLOBAL_UNKNOWN", pose_raw=last, e4_state=e4.get("state"),
                    why="FAST-LIO odometry is relative to its start; E2/E3 hypotheses are unverified and E6/E7 are not connected "
                        "-> the global pose stays UNKNOWN")

    def e6_bridge(self):
        """Automatic E6 (ctl module e6) as the bridge reports it, cross-checked with the process: a RUNNING record without a
        live bridge (or a silent one) is LOST -> treated as blocked (no drive-on guidance for later stops)."""
        b = C.read_json(self.state / "e6_bridge.json") or {}
        alive = C.owner_alive(C.owner_record("e6")) or bool(C.find_pids("e6_precision_bridge.py"))   # module e6 or a manual bridge
        st, why = b.get("state") or ("RUNNING" if alive else "OFF"), b.get("why")
        age = None if not b.get("wall") else time.time() - float(b["wall"])
        if st == "RUNNING" and b and (not alive or (age or 0) > 10.0):
            st, why = "LOST", "E6 bridge stopped or silent during the experiment"
        blocked = (why or st) if st in ("BLOCKED", "BLOCKED_ENDED", "LOST") else None
        return dict(state=st, why=why, alive=alive, age_s=_r(age), focus_mode=b.get("focus_mode"), af_strategy=b.get("af_strategy"),
                    n_observations=b.get("n_observations"), blocked=blocked,
                    precision_mode=alive or st in ("BLOCKED", "BLOCKED_ENDED", "LOST"))

    def e6_aim(self, br):
        """Aim diagnosis of the newest observation (live2/e6_aim_diag.report): times, detection->stop motion, target before /
        after, uncorrected / corrected / re-detected aim, envelope and ONE specific reason when the pan-tilt does not move."""
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live2"))
            import e6_aim_diag as AD
            import e6_precision_bridge as B
            o = AD.latest((C.read_json(self.state / "e6_state.json") or {}).get("recent"))
            if o is None:
                return None
            prec = None
            pp = self.state / "e6_precision_results.jsonl"
            if pp.exists():
                for ln in pp.read_text().splitlines()[-60:][::-1]:
                    r = json.loads(ln)
                    if r.get("observation_id") == o["observation_id"]:
                        prec = r
                        break
            out = AD.report(o, prec, br, C.approval_status("B", "pantilt"), B.load_envelope())
            out["event_image"] = AD.event_image(self.state, C.read_json(self.shm / "status.json"), o["observation_id"]) is not None
            return out
        except Exception as e:                                                # noqa: BLE001  display only
            return dict(status="DIAG_ERROR", ko=f"{type(e).__name__}: {e}")

    def stage_milestones(self, G, e3, now):
        """2026-10-31: first time each display stage was seen in THIS odometry session -> <state>/stage_milestones.json (record only;
        the PROVISIONAL time is the 'first provisional position' of the time metrics). Written by the web process only on change."""
        sess = (e3 or {}).get("session")
        st = (G or {}).get("stage")
        if not sess or not st:
            return
        p = self.state / "stage_milestones.json"
        M = getattr(self, "_stage_ms", None)
        if not M or M.get("session") != sess:
            M = C.read_json(p) or {}
            if M.get("session") != sess:
                M = dict(kind="LIVE2_STAGE_MILESTONES", session=sess, first={})
            self._stage_ms = M
        if st not in M["first"]:
            ev = (G.get("evidence") or {})
            M["first"][st] = dict(wall=now, leader=ev.get("leader"), floor=ev.get("leader_floor"), floors_exact=ev.get("floors_exact"))
            try:
                C.atomic_json(p, M)
            except OSError:
                pass

    def fix_policy(self):
        """2026-10-30 (staging): <state>/e7_policy.json as written by the RUNNING e34 (e7_policy.write_report) + whether that
        pid is still an e34_node process — a file left by an earlier run is shown as such, never as the current setting."""
        P = C.read_json(self.state / "e7_policy.json")
        if not P:
            return None
        try:
            cmd = Path(f"/proc/{int(P.get('pid'))}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, TypeError, ValueError):
            cmd = ""
        return dict(P, module_alive="e34_node.py" in cmd)

    def e6_manual(self):
        """Manual-focus snapshot (e6_manual.json) with its age; only while E6 runs in precision mode."""
        d = C.read_json(self.state / "e6_manual.json")
        if not d:
            return None
        d["age_s"] = _r(time.time() - float(d.get("wall") or 0))
        return d

    def live2(self):
        """E2 / E3 / E4 / feed state as written by the LIVE2 processes (age on every block; nothing is synthesised)."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live2"))
        e2p, e3p, e4p, fp = self.state / "e2" / "e2_state.json", self.state / "e3_state.json", self.state / "e4_state.json", \
            self.state / "e2_feed_state.json"
        e2, e3, e4, fd = (C.read_json(p) for p in (e2p, e3p, e4p, fp))
        ages = {k: _r(age_of(p)) for k, p in (("e2", e2p), ("e3", e3p), ("e4", e4p), ("feed", fp))}
        br = self.e6_bridge()
        precision = br["precision_mode"]                                    # E6 hardware path only runs through this bridge
        last = None                                                         # newest E6 precision record (AF / capture / OCR)
        pp = self.state / "e6_precision_results.jsonl"
        if pp.exists():
            try:
                r = json.loads(pp.read_text().splitlines()[-1])
                last = dict({k: r.get(k) for k in ("observation_id", "status", "why", "ocr_lines", "vlm", "capture_file")},
                            focus=r.get("focus"), image_quality=r.get("image_quality"), focus_mode=r.get("focus_mode"))
            except (ValueError, IndexError, OSError):
                last = None
        out = dict(ages=ages, final_pose="UNKNOWN",
                   final_pose_why="no E7 global decision yet: every candidate is an unverified hypothesis",
                   e6_mode=dict(mode="E2_E7_PRECISION" if precision else "E2_E5_OBSERVATIONAL",
                                ko=("E6 자동 실행 중단됨 — 운용자 점검 필요" if br.get("blocked") else
                                    "E6 자동 실행 — 정지 확인 후 팬틸트·자동 초점·촬영·OCR" if precision else
                                    "관측 전용 (E6 팬틸트·초점 비활성: 정지해도 팬틸트는 움직이지 않음)")),
                   e6_precision_last=last, e6_bridge=br,
                   e7_observation=C.read_json(self.state / "e7_observation.json"),
                   e8_pair=C.read_json(self.state / "e8_pair_state.json"),
                   e6_manual=self.e6_manual(), e6_aim=self.e6_aim(br),
                   structure_path=C.read_json(self.state / "structure_path.json"),
                   auto_state=C.read_json(self.state / "auto_state.json"), fix_current=C.read_json(self.state / "fix_current.json"), e7_long=C.read_json(self.state / "e7_long.json"), vlm=C.read_json(self.state / "vlm_state.json"), semantic_estimate=C.read_json(self.state / "semantic_estimate.json"), e7_semantic_shadow=C.read_json(self.state / "e7_semantic_shadow.json"), final_verify=C.read_json(self.state / "final_verify.json"),
                   fix_policy=self.fix_policy(),
                   semantic_recovery=C.read_json(self.state / "semantic_recovery.json"),
                   semantic_recovery_single=C.read_json(self.state / "semantic_recovery_single.json"),
                   loc_overlay=C.read_json(self.state / "loc_overlay.json"),
                   fix_overlay=C.read_json(self.state / "fix_overlay.json"),          # 2026-11-01 r33: E7 points of the FIX (e34)
                   stage_milestones=C.read_json(self.state / "stage_milestones.json"))
        if fd:
            out["feed"] = fd
        if e2:
            pf = e2.get("per_floor") or {}
            out["e2"] = dict(status=e2.get("status") if (ages["e2"] or 99) <= E2_STALE_S else "STALE", raw_status=e2.get("status"),
                             building=e2.get("building"), building_name=e2.get("building_name"), pass_id=e2.get("pass_id"),
                             pass_why=e2.get("pass_why"), t_now=e2.get("t_now"), floors_total=e2.get("floors_total"),
                             floors=e2.get("floors"), floors_done=e2.get("floors_done"), floors_pending=e2.get("floors_pending"),
                             floors_failed=e2.get("floors_failed"), held=e2.get("held"), wait_note=e2.get("wait_note"),
                             pass_sec=e2.get("pass_sec"), device=e2.get("device"), procs=e2.get("procs"), algorithm=e2.get("algorithm"),
                             regional=[{k: r.get(k) for k in ("request_id", "floor", "status", "n", "sec", "attempt", "why_failed")}
                                       for r in (e2.get("regional") or [])[-3:]],
                             pass_started_wall=e2.get("pass_started_wall"), finished_wall=e2.get("finished_wall"), started_wall=e2.get("started_wall"),
                             per_floor={k: {kk: v.get(kk) for kk in ("status", "sec", "open_s", "fit_s", "score_s", "n_raw", "n_picked",
                                                                     "rss_peak_mb", "beta_deg", "mirror", "H", "W", "error", "open_src")}
                                        for k, v in pf.items()},
                             top=(e2.get("candidates_pass") or e2.get("candidates") or [])[:12], events=(e2.get("events") or [])[-6:],
                             coarse={k: v for k, v in (e2.get("coarse") or {}).items() if k != "per_floor"} or None,
                             top_coarse=(e2.get("candidates_coarse") or [])[:10])
        if e3:
            out["e3"] = dict(status="STALE" if (ages["e3"] or 99) > E3_STALE_S else "LIVE", **{k: e3.get(k) for k in (
                "session", "n_alive", "n_active", "n_suspended", "n_dropped", "n_keyframes", "kf_pending", "last_kf", "timing",
                "events", "params")}, hypotheses=(e3.get("hypotheses") or [])[:40], dropped=(e3.get("dropped") or [])[-8:])
        if e4:
            out["e4"] = dict(e4, status="STALE" if (ages["e4"] or 99) > E4_STALE_S else "LIVE")
        e6 = C.read_json(self.state / "e6_state.json")
        if e6:                                       # E6 observation geometry (prediction only)
            out["e6"] = dict(e6, age_s=_r(age_of(self.state / "e6_state.json")))
        e7 = C.read_json(self.state / "e7_state.json")
        if e7:                                       # E6 semantic + E7 geometry -> global decision (never shown as confirmed otherwise)
            out["e7"] = dict({k: e7.get(k) for k in ("observation_id", "source", "semantic_hypothesis", "global_decision", "why", "e7_s",
                                                      "path", "written_wall", "used_observation_ids")},
                             best_result=__import__("e8_pair").e7_best(e7) if e7.get("e7") else None,
                             best=(e7.get("e7") or {}).get("best"), gap_m=(e7.get("e7") or {}).get("gap_m"),
                             age_s=_r(age_of(self.state / "e7_state.json")))
        try:                                         # 2026-10-30 staged view (display only; FIX path unchanged) — live2/e4_stage.py
            import e4_stage as STG
            now = time.time()
            tk = getattr(self, "_stage_trk", None)
            sess = (e3 or {}).get("session")
            if not tk or tk.get("session") != sess:
                tk = self._stage_trk = dict(session=sess, lead=None, lead_since=None, last_amb=None, full_since=None)
            Hs = sorted([h for h in (e3 or {}).get("hypotheses") or [] if h.get("status") in ("ACTIVE", "SUSPENDED")], key=lambda h: h.get("score", 9))
            if Hs and Hs[0].get("id") != tk["lead"]:
                tk.update(lead=Hs[0].get("id"), lead_since=now)
            if (e4 or {}).get("state") == "AMBIGUOUS":
                tk["last_amb"] = now
            if e2 and e2.get("floors_total") and len(e2.get("floors_done") or []) >= e2["floors_total"] and tk["full_since"] is None:
                tk["full_since"] = now
            out["stage"] = STG.stage(e2, e3, e4, C.read_json(self.state / "e7_state.json"), now, lead_since=tk["lead_since"],
                                     full_since=tk["full_since"], last_amb=tk["last_amb"], sp=C.read_json(self.state / "structure_path.json"))
            self.stage_milestones(out["stage"], e3, now)
        except Exception as ex:                      # noqa: BLE001  display add-on must never break the snapshot
            out["stage"] = dict(stage=None, error=f"{type(ex).__name__}: {ex}")
        sr = C.read_json(self.state / "siglip_request.json")
        if sr:                                       # E4 -> SigLIP activation request (E4 asks; observe decides)
            out["siglip_request"] = dict(sr, age_s=_r(age_of(self.state / "siglip_request.json")))
        return out

    def components(self):
        out = {}
        for m in ("fastlio", "lio", "lio2", "hub", "observe", "rec", "e2", "e34", "replay", "web"):
            rec = C.owner_record(m)
            sup = C.read_json(C.RUNTIME / "owned" / f"{m}.sup_state.json")
            out[m] = dict(owned=bool(rec), alive=C.owner_alive(rec), pid=(rec or {}).get("pid"), started=(rec or {}).get("started_wall"),
                          supervisor=sup, supervisor_state_age_s=_r(age_of(C.RUNTIME / "owned" / f"{m}.sup_state.json")))
        return out

    def snapshot(self):
        e5 = self.e5()
        lio = self.lio()
        ocr_state = "NOT_RUN (reader not started; G2 static optics needs approval B)"
        sess = C.read_json(C.STATE / "session.json") or {}
        l2 = self.live2()
        return dict(rgb=self.rgb(), siglip=self.siglip(e5), localization=self.localization(lio, l2), session=sess, live2=l2,
                    meta=dict(kind="LIVE1_STATE", test_only=self.test_only, trial_id=self.trial_id, server_wall=time.time(),
                              server_time=C.now_wall(), boot_id=C.boot_id(), host=C.host_addresses(),
                              note="read-only panel; no command endpoint"),
                    approvals={k: C.approval_status(k) for k in ("A", "B")},
                    contract=self.contract(), components=self.components(), lio=lio, e5=e5,
                    resources=C.resources(),
                    research=dict(
                        E2=dict(status=((l2.get("e2") or {}).get("status") or "NOT_RUNNING"),
                                why="student metric_fit_v2 internal candidates on FAST-LIO output proxy (no raw /lidar_points)"),
                        E3=dict(status=((l2.get("e3") or {}).get("status") or "NOT_RUNNING"), why="odometry propagation + keyframe re-scoring"),
                        E4=dict(status=((l2.get("e4") or {}).get("state") or "NOT_RUNNING"),
                                why="; ".join((l2.get("e4") or {}).get("reasons") or []) or "E4 not running"),
                        E5_SEES_E4=dict(status=(e5.get("e4") or {}).get("status", "E4_UNAVAILABLE"), why=(e5.get("e4") or {}).get("why", "")),
                        E5=dict(status=e5.get("mode", "E5_STANDALONE") if e5.get("status") != "DATA_UNAVAILABLE" else "NOT_RUNNING"),
                        OCR_VLM=dict(status=ocr_state),
                        SEMANTIC_MAP=dict(status="NO_TEXT_YET", candidates=[], why="no OCR/VLM text; label matching needs a real reading"),
                        E6=dict(status="NOT_RUN", why="needs OCR text + verified map metric/anchor contract"),
                        E7=dict(status="NOT_WIRED", why="no live E7 rematch connected; no final pose is shown")))

    def contract(self):
        d = C.read_json(C.CONTRACT)
        if d is None:
            return dict(status="UNVERIFIED_FRAME_CONTRACT", why="no verified /Odometry frame contract; geometry/STOP evaluation off")
        ok = bool(d.get("verified_by") and d.get("evidence") and d.get("body_transform"))
        return dict(status="CONTRACT_PRESENT" if ok else "CONTRACT_INCOMPLETE", body_transform=d.get("body_transform"),
                    verified_by=d.get("verified_by"), frame_id=d.get("frame_id"), child_frame_id=d.get("child_frame_id"))


def _r(v):
    return None if v is None else round(v, 2)


def make_handler(panel):
    files = {"/img/live.jpg": lambda: panel.shm / "frame.jpg", "/img/inference.jpg": lambda: panel.shm / "frame_inference.jpg",
             "/img/map.png": lambda: panel.state / "lio_map.png", "/img/rgb.jpg": lambda: HUB / "live.jpg"}

    class H(BaseHTTPRequestHandler):
        server_version = "LIVE1Panel/1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-LIVE1-MODE", "TEST_ONLY" if panel.test_only else "LIVE")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(200, (HERE / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/state":
                return self._send(200, json.dumps(panel.snapshot(), default=str).encode(), "application/json")
            if path.startswith("/img/floor/"):
                try:
                    _, _, _, b, f = path.split("/")
                    data, fac, size = floor_thumb(b, int(f.split(".")[0]))
                except Exception:
                    return self._send(404, b"NO_FLOOR_PLAN", "text/plain")
                return self._send(200, data, "image/png")
            if path.startswith("/img/e5event/"):       # RealSense scene of the SigLIP detection behind one STOP request
                name = path[len("/img/e5event/"):]
                if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}\.jpg", name) or ".." in name:
                    return self._send(404, b"BAD_NAME", "text/plain")
                sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live2"))
                import e6_aim_diag as AD
                f = AD.event_image(panel.state, C.read_json(panel.shm / "status.json"), name[:-4])
                if f is None:
                    return self._send(404, b"NO_EVENT_IMAGE", "text/plain")
                return self._send(200, f.read_bytes(), "image/jpeg")
            if path.startswith("/img/e8/"):            # E8 pair: stored IMX477 capture of one observation (read-only)
                name = path[len("/img/e8/"):]
                if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}\.jpg", name) or ".." in name:
                    return self._send(404, b"BAD_NAME", "text/plain")
                try:
                    return self._send(200, (panel.state / "e6_captures" / name).read_bytes(), "image/jpeg")
                except OSError:
                    return self._send(404, b"NO_CAPTURE", "text/plain")
            if path == "/api/floor_meta":
                q = dict(x.split("=") for x in self.path.split("?", 1)[1].split("&")) if "?" in self.path else {}
                try:
                    data, fac, size = floor_thumb(q["b"], int(q["f"]))
                    return self._send(200, json.dumps(dict(factor=fac, size=size)).encode(), "application/json")
                except Exception:
                    return self._send(404, b"{}", "application/json")
            if path in files:
                p = files[path]()
                try:
                    data = p.read_bytes()
                except OSError:
                    return self._send(404, b"DATA_UNAVAILABLE", "text/plain")
                return self._send(200, data, "image/png" if path.endswith(".png") else "image/jpeg")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            self._send(405, b"read-only panel: no commands", "text/plain")
        do_PUT = do_DELETE = do_PATCH = do_POST

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--state", default=str(C.STATE))
    ap.add_argument("--e5-shm", default=str(C.RUNTIME / "e5_shm"))
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--trial-id", default=None)
    a = ap.parse_args()
    panel = Panel(a.state, a.e5_shm, test_only=a.test_only, trial_id=a.trial_id)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(panel))
    print(f"LIVE1_WEB listening {a.host}:{a.port} test_only={a.test_only}", flush=True)
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    srv.server_close()


if __name__ == "__main__":
    main()
