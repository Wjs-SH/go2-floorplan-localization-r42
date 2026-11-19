#!/usr/bin/env python3
"""e34_node.py — LIVE2 E3 (incremental hypothesis update) + E4 (ambiguity) process. Files only: no ROS, no GPU, no device.

Reads : <store> (student LiveStore from lio_monitor2), <state>/e2/e2_state.json, <state>/lio_state.json,
        <state>/e2_feed_state.json, semantic map packages (read-only)
Writes: <state>/e3_state.json, <state>/e4_state.json (2 Hz), <state>/e4_feed.json (E5 input: stamp, level, margin + the
        semantic request), <state>/e3_history.jsonl, <state>/e4_history.jsonl, <state>/e2/e2_request.json (re-search)
E4 'level' sent to E5 is the E4 state name itself; E5 suppresses only on level == 'CONFIDENT', which LIVE2 never sends
(geometry-only confidence is not a verified pose, so semantic observation is never suppressed by it).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CAND / "external" / "sci_live" / "floor_id" / "localizer"))     # twoway (numpy only)
import e3_tracker as E3  # noqa: E402
import e4_ambiguity as E4  # noqa: E402
import e6_tracker as E6  # noqa: E402
import e6e7_link as E67  # noqa: E402


RESEARCH_AFTER_M = 30.0


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, default=str))
    os.replace(tmp, path)


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--building", required=True)
    ap.add_argument("--semantic-root", default=str(Path.home() / "sci_ws" / "maps" / "kci_handoff"))
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--hub", default="/dev/shm/live1_hub", help="sensor hub (E6: raw odometry ring + camera K)")
    ap.add_argument("--e5-status", default=None, help="E5 status.json (E6: run dir of the stop requests); default <state>/../e5_shm")
    ap.add_argument("--max-s", type=float, default=3600.0)
    ap.add_argument("--clock", choices=("wall", "replay"), default="wall",
                    help="replay: 'now' for freshness = newest sample wall time written by the replay feeder")
    a = ap.parse_args()
    state = Path(a.state)
    e2_dir = state / "e2"
    e2_dir.mkdir(parents=True, exist_ok=True)
    tr = E3.Tracker(a.store, e2_dir, log_path=state / "e3_events.jsonl")
    sem = E4.SemanticMaps(a.semantic_root, a.building)
    e6 = E6.E6Tracker(state, a.hub, a.e5_status or (state.parent / "e5_shm" / "status.json"))
    link = E67.E6E7Link(state, tr)
    import loc_overlay as LO                     # 2026-10-31: LiDAR-on-map view of the leader (display only, <state>/loc_overlay.json)
    ovl = LO.Overlay(state, tr, E67.E7)
    import e7_long as EL                         # R36: whole-path floor comparison, record and display only, in its own low-priority process
    long_st = EL.LongStructure(state, tr)
    import semantic_view as SV                   # R37: the map's estimate from the first room reading, display only
    sem_view = SV.View(state, tr, link.pair)
    loc_mode = (read_json(state.parent / "ctl" / "experiment.json") or {}).get("loc_mode")
    final_v = None
    auto = loc_mode == "AUTO_INITIAL_GLOBAL"     # R39: one automatic run; the machine below chooses the method from E2/E3/E4/E7
    fsm = None
    hw_now = ["E6_IDLE"]
    fc_last = [None, 0.0]                 # R42: last odom_t written to fix_current.jsonl, wall
    if auto:                                     # R37: the structure path's stop request, its own sound and text
        import auto_fsm as AF
        import final_verify as FV
        import e6_core as G6_
        fsm = AF.AutoFSM(state)

        def _stop_now():
            smp_ = [tuple(o_[:4]) for o_ in ((e6.hub_index() or {}).get("odom") or [])]
            return G6_.stop_state(smp_, smp_[-1][0], wall_now=time.time()) if smp_ else dict(stopped=False, why="NO_ODOMETRY")
        final_v = FV.FinalVerify(state, link, long_st, _stop_now, busy_fn=lambda: hw_now[0] in AF.HW_BUSY)
    last_state, last_req, sreq, sreq_base, milestones = None, None, None, None, {}
    t_end = time.time() + a.max_s
    stop = [False]
    import signal
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    n_kf_hist = 0
    sys.path.insert(0, str(CAND / "live1"))
    import live1_common as LC
    t_res = 0.0
    while not stop[0] and time.time() < t_end:
        t0 = time.perf_counter()
        pause = read_json(state / "control" / "pause.json") or {}
        paused = bool(pause.get("paused"))             # operator pause: no new keyframe evaluation, no re-search request
        n = 0 if paused else tr.tick()
        t_e3 = time.perf_counter()
        e3 = tr.snapshot()
        e2 = read_json(e2_dir / "e2_state.json")
        lio = read_json(state / "lio_state.json")
        feed = read_json(state / "e2_feed_state.json")
        now = time.time()
        if a.clock == "replay" and lio:
            lio = dict(lio, odom=dict(lio.get("odom") or {}, last_rx_age_s=round(now - float(lio.get("written_wall", now)), 3)))
        for f, pf in ((e2 or {}).get("per_floor") or {}).items():
            if pf.get("H") and pf.get("W"):
                sem.loc_sizes[int(f)] = (pf["W"], pf["H"])
        e4 = E4.classify(e2, e3, feed, lio, tr.store.rows, now)
        walls = {}
        for h in tr.alive():
            if h.floor not in walls:
                walls[h.floor] = tr.light(h.light_dir).wall_real
        need = E4.semantic_need(e4, e3["hypotheses"], sem, walls)
        t_e4 = time.perf_counter()
        e4.update(semantic=need, timing_ms=dict(e3_tick=round(1000 * (t_e3 - t0), 1), e4=round(1000 * (t_e4 - t_e3), 1)),
                  e3_counts=dict(alive=e3["n_alive"], active=e3["n_active"], suspended=e3["n_suspended"], dropped=e3["n_dropped"],
                                 keyframes=e3["n_keyframes"], kf_pending=e3["kf_pending"]),
                  final_pose=dict(global_decision=(link.snapshot() or {}).get("global_decision"),
                                  owner="e6e7_link (decisive semantic + independent E7 geometry)", hypotheses_are="UNVERIFIED"),
                  fix_contract=E4.FIX_CONTRACT,
                  milestones=E4.update_milestones(milestones, now, e3, need, state, link.snapshot()),
                  paused=dict(paused=paused, since=pause.get("since"), request_id=pause.get("request_id")) if paused else None)
        try:                                     # E6: observation geometry (prediction only); never blocks E3/E4
            hy = [dict(id=h.id, floor=h.floor, score=h.score(), rec=h.rec) for h in sorted(tr.alive(), key=lambda h: h.rank_key())]
            e6.tick(hy, sem)
            atomic_json(state / "e6_state.json", dict(e6.snapshot(), written_wall=now))
            lr = ((lio or {}).get("odom") or {}).get("last_raw") or {}
            odom_now = None if lr.get("x") is None else (lr.get("stamp"), float(lr["x"]), float(lr["y"]), None)
            link.tick(e6.snapshot().get("recent"), odom_now)   # E8 pair (default) / single: semantic -> E7 (background)
            import e6_core as G6                                # PATH A: E4 CONFIDENT_GEOMETRY -> STOP request -> structure-only E7
            smp = [tuple(o_[:4]) for o_ in ((e6.hub_index() or {}).get("odom") or [])]
            robot_stop = G6.stop_state(smp, smp[-1][0], wall_now=now) if smp else dict(stopped=False, why="NO_ODOMETRY")
            link.structure_tick(e4, now, stop=robot_stop)      # NOT 'stop': that name is the loop's shutdown flag (2026-10-28)
            link.early_shadow_tick(now)                        # R34: record only, no FIX path (R36: off by default; e7_long replaces it)
            if link.pair is None or (link.pair.status != "E7_VERIFYING" and not link.pair.vlm_pending()):   # R41: one E7 at a time: the plates' E7 waited 52-79 s behind the whole-path comparison and 3 E2 children
                long_st.tick(now)
            sem_view.tick(now)
            if final_v is not None:
                final_v.tick(now)
            ovl.tick(now)
            fc_ = link.current_pose()
            if fc_:
                atomic_json(state / "fix_current.json", fc_)
                if fc_.get("odom_t") != fc_last[0] and now - fc_last[1] >= 1.0:     # R42: the post-FIX pose as a series (R41 kept only the newest: the current-pose error after a FIX could not be computed from a field record)
                    fc_last[:] = [fc_.get("odom_t"), now]
                    with open(state / "fix_current.jsonl", "a") as f_:
                        f_.write(json.dumps(fc_, default=str) + "\n")
            if fsm is not None:                  # R39: after every input of this tick; a failure here leaves auto_state old, and visual_stop then fails towards asking
                man, stl = read_json(state / "e6_manual.json") or {}, read_json(state / "e6_settle.json") or {}
                hw_now[0] = AF.phase_to_hw(man.get("phase") if now - float(man.get("wall") or 0) < 600 else None, stl.get("state"))
                rows_ = getattr(tr.store, "rows", None) or []
                px_ = sum(((b_[2] - a_[2]) ** 2 + (b_[3] - a_[3]) ** 2) ** 0.5 for a_, b_ in zip(rows_, rows_[1:]))
                xp = read_json(state.parent / "ctl" / "experiment.json") or {}
                vs_ = (read_json(state / "vlm_state.json") or {}).get("state")
                fsm.tick(e2=e2, e3=e3, e4=e4, long_d=long_st.d, path_m=px_, global_decision=(read_json(state / "e7_state.json") or {}).get("global_decision"),
                         pair_status=link.pair.status if link.pair is not None else None, semantic_status=(read_json(state / "semantic_estimate.json") or {}).get("status"),
                         final_status=final_v.d.get("status"), hw=hw_now[0], e6_alive=((xp.get("modules") or {}).get("e6") or {}).get("state") == "RUNNING",
                         vlm_busy=vs_ in ("LOADING", "ASKING"), vlm_resident=vs_ in ("LOADING", "ASKING", "READY"), vlm_pending=link.pair.vlm_pending() if link.pair is not None else False, now=now)
        except Exception as ex:                  # noqa: BLE001
            atomic_json(state / "e6_state.json", dict(kind="LIVE2_E6_STATE", error=f"{type(ex).__name__}: {ex}", written_wall=now))
        atomic_json(state / "e3_state.json", dict(e3, written_wall=now))
        atomic_json(state / "e4_state.json", dict(e4, written_wall=now))
        atomic_json(state / "e4_feed.json", dict(stamp=now, level=e4["state"], margin=e4["metrics"].get("gap"),
                                                  priority=need["priority"], needed=need["needed"], why=need["why"],
                                                  wanted_labels=need["wanted_labels"], targets=need["targets"][:5],
                                                  floors_in_question=need["floors_in_question"], source="LIVE2_E4"))
        sreq_base = E4.siglip_request(e4, need, sreq_base, now)
        sreq = link.pair.siglip_override(sreq_base) if link.pair is not None else sreq_base   # E8: next plate / reading
        if fsm is not None:                      # R39: SigLIP stays on until a real FIX, never stopped by a time limit or a dialog; after the FIX it has nothing left to find
            if fsm.d.get("state") == "GLOBAL_FIX":
                sreq = dict(sreq, mode="IDLE_OK", why="GLOBAL_FIX")
            elif fsm.d.get("structure_class") == "AMBIGUOUS":
                sreq = dict(sreq, mode="ACTIVE", why=f"AUTO_AMBIGUOUS ({sreq.get('why')})")
        atomic_json(state / "siglip_request.json", sreq)
        if sreq["event"]:
            with open(state / "e4_events.jsonl", "a") as f:
                f.write(json.dumps(dict(wall=now, event=sreq["event"], seq=sreq["seq"], mode=sreq["mode"], e4_state=e4["state"],
                                        display_ko=e4["display_ko"], candidates=e4["candidates"], reasons=e4["reasons"][:3]),
                                   ensure_ascii=False, default=str) + "\n")
        if e4["state"] != last_state or n:
            with open(state / "e4_history.jsonl", "a") as f:
                f.write(json.dumps(dict(wall=now, state=e4["state"], reasons=e4["reasons"], metrics=e4["metrics"],
                                        counts=e4["e3_counts"], priority=need["priority"], n_targets=len(need["targets"]),
                                        top_target=(need["targets"] or [None])[0], timing_ms=e4["timing_ms"]),
                                   ensure_ascii=False, default=str) + "\n")
            last_state = e4["state"]
        if e3["n_keyframes"] != n_kf_hist or n:
            n_kf_hist = e3["n_keyframes"]
            with open(state / "e3_history.jsonl", "a") as f:
                f.write(json.dumps(dict(wall=now, last_kf=e3["last_kf"], alive=e3["n_alive"], active=e3["n_active"],
                                        suspended=e3["n_suspended"], dropped=e3["n_dropped"], kf_pending=e3["kf_pending"],
                                        top=[dict(id=h["id"], floor=h["floor"], score=h["score"], status=h["status"], n_kf=h["n_kf"])
                                             for h in e3["hypotheses"][:6]], timing=e3["timing"]), default=str) + "\n")
        if now - t_res >= 2.0:                      # resource + latency sample (RESOURCE_AND_LATENCY.csv)
            t_res = now
            r = LC.resources()
            with open(state / "res_history.jsonl", "a") as f:
                f.write(json.dumps(dict(wall=now, temp=r["max_temp_c"], mem_avail_mb=round(r["mem_available_mb"]), cpu=r["cpu_pct"],
                                        gpu=r["gpu_load_pct"], load1=r["loadavg"][0], e3_tick_ms=e4["timing_ms"]["e3_tick"],
                                        e4_ms=e4["timing_ms"]["e4"], kf_ms_p95=e3["timing"]["kf_ms_p95"],
                                        odom_hz=((lio or {}).get("odom") or {}).get("hz_10s"),
                                        odom_age=((lio or {}).get("odom") or {}).get("last_rx_age_s"),
                                        feed_ms_p95=(feed or {}).get("sample_ms_p95"), state=e4["state"],
                                        cpu_by=LC.cpu_by_role())) + "\n")
        req = None
        if paused:
            time.sleep(max(0.0, 1.0 / a.hz - (time.perf_counter() - t0)))
            continue
        actionable = [r for r in e4["reasons"] if "STALE" not in r and "UNAVAILABLE" not in r]
        if e4["state"] == "RECOVERY_REQUIRED" and actionable:      # stale / missing input: no new data, re-search is pointless
            req = dict(why="E4_RECOVERY:" + "|".join(actionable)[:200], wall=now)
        elif e4["state"] == "GEOMETRY_INCONSISTENT" and not auto:   # R37: repeat searches feed E3 all the time; a new pass would throw the exact rivals away
            req = dict(why="E4_GEOMETRY_INCONSISTENT:" + "|".join(e4["reasons"])[:200], wall=now)
        elif e4["state"] == "AMBIGUOUS" and (e2 or {}).get("status") == "IDLE" and (e2 or {}).get("t_now") and not auto and not (link.pair is not None and (link.pair.group or link.pair.history)):
            # R37: not once readings guide the search (field +683 s: a second pass wiped the exact rivals of the running E7, and with the VLM ran out of memory)
            moved = E4.path_since(tr.store.rows, float(e2["t_now"]))
            if moved >= RESEARCH_AFTER_M:              # a new E2 window over new geometry may add / separate hypotheses
                req = dict(why=f"AMBIGUOUS_RESEARCH moved {moved:.0f} m since E2 window", wall=now)
        elif tr.request and tr.request.get("wall", 0) > (last_req or {}).get("wall", 0) and (not auto or tr.request.get("why") == "ODOM_SESSION_CHANGED"):   # R40 X20261103_001810 +1042 s: a GEOMETRY_INCONSISTENT pass wiped the exact rivals of 9/9 and found nothing
            req = tr.request
        if req and (last_req is None or now - last_req["wall"] > 120.0):
            atomic_json(e2_dir / "e2_request.json", req)
            last_req = req
            tr.request = None
        time.sleep(max(0.0, 1.0 / a.hz - (time.perf_counter() - t0)))


if __name__ == "__main__":
    main()
