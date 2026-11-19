"""ready_gate.py — EXPERIMENT_READY_TO_MOVE: may the operator start driving? (2026-10-27 field: no signal after [실험 시작]).

Evaluated by the console watchdog every 2 s from the SAME read-only panel state the tablet shows; latched ONCE per experiment
(experiment.json 'ready', timeline event experiment_ready_to_move, event_id '<exp_id>:READY'). Nothing is sent to the robot.
Required (all, measured — a pressed button or a live process is not enough):
  session      experiment RUNNING (no failed module), its boot_id == this boot
  fastlio      FAST-LIO module RUNNING (its start gate = map kd-tree initialised), odometry status OK, newest pose < 0.5 s old,
               10-s rate >= 8 Hz (the LIVE2 E7 / E8 line)
  rgbd         RealSense colour frames LIVE through the sensor hub
  depth        R34: the newest colour frame has aligned depth within the freshness limit (11-01: depth froze and nothing said so)
  siglip       SigLIP keeper LIVE (only when the experiment was started with SigLIP)
  e2_e3_e4     E2 coordinator + E3 + E4 state files written within 5 s and the E2 floor set loaded (maps read); E2 / E4 RESULTS
               are not required: they need motion
  safety       the safety guard has evaluated this experiment and nothing is pending
Reported, NOT required: E6 (pan-tilt / IMX477) — off without approval B; the page shows it separately.
"""
HZ_MIN, POSE_AGE_MAX, STATE_AGE_MAX = 8.0, 0.5, 5.0


def evaluate(snap, exp, guard_last, boot_id, e6=None):
    lio, od = (snap or {}).get("lio") or {}, ((snap or {}).get("lio") or {}).get("odom") or {}
    l2 = (snap or {}).get("live2") or {}
    ages, e2 = l2.get("ages") or {}, l2.get("e2") or {}
    mods = (exp or {}).get("modules") or {}
    items = []

    def add(key, ok, ko):
        items.append(dict(key=key, ok=bool(ok), ko=ko))
    add("session", (exp or {}).get("state") == "RUNNING" and not (exp or {}).get("failed") and (exp or {}).get("boot_id") in (None, boot_id),
        f"실험 세션 {(exp or {}).get('state')}" + (f" · 실패 모듈 {(exp or {}).get('failed')}" if (exp or {}).get("failed") else ""))
    hz, age = od.get("hz_10s"), od.get("last_rx_age_s")
    add("fastlio", (mods.get("fastlio") or {}).get("state") == "RUNNING" and lio.get("status") == "OK" and (od.get("n") or 0) > 0
        and age is not None and age < POSE_AGE_MAX and (hz or 0) >= HZ_MIN,
        f"FAST-LIO {lio.get('status')} · {hz} Hz · 최신 자세 {age} s 전 (기준 ≥ {HZ_MIN:g} Hz, < {POSE_AGE_MAX} s)")
    add("rgbd", ((snap or {}).get("rgb") or {}).get("status") == "LIVE", f"RealSense {((snap or {}).get('rgb') or {}).get('status')}")
    rgb_ = (snap or {}).get("rgb") or {}
    add("depth", rgb_.get("depth_state") == "DEPTH_OK",
        f"RealSense 깊이 {rgb_.get('depth_state')}" + (f" · {rgb_.get('depth_why')} — 깊이가 없으면 E5 표지판 거리와 팬틸트 조준을 쓸 수 없습니다. FAST-LIO·E2는 계속됩니다." if rgb_.get("depth_state") != "DEPTH_OK" else ""))
    if (exp or {}).get("siglip", True):
        add("siglip", ((snap or {}).get("siglip") or {}).get("status") == "LIVE", f"SigLIP {((snap or {}).get('siglip') or {}).get('status')}")
    fresh = all(ages.get(k) is not None and ages[k] < STATE_AGE_MAX for k in ("e2", "e3", "e4"))
    add("e2_e3_e4", fresh and (e2.get("floors_total") or 0) > 0,
        f"E2/E3/E4 갱신 {ages.get('e2')}/{ages.get('e3')}/{ages.get('e4')} s · 지도 층 {e2.get('floors_total')}")
    add("safety", guard_last is not None and not (guard_last or {}).get("pending"),
        "안전 감시 동작" + (f" · 대기 {[p.get('condition') for p in guard_last.get('pending')]}" if (guard_last or {}).get("pending") else ""))
    ready = all(i["ok"] for i in items)
    return dict(ready=ready, items=items, e6=e6, missing=[i["key"] for i in items if not i["ok"]],
                event_id=f"{(exp or {}).get('exp_id')}:READY" if ready else None)


def latch(X, res, now):
    """Record the first READY of this experiment (time-to-ready from the start). -> True when it was latched now."""
    if not res.get("ready") or X.d.get("ready"):
        return False
    t0 = float(X.d.get("created_wall") or now)
    X.d["ready"] = dict(event_id=res["event_id"], wall=now, after_start_s=round(now - t0, 1), items=res["items"], e6=res.get("e6"))
    X.event("experiment_ready_to_move", event_id=res["event_id"], after_start_s=round(now - t0, 1))
    return True
