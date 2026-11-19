#!/usr/bin/env python3
"""status_view.py (R34, 2026-11-01): what the tablet says about E2 / depth / E4 / E7 / anchored candidates / observations, from the backend's own state files.
Read only, never decides anything. 11-01 lesson: E2 had crashed five times and the page showed 0/9 with no reason, depth was frozen with no warning, and VLM that never ran looked like a pass."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sci_sensor_bridge"))
import depth_freshness as DF  # noqa: E402

VLM_TEXT = {"NOT_NEEDED": "VLM 불필요 (OCR 충분)", "UNAVAILABLE": "VLM 실행 안 됨 (서버 없음) — 확인된 것이 아닙니다", "NOT_ADMITTED_RESOURCE": "VLM 실행 안 됨 (자원 부족, OCR 우선)",
            "NOT_ADMITTED": "VLM 실행 안 됨 (자원 부족, OCR 우선)", "ERROR": "VLM 오류", "ASKED": "VLM 실행됨"}


def _read(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _tail_error(logs, module="e2", n=1500):
    try:
        f = max(Path(logs).glob(f"{module}_*.out"), key=lambda q: q.stat().st_mtime)
        with open(f, "rb") as h:
            h.seek(0, 2)
            h.seek(max(0, h.tell() - n))
            lines = [x for x in h.read().decode("utf-8", "replace").splitlines() if x.strip()]
    except (OSError, ValueError):
        return None
    err = [x for x in lines if "Error" in x or "Exception" in x]
    return err[-1][:200] if err else None


def e2_chip(X, st, logs, now):
    e2 = _read(st / "e2" / "e2_state.json") or {}
    tot, done = e2.get("floors_total"), len(e2.get("floors_done") or [])
    prog = f"E2 {done}/{tot}" if tot else "E2 시작 전"
    tl = X.get("timeline") or []
    last_end = next((e for e in reversed(tl) if e.get("module") == "e2" and e.get("kind") == "module_ended_unexpectedly"), None)
    last_start = next((e for e in reversed(tl) if e.get("module") == "e2" and e.get("kind") == "module_started"), None)
    restore = e2.get("restore") or {}
    extra = ""
    if restore.get("status") == "RESTORED":
        extra = f" · 재시작 {e2.get('restarts')}회 후 복구 ({done}층 유지)"
    elif e2.get("restarts") or (restore.get("status") == "FRESH" and "identity" in str(restore.get("why"))):
        extra = f" · 복구 안 함: {restore.get('why')}"
    if last_end and (not last_start or float(last_start.get("wall", 0)) < float(last_end.get("wall", 0))):
        return dict(key="e2", level="bad", code="E2_CRASHED", text=f"{prog} · E2 CRASHED ({last_end.get('end_reason')})", detail=_tail_error(logs),
                    next="[E2 초기 후보 계산]으로 다시 시작하세요. 완료한 층과 후보는 복구됩니다.")
    regional = (e2.get("regional") or [])[-1:] or [None]
    r = regional[0]
    if r and r.get("status") == "RUNNING":
        return dict(key="e2", level="warn", code="E2_REGIONAL_FIT", text=f"{prog} · 표지판 주변 지역 탐색 중 ({r.get('floor')}F, 반경 {r.get('radius_px_at_fit')} px)" + extra, detail=None, next=None)
    held = str(e2.get("held") or "")
    if held.startswith("HELD_FASTLIO"):
        return dict(key="e2", level="warn", code="HELD_FASTLIO", text=f"{prog} · FAST-LIO 보호를 위해 E2 대기", detail=held[:160], next="계속 이동하세요. 자동으로 재개됩니다.")
    if held.startswith("HELD_THERMAL"):
        return dict(key="e2", level="warn", code="HELD_THERMAL", text=f"{prog} · 온도 때문에 E2 대기", detail=held[:160], next="자동으로 재개됩니다.")
    if held:
        return dict(key="e2", level="warn", code="HELD", text=f"{prog} · E2 대기", detail=held[:160], next=None)
    if r and r.get("status") == "RETRY_WAIT":
        return dict(key="e2", level="warn", code="E2_REGIONAL_WAIT", text=f"{prog} · 지역 탐색 대기: {str(r.get('why_failed'))[:90]}" + extra, detail=None, next="계속 이동하면 다시 시도합니다.")
    if r and r.get("status") == "FAILED" and "NEEDS_NEW_OBSERVATION" in str(r.get("why_failed")):
        return dict(key="e2", level="warn", code="E2_REGIONAL_NEW_OBS", text=f"{prog} · 지역 탐색 종료: 새 표지판 관측 필요" + extra, detail=str(r.get("why_failed"))[:160], next="다음 표지판을 촬영하세요.")
    if e2.get("status") == "RUNNING":
        return dict(key="e2", level="neu", code="E2_SEARCHING", text=f"{prog} · 층 탐색 중" + extra, detail=None, next=None)
    if tot and done + len(e2.get("floors_failed") or {}) >= tot:
        return dict(key="e2", level="ok", code="E2_DONE", text=f"{prog} · 탐색 완료" + extra, detail=None, next=None)
    return dict(key="e2", level="neu", code=str(e2.get("status") or "E2_WAITING"), text=f"{prog} · {e2.get('status') or '대기'}" + extra, detail=e2.get("wait_note") if isinstance(e2.get("wait_note"), str) else None, next=None)


def depth_chip(hub_index, now):
    if not hub_index or now - float(hub_index.get("wall", 0)) > 2.0:
        return dict(key="depth", level="bad", code="SENSOR_HUB_STALE", text="RealSense 상태 STALE (센서 허브 갱신 없음)", detail=None, next="FAST-LIO·E2는 계속됩니다.")
    d = DF.hub_depth_state(hub_index)
    if d["state"] == DF.DEPTH_OK:
        return dict(key="depth", level="ok", code=DF.DEPTH_OK, text=f"깊이 정상 (컬러와 {d['gap_s']} s 차이)", detail=None, next=None)
    return dict(key="depth", level="bad", code=d["state"], text=f"{d['state']} — 표지판 거리·조준 사용 불가 (DEPTH_UNAVAILABLE)", detail=d.get("why"),
                next="FAST-LIO·E2는 계속됩니다. 깊이를 끄는 다른 RealSense 구독자가 있는지 확인하세요.")


def e4_chip(st):
    e4 = _read(st / "e4_state.json") or {}
    m = e4.get("metrics") or {}
    if m.get("partial_competitor_floor") is not None:
        return dict(key="e4", level="warn", code=str(e4.get("state")), text=f"E4 {e4.get('state')} · {m.get('partial_best_floor')}F vs 경쟁층 {m.get('partial_competitor_floor')}F gap {m.get('partial_gap')}", detail=None, next=None)
    if m.get("competitor_floor") is not None:
        return dict(key="e4", level="warn", code=str(e4.get("state")), text=f"E4 {e4.get('state')} · {m.get('best_floor')}F vs 경쟁층 {m.get('competitor_floor')}F gap {m.get('gap')}", detail=None, next=None)
    return dict(key="e4", level="neu", code=str(e4.get("state") or "NOT_RUN"), text=f"E4 {e4.get('state') or 'NOT_RUN'}", detail=None, next=None)


def e7_chip(st, t0):
    e7 = _read(st / "e7_state.json") or {}
    sp = _read(st / "structure_path.json") or {}
    es = _read(st / "e7_shadow_early.json") or {}
    if e7.get("global_decision") == "GLOBAL_CONFIRMED" and float(e7.get("written_wall") or 0) >= t0:
        return dict(key="e7", level="ok", code="GLOBAL_CONFIRMED", text="E7 GLOBAL_CONFIRMED", detail=None, next=None)
    if sp.get("status"):
        return dict(key="e7", level="neu", code=f"E7_{sp['status']}", text=f"E7 구조 경로 {sp['status']} (mode {sp.get('mode')}) — 최종 확정 아님", detail=sp.get("why"), next=None)
    if es.get("status"):
        last = es.get("last") or {}
        return dict(key="e7", level="neu", code=f"E7_SHADOW_EARLY_{es['status']}", text=f"E7 SHADOW 사전 시험 {es['status']} ({es.get('n_pass', 0)}창) — 기록 전용, 최종 확정 아님",
                    detail=None if not last else f"best {last.get('best')} · {last.get('final_decision')} · gap {last.get('gap_m')} · 미탐색 층 {last.get('unsearched_floors')}", next=None)
    if e7.get("global_decision") and float(e7.get("written_wall") or 0) >= t0:
        cls = e7.get("structural_class")
        txt = {"STRUCTURALLY_AMBIGUOUS": "구조로 구별되지 않음 — 다른 표지판으로 층을 구분해야 합니다", "STRUCTURAL_CONTRADICTION": "LiDAR 구조와 모순 — 이 후보는 맞지 않습니다", "STRUCTURALLY_COMPATIBLE": "구조와 일치 — 의미 조건 대기"}.get(cls)
        return dict(key="e7", level="bad" if cls == "STRUCTURAL_CONTRADICTION" else "warn", code=f"E7_{e7['global_decision']}", text=f"E7 {e7['global_decision']}" + (f" · {txt}" if txt else "") + " — 최종 확정 아님", detail=str(e7.get("why"))[:200], next="다른 호실 번호 표지판을 관측하세요." if cls == "STRUCTURALLY_AMBIGUOUS" else None)
    return dict(key="e7", level="neu", code="E7_NOT_RUN", text="E7 NOT_RUN", detail=None, next=None)


def anchor_chip(st):
    a = _read(st / "e2" / "semantic_anchor.json") or {}
    runs = a.get("runs") or []
    if not runs:
        return None
    r = runs[-1]
    if r.get("status") == "ANCHORED_PROVISIONAL":
        top = (r.get("top") or [{}])[0]
        return dict(key="anchor", level="warn", code="SEMANTIC_ANCHORED", text=f"SEMANTIC_ANCHORED 후보 · {r.get('room')} ({r.get('floor')}F) — 잠정, 확정 불가",
                    detail=f"LiDAR cost {top.get('lidar_cost')} · 다음 후보 대비 {top.get('separation_ratio')}배 · {a.get('model_status')}", next="다른 표지판 하나를 더 관측하세요.")
    return dict(key="anchor", level="neu", code=str(r.get("status")), text=f"장소 고정 후보 {r.get('status')}", detail=r.get("why"), next=None)


def observations(st):
    p = _read(st / "e8_pair_state.json") or {}
    ev = p.get("evaluation") or {}
    per = {r.get("observation_id"): r for r in ev.get("per_observation") or []}
    src = p.get("observations") or (p.get("last_closed") or {}).get("observations") or []
    out = []
    for i, o in enumerate(src[-8:], 1):
        r = per.get(o.get("observation_id")) or {}
        lines = o.get("ocr_lines") or []
        vm = ((o.get("reading") or {}).get("vlm") or {}).get("state")
        vs = vlm_state(o.get("reading"), _read(st / "vlm_state.json"))
        cat = (r.get("category") or {}).get("category")
        out.append(dict(n=i, observation_id=o.get("observation_id"), reading=lines[0] if lines else None, kind=r.get("reading_kind"), usable=r.get("usable"), why_not=r.get("why_not"),
                        category=cat, vlm=f"{vs} · {VLM_TEXT.get(vm, vm)}" if vm else vs, vlm_state=vs, vlm_room=(o.get("vlm") or {}).get("room_number"), roi_cut=bool(((o.get("reading") or {}).get("ocr_roi_source")) and r.get("roi_cut_warning"))))
    labels = []
    for r in ev.get("per_observation") or []:
        if r.get("usable") and r.get("labels"):
            labels.append(str(r["labels"][0][0]).replace("cat:restroom", "TOILETS").replace("cat:", ""))
    comp = None
    if len(labels) >= 2 and ev.get("verdict") not in (None, "PROVISIONAL"):
        comp = dict(key="pair", level="ok" if ev.get("verdict") == "REGION_CANDIDATE_READY" else "warn", code=str(ev.get("verdict")), text="+".join(labels[:4]) + f" 비교 중 → {ev.get('verdict')}", detail=str(ev.get("why"))[:200], next=None)
    return out, comp, ev.get("n_valid_independent"), len(src), p.get("state")


def aim_chip(st):
    last = None
    try:
        for ln in open(st / "e6_observations.jsonl"):
            try:
                o = json.loads(ln)
            except ValueError:
                continue
            if o.get("state") == "WAIT_STOP" and o.get("pre_stop_aim"):
                last = o
    except OSError:
        return None
    if not last:
        return None
    pa = last["pre_stop_aim"]
    lvl = "ok" if pa.get("status") == "AIM_OK_AT_TYPICAL_STOP" else "warn"
    return dict(key="aim", level=lvl, code=str(pa.get("status")), text=f"정지 전 조준 예상: {pa.get('guidance') or pa.get('why')}", detail=None, next=None)


SHADOW_TEXT = dict(WOULD_FIX="조건 충족", WOULD_AMBIGUOUS="조건 부족", WOULD_REJECT="거부", SHADOW_ERROR="기록 오류")


def shadow_chip(st):
    s = _read(st / "e7_semantic_shadow.json")
    if not s:
        return None
    return dict(key="shadow", level="neu", code=f"SHADOW_{s.get('verdict')}", text=f"구조·의미 통합 SHADOW {s.get('verdict')} ({SHADOW_TEXT.get(s.get('verdict'), '')}) — 기록 전용, 최종 확정 아님", detail="차단: " + ", ".join(map(str, (s.get("rejected_by") or []) + (s.get("blocked_by") or []))) if (s.get("rejected_by") or s.get("blocked_by")) else None, next=None)


LONG_TEXT = dict(LEADER_UNRESOLVED="유력 층, 다른 층 정밀 탐색 미완료", AMBIGUOUS_FLOORS="구조로 구분되지 않는 층 있음", WOULD_CONFIRM="단독 선두 (SHADOW, FIX 아님)")


def long_chip(st):
    d = _read(st / "e7_long.json")
    if not d or d.get("leader_floor") is None:
        return None
    close = d.get("close_floors") or []
    return dict(key="e7_long", level="warn" if d.get("status") == "AMBIGUOUS_FLOORS" else "neu", code=f"E7_LONG_{d.get('status')}",
                text=f"전 경로 구조 비교: {d['leader_floor']}F 선두 · {LONG_TEXT.get(d.get('status'), d.get('status'))}" + (f" · {close}F 차이 < 0.05 m" if close else ""),
                detail=f"경로 {d.get('path_m')} m · 차이 {d.get('next_gap_m')} m · 정밀 {d.get('floors_exact')}/{d.get('floors_total')}층 · 기록 전용", next=None)


def vlm_chip(st):
    d = _read(st / "vlm_state.json")
    if not d or (d.get("state") == "OFF" and not d.get("n_asked")):
        return None
    txt = dict(LOADING="VLM 서버 시작 중", ASKING="VLM 판독 중", READY="VLM 준비됨", FAILED="VLM 실행 실패", OFF="VLM 꺼짐").get(d.get("state"), d.get("state"))
    return dict(key="vlm", level="bad" if d.get("state") == "FAILED" else "neu", code=f"VLM_{d.get('state')}", text=f"{txt} · 판독 {d.get('n_asked')}회" + (f" · 최근 {d.get('last_s')} s" if d.get("last_s") else ""), detail=d.get("why"), next=None)


def auto_chip(st):
    """R39: what the robot decided by itself and why."""
    d = _read(st / "auto_state.json")
    if not d:
        return None
    a = d.get("assessment") or {}
    cons = "하드웨어 사용 동의 필요: 이번 실험은 팬틸트 동의 없이 시작해 표지판을 촬영할 수 없습니다. 구조 기반 추정은 계속합니다." if d.get("needs_pantilt_consent") else None
    return dict(key="auto", level="warn" if cons else "neu", code=str(d.get("state")), text=f"자동 판단: {d.get('state_ko')}", detail=a.get("why"), next=cons)


VLM_STATE = dict(QUEUED="QUEUED", LOADING="LOADING", ASKING="RUNNING", RUNNING="RUNNING", ASKED="DONE", NOT_ADMITTED_RESOURCE="SKIPPED_RESOURCE", ERROR="FAILED", TIMEOUT="TIMEOUT", UNAVAILABLE="NOT_REQUESTED")


def vlm_state(reading, server):
    """One observation's VLM status from what really happened: a state only appears if the bridge recorded it; no record means it was not requested."""
    v = (reading or {}).get("vlm") or {}
    s = v.get("state")
    if s is None:
        return "NOT_REQUESTED"
    if s == "ERROR" and "imeout" in str(v.get("error")):
        return "TIMEOUT"
    if s == "QUEUED" and (server or {}).get("state") == "LOADING":
        return "LOADING"
    return VLM_STATE.get(s, s)


def kci_confirm_chip(st):
    """2026-11-15 esy: structure-only confirmation that e3_tracker records (e3_state kci_confirm). Display only, decides nothing.
    Page levels are ok / warn / bad / (anything else = neutral). Shown from 5 m of sustained lead; 'confirmed' = the leader kept every other hypothesis suspended for rule.streak_m of walking."""
    try:
        c = (_read(st / "e3_state.json") or {}).get("kci_confirm") or {}
        need = float((c.get("rule") or {}).get("streak_m") or 0)
        m = float(c.get("streak_m") or 0)
        f, g = c.get("confirmed"), c.get("gather_floor")
        if f is not None:
            now = f"지금 {g}층 유지 {m:.0f} m" if g is not None else "지금은 유지가 끊김"
            return dict(key="kci_confirm", level="ok", code="KCI_STRUCT_CONFIRMED", text=f"구조 확정: {f}층",
                        detail=f"1위가 나머지를 따돌린 채 {need:.0f} m 걸음 · {now} · 구조만 본 판정(번호판과 다르면 번호판 우선)", next=None)
        if g is not None and m >= 5.0 and need > 0:
            return dict(key="kci_confirm", level="neu", code="KCI_STRUCT_HOLDING", text=f"구조: {g}층 유지 {m:.0f} / {need:.0f} m", detail="이 거리를 채우면 구조 확정으로 표시", next=None)
    except Exception:
        pass
    return None


def e3_chip(st):
    e3 = _read(st / "e3_state.json") or {}
    lag = e3.get("leader_kf_lag_s")
    if lag is None or lag < 20:
        return None
    return dict(key="e3", level="warn", code="E3_LAG", text=f"E3 지연: 선두 후보 키프레임 {lag} s 늦음 · 대기 {e3.get('kf_pending')}개", detail=None, next=None)


def build(X, state_dir, logs_dir, hub_index, now=None):
    now = time.time() if now is None else now
    st = Path(state_dir)
    t0 = float(X.get("created_wall") or 0)
    chips = [e2_chip(X, st, logs_dir, now), depth_chip(hub_index, now), e4_chip(st), e7_chip(st, t0)]
    k = kci_confirm_chip(st)                # 2026-11-15 esy: structure confirmation chip (display only), first line of the card so it is seen
    if k:
        chips.insert(0, k)
    for c in (auto_chip(st), long_chip(st), vlm_chip(st)):
        if c:
            chips.append(c)
    a = anchor_chip(st)
    if a:
        chips.append(a)
    obs, comp, nv, na, pst = observations(st)
    if comp:
        chips.append(comp)
    aim = aim_chip(st)
    if aim:
        chips.append(aim)
    sh = shadow_chip(st)
    if sh:
        chips.append(sh)
    e3 = e3_chip(st)
    if e3:
        chips.append(e3)
    return dict(chips=chips, observations=obs, n_valid_independent=nv, n_attempts=na, pair_state=pst, gpu_e2="NOT_AUTHORIZED")
