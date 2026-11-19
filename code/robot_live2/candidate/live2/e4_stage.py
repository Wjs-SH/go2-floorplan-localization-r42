"""e4_stage.py — LIVE2 staged localisation status (2026-10-30). READ-ONLY view over the existing E2 / E3 / E4 / E7 outputs; it never
changes a hypothesis, an E4 state or the FIX path. Purpose: show the operator (and the offline benchmark) where the search stands
when there is no FIX yet, with honest evidence counts.

Stages (first match wins):
  FIXED             E7 GLOBAL_CONFIRMED (existing E6/E8 or structure path)
  VERIFYING         (2026-10-31) READY_FOR_VERIFY while the E7 structure path is RUNNING on that leader (retro / live windows)
  READY_FOR_VERIFY  E4 CONFIDENT_GEOMETRY — the UNCHANGED E4 rule (every floor searched exactly, gap >= 0.15 score units, leader
                    >= 15 KF, >= 8 m path, recent cost <= 0.35); E7 may run. Nothing earlier is ever 'ready'.
  UNRESOLVED        the first exact pass covered every floor and E4 says GEOMETRY_INCONSISTENT / RECOVERY_REQUIRED, or stays
                    AMBIGUOUS for >= UNRESOLVED_S after the full search: structure alone cannot decide in this search range
  AMBIGUOUS         E4 AMBIGUOUS (distinct competitor within 0.05 score units, or >= 2 floors)
  PROVISIONAL       a leader with >= 8 keyframes that has stayed first AND without any E4 AMBIGUOUS for >= PROV_HOLD_S
                    (shown as '추정 중 / 미확정'); floors still unsearched are listed — a better candidate there is NOT excluded.
                    PROV_HOLD_S = 30 s is an engineering value chosen while looking at the 10-30 runs (field loc2 showed wrong
                    leaders for 1-10 s between AMBIGUOUS flickers); it is NOT validated on independent data and gates nothing.
  SEARCHING         otherwise
Evidence (reported, not used for any decision):
  leader_hold_s     wall time the current leader id has been first (from e3 history)
  independent_views keyframes of the leader taken >= INDEP_PATH_M of new odometry path apart (consecutive keyframes 0.5 m apart are
                    strongly correlated; only views separated by fresh path count) — a conservative count, not a probability
  unsearched_floors floors without an exact E2 result (coarse-only floors may still hide a better candidate)
"""
import math

INDEP_PATH_M = 3.0
PROV_HOLD_S = 30.0
UNRESOLVED_S = 120.0
STAGES = ("SEARCHING", "PROVISIONAL", "AMBIGUOUS", "READY_FOR_VERIFY", "VERIFYING", "FIXED", "UNRESOLVED")
KO = dict(SEARCHING="전역 탐색 중", PROVISIONAL="유력 후보 있음 — 추정 중 · 미확정", AMBIGUOUS="구조적으로 모호",
          READY_FOR_VERIFY="구조 근거 충분 — E7 검증 가능", VERIFYING="E7 LiDAR 검증 중 — 아직 FIX 아님", FIXED="E7 검증 성공 (FIX)", UNRESOLVED="이 탐색 범위에서 확인 불가")


def _path(rows, t0, t1):
    pts = [(r[2], r[3]) for r in rows or [] if t0 <= r[1] <= t1]
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def independent_views(kf_times, rows, sep_m=INDEP_PATH_M):
    """count keyframes whose odometry path since the previous COUNTED one is >= sep_m."""
    n, last = 0, None
    for t in sorted(kf_times):
        if last is None or _path(rows, last, t) >= sep_m:
            n += 1
            last = t
    return n


def stage(e2, e3, e4, e7, now, lead_since=None, full_since=None, rows=None, leader_kf_times=None, last_amb=None, sp=None):
    e2, e3, e4, e7, sp = e2 or {}, e3 or {}, e4 or {}, e7 or {}, sp or {}
    tot = e2.get("floors_total") or len(e2.get("floors") or []) or 0
    done = e2.get("floors_done") or []
    full = bool(tot) and len(done) >= tot
    unsearched = [f for f in (e2.get("floors") or []) if f not in done]
    H = sorted([h for h in e3.get("hypotheses") or [] if h.get("status") in ("ACTIVE", "SUSPENDED")], key=lambda h: h.get("score", 9))
    lead = H[0] if H else None
    st4 = e4.get("state")
    ev = dict(e4_state=st4, floors_exact=f"{len(done)}/{tot}", unsearched_floors=unsearched, leader=None if not lead else lead.get("id"),
              leader_floor=None if not lead else lead.get("floor"), leader_n_kf=None if not lead else lead.get("n_kf"),
              leader_hold_s=None if lead_since is None else round(now - lead_since, 1),
              independent_views=None if (leader_kf_times is None or rows is None) else independent_views(leader_kf_times, rows),
              gap=(e4.get("metrics") or {}).get("gap") or (e4.get("metrics") or {}).get("partial_gap"))
    if e7.get("global_decision") == "GLOBAL_CONFIRMED":
        s = "FIXED"
    elif st4 == "CONFIDENT_GEOMETRY":
        s = "VERIFYING" if sp.get("status") == "RUNNING" and sp.get("leader") == (lead or {}).get("id") else "READY_FOR_VERIFY"
    elif full and (st4 in ("GEOMETRY_INCONSISTENT", "RECOVERY_REQUIRED") or
                   (st4 == "AMBIGUOUS" and full_since is not None and now - full_since >= UNRESOLVED_S)):
        s = "UNRESOLVED"
    elif st4 == "AMBIGUOUS":
        s = "AMBIGUOUS"
    elif (lead is not None and (lead.get("n_kf") or 0) >= 8 and lead_since is not None and now - lead_since >= PROV_HOLD_S
          and (last_amb is None or now - last_amb >= PROV_HOLD_S)):
        s = "PROVISIONAL"
    else:
        s = "SEARCHING"
    if s == "SEARCHING" and lead is not None:
        ev["leader_note"] = "1위 후보 변동 중 / 근거 부족 — 표시만"
    why = {"PROVISIONAL": ("미탐색 층 " + str(unsearched) + " 에 더 좋은 후보가 있을 수 있음 — FIX 불가") if unsearched else
                          "전 층 탐색 완료, E4 확신 조건 미충족 (" + "; ".join((e4.get("reasons") or [])[:1]) + ")"}.get(s, "; ".join((e4.get("reasons") or [])[:1]))
    return dict(stage=s, ko=KO[s], why=why, evidence=ev, fix_allowed_by_this_view=False,
                note="표시 전용: FIX 는 기존 E4 CONFIDENT_GEOMETRY + E7 독립 재정합으로만 결정됨")
