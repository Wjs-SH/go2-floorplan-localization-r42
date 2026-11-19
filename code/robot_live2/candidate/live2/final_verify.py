#!/usr/bin/env python3
"""final_verify.py (R37, loc1): the last step of the structure path: FINAL_VERIFICATION_STOP_REQUESTED -> operator stops -> whole-path E7 with the newest LiDAR at rest -> FIX or back to driving.

The stop is requested only when e7_long says WOULD_CONFIRM: every floor had an exact search and repeat coarse searches, and the leader is first by >= 0.05 m in evaluations with new path between them. Never because of the time.
The FIX is written only when e7_policy.long_structure_policy() says ENABLED; otherwise the record says SHADOW_WOULD_CONFIRM. A failed verification does not sound again for COOLDOWN_S and NEW_PATH_M of new path.
Nothing here moves anything. GT is never an input."""
import json
import math
import os
import time
from pathlib import Path

STOP_WAIT_S, STABLE_S, COOLDOWN_S, NEW_PATH_M, MAX_ATTEMPTS = 45.0, 3.0, 120.0, 10.0, 3
GAP_M, MIN_INLIER = 0.05, 0.45
NOTICE_S = 240.0
REFRESH_WAIT_S = 120.0


def verdict(res, leader_floor, xw_veto, policy_enabled):
    """res = e7_long.evaluate() at rest. -> (status, why). Pure."""
    fl = (res or {}).get("floors") or []
    if len(fl) < 2:
        return "VERIFY_FAILED", "no comparison"
    b = fl[0]
    if b["floor"] != leader_floor:
        return "VERIFY_FAILED", f"{b['floor']}F first at rest, the request was for {leader_floor}F"
    gap = fl[1]["cost"] - b["cost"]
    if gap < GAP_M:
        return "VERIFY_FAILED", f"{fl[1]['floor']}F within {gap:.3f} m < {GAP_M} m"
    if not b.get("converged"):
        return "VERIFY_FAILED", "registration on its search bound"
    if b.get("scale_rel") is not None and abs(b["scale_rel"]) > 0.15:
        return "VERIFY_FAILED", f"map scale {b['scale_rel']:+.0%} from that floor's"
    if (b.get("pos_rival") or {}).get("gap") is not None and b["pos_rival"]["gap"] < GAP_M:
        return "VERIFY_FAILED", f"the same floor {b['pos_rival']['dist_m']} m away fits within {b['pos_rival']['gap']} m"
    if float(b.get("inlier") or 0) < MIN_INLIER:
        return "VERIFY_FAILED", f"inlier {b.get('inlier')} < {MIN_INLIER}"
    others = [x.get("crossing") for x in fl[1:] if x.get("crossing") is not None]
    if b.get("crossing") is not None and others and b["crossing"] - min(others) >= xw_veto:
        return "VERIFY_FAILED", f"wall crossing {b['crossing']} vs {min(others)}"
    why = f"{b['floor']}F first by {gap:.3f} m against {len(fl) - 1} floors, inlier {b.get('inlier')}, converged, at rest"
    return ("VERIFIED_FIX" if policy_enabled else "SHADOW_WOULD_CONFIRM"), why


class FinalVerify:
    def __init__(self, state, link, long_st, stop_fn, clock=time.time, busy_fn=None):
        self.state, self.link, self.long, self.stop_fn, self.clock = Path(state), link, long_st, stop_fn, clock
        self.busy_fn = busy_fn          # R39: E6 hardware at work: no new stop request; it stays due and goes out when E6 is free
        self.d = dict(kind="LIVE2_FINAL_VERIFY", status="WATCHING", attempts=0, event_id=None, why="waiting for a leader that other floors do not match")
        self.t_state, self.t_last, self.path_last, self.t_still = 0.0, -1e18, 0.0, None
        self.floor = None
        self._write()

    def _write(self):
        tmp = self.state / f".final_verify.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(dict(self.d, written_wall=self.clock()), ensure_ascii=False, default=str))
        os.replace(tmp, self.state / "final_verify.json")
        with open(self.state / "final_verify.jsonl", "a") as f:
            f.write(json.dumps(dict(self.d, wall=self.clock()), ensure_ascii=False, default=str) + "\n")

    def _set(self, status, why, **kw):
        self.d.update(status=status, why=why, **kw)
        self.t_state = self.clock()
        self._write()

    def _e2_dir(self):
        d = getattr(getattr(self.link, "tr", None), "e2_dir", None)
        return None if d is None else Path(d)

    def _request_refresh(self, now):
        d = self._e2_dir()
        if d is None or not d.exists():
            return False
        (d / "e2_refresh_now.json").write_text(json.dumps(dict(wall=now, why="FINAL_VERIFY")))
        return True

    def _refresh_done(self, t_stop):
        """One whole repeat search started after the stop."""
        try:
            e2 = json.loads((self._e2_dir() / "e2_state.json").read_text())
        except (OSError, ValueError, TypeError):
            return False
        R = e2.get("refresh") or {}
        if float(R.get("last_wall") or 0) < t_stop:
            return False
        n = R.get("n_passes")
        return len({c.get("floor") for c in e2.get("candidates_refresh") or [] if c.get("refresh_n") == n}) >= int(e2.get("floors_total") or 9)

    def policy_mode(self):
        import e7_policy as P7
        return P7.long_structure_policy()["mode"]

    def _path(self):
        rows = getattr(self.link.tr.store, "rows", None) or []
        return sum(math.hypot(b[2] - a[2], b[3] - a[3]) for a, b in zip(rows, rows[1:]))

    def tick(self, now=None):
        now = self.clock() if now is None else now
        s = self.d["status"]
        if s in ("VERIFIED_FIX",):
            return self.d
        if s in ("WATCHING", "VERIFY_FAILED", "STOP_EXPIRED", "SHADOW_WOULD_CONFIRM"):
            L = self.long.d
            if L.get("status") != "WOULD_CONFIRM" or self.d["attempts"] >= MAX_ATTEMPTS:
                return self.d
            if (self.busy_fn is not None and self.busy_fn()) or (getattr(self.link, "last", None) or {}).get("global_decision") == "GLOBAL_CONFIRMED":
                return self.d
            p = self._path()
            if now - self.t_last < COOLDOWN_S or (self.t_last > 0 and p - self.path_last < NEW_PATH_M):
                return self.d
            self.floor, self.t_last, self.path_last, self.t_still = L.get("leader_floor"), now, p, None
            self.d["attempts"] += 1
            self._set("FINAL_VERIFICATION_STOP_REQUESTED", "최종 위치 검증을 위해 정지해 주세요.", event_id=f"FINAL_{self.floor}F_{self.d['attempts']}_{int(now)}", leader_floor=self.floor, gap_m=L.get("next_gap_m"))
            return self.d
        if s == "FINAL_VERIFICATION_STOP_REQUESTED":
            st = self.stop_fn() or {}
            if st.get("stopped"):
                self.t_still = self.t_still or now
                if now - self.t_still >= STABLE_S:
                    # R40: every floor searched once more on the newest window before the deciding comparison. Live rivals can look worse than they are (X20261103_000434 +686 s: live gap 0.0505, the fair replay under 0.05 at every time): the FIX must not rest on a search that was not done
                    if self._request_refresh(now):
                        self._set("VERIFYING", "정지 상태: 모든 층을 최신 LiDAR로 다시 탐색한 뒤 E7 검증", t_stop=now, t_eval=None)
                    else:
                        self.long.request_now()
                        self._set("VERIFYING", "정지 상태 최신 LiDAR로 E7 검증 중", t_stop=now, t_eval=now)
            else:
                self.t_still = None
                if now - self.t_state > STOP_WAIT_S:
                    self._set("STOP_EXPIRED", f"{STOP_WAIT_S:.0f} s 안에 정지하지 않음 — 계속 이동. 근거가 다시 충분해지면 다시 요청합니다")
            return self.d
        if s == "VERIFYING":
            if self.d.get("t_eval") is None:
                if self._refresh_done(self.d["t_stop"]) or now - self.d["t_stop"] > REFRESH_WAIT_S:
                    self.d["refresh_complete"] = self._refresh_done(self.d["t_stop"])
                    self.d["t_eval"] = now
                    self.long.request_now()
                    self._write()
                return self.d
            r = self.long.latest_after(self.d["t_eval"])
            if r is None:
                if now - self.t_state > 150.0:
                    self._set("VERIFY_FAILED", "E7 결과 시간 초과 — 계속 이동")
                return self.d
            import e7_rematch as E7
            st, why = verdict(r, self.floor, E7.XW_VETO, self.policy_mode() == "ENABLED")
            if st == "VERIFIED_FIX" and self.d.get("refresh_complete") is False:
                st, why = "VERIFY_FAILED", "the last all-floor search did not finish"
            if st == "VERIFIED_FIX":
                ok, wf = self.link.write_structure_long_fix(r, why)
                if not ok:
                    st, why = "VERIFY_FAILED", wf
            self._set(st, why, result=[{k: x.get(k) for k in ("floor", "cost", "inlier", "converged", "crossing", "stage")} for x in r["floors"][:5]], path_m=r.get("path_m"), n_frames=r.get("n_frames"))
        return self.d
