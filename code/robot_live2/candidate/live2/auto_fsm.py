#!/usr/bin/env python3
"""auto_fsm.py (R39): the single automatic initial-localisation state machine. The robot chooses its method from what its sensors show; there is no loc1 / loc2 input.

Inputs: E2 state, E3 hypotheses, E4 state, e7_long, E8 pair, final_verify, E6 hardware state, odometry path. Never GT, never an operator's zone label.
Outputs: <state>/auto_state.json (read by visual_stop for the stop gate, by e2_worker through e2_budget.json, by the page) and <state>/auto_transitions.jsonl (every change, with its reason).

Structure class (what E4's evidence says about this place):
  UNDETERMINED     too little evidence: no stop request, structure keeps accumulating
  DISCRIMINATIVE   one floor stands clear of every other floor that has LiDAR evidence: no stop request, E7 structure verification is awaited
  AMBIGUOUS        other floors explain the LiDAR about as well, or structure could not confirm: E5 stop requests -> E6 -> OCR / VLM
It decides only HOW to collect evidence. It never makes a FIX: the FIX stays with final_verify (structure) and e6e7_link (semantic). An early call therefore cannot weaken a FIX rule.

Hysteresis, written with what each value was chosen from (engineering values from the field runs of 11-02 and 11-03; not validated on independent data):
  T_AMB_EARLY_S 20  other-floor rivals within GAP_AMB before all floors had a coarse search
  T_AMB_S       10  the same after all floors had a coarse search
  T_DISC_S      30  + NEW_PATH_M 5 m: clear, believable leader held this long over new ground before stops are withheld
  T_DECIDE_S     0  R40 (was 60): all floors have a coarse search but structure is not clearly enough: use signs now. No evidence for structure = signs allowed
  T_VERIFY_S   150  DISCRIMINATIVE without a structure confirmation for this long -> AMBIGUOUS
After that escalation only e7_long WOULD_CONFIRM (fair rivals) leads back to DISCRIMINATIVE.
"""
import json
import os
import time
from pathlib import Path

GAP_AMB, GAP_DISC, KF_MIN = 0.05, 0.15, 8
T_AMB_EARLY_S, T_AMB_S, T_DISC_S, T_DECIDE_S, T_VERIFY_S, NEW_PATH_M = 20.0, 10.0, 30.0, 0.0, 150.0, 5.0
RECENT_OK, DISC_KF, SCALE_TOL = 0.35, 15, 0.15
_PRIOR = {}


def scale_prior(floor):
    """px/m of a floor: semantic_scale_v1 (estimated from the stacked core), else the semantic map. None if unknown. Never modified here."""
    if not _PRIOR:
        here = Path(__file__).resolve().parent
        try:
            for f, v in json.loads((here / "semantic_scale_v1.json").read_text())["floors"].items():
                _PRIOR[int(f)] = float(v["px_per_m"])
        except (OSError, ValueError, KeyError):
            pass
        for f in range(1, 10):
            if f not in _PRIOR:
                try:
                    _PRIOR[f] = 1.0 / float(json.loads((Path.home() / "sci_ws/maps/kci_handoff" / f"{f}F" / f"{f}F_semantic_map.json").read_text())["map_meta"]["resolution_m_per_px"])
                except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
                    pass
    return _PRIOR.get(int(floor))


def pose_quality(h):
    """Is this leader's PLACE believable, whatever its rank? From the hypothesis alone. X20261103_003729 H106_F4: 33.1 px/m on a 24 px/m floor, recent merit 0.96 m, 29 m off. -> dict(low, reasons)."""
    why = []
    if h is None:
        return dict(low=True, reasons=["후보 없음"])
    p = scale_prior(h["floor"]) if h.get("ppm") else None
    if p and abs(float(h["ppm"]) / p - 1.0) > SCALE_TOL:
        why.append(f"지도 축척 {float(h['ppm']) / p - 1.0:+.0%}")
    if h.get("recent") is not None and float(h["recent"]) > RECENT_OK:
        why.append(f"최근 정합 {float(h['recent']):.2f} m")
    if int(h.get("n_kf") or 0) < KF_MIN:
        why.append(f"키프레임 {h.get('n_kf')}개")
    return dict(low=bool(why), reasons=why, id=h.get("id"), floor=h.get("floor"))
CLASSES = ("UNDETERMINED", "DISCRIMINATIVE", "AMBIGUOUS")
STATES = ("AUTO_INITIALIZING", "STRUCTURE_SEARCHING", "STRUCTURE_DISCRIMINATIVE", "E7_STRUCTURE_VERIFY", "STRUCTURE_AMBIGUOUS", "SEMANTIC_ACQUISITION_ENABLED", "SEMANTIC_CANDIDATE", "E7_SEMANTIC_VERIFY", "GLOBAL_FIX")
HW_BUSY = ("WAIT_BODY_SETTLE", "PANTILT_AIMING", "WAIT_MANUAL_FOCUS", "CAPTURE_REVIEW", "REPOSITION_NEEDED", "HOME_RETURNING")
KO = dict(AUTO_INITIALIZING="센서와 지도 준비 중", STRUCTURE_SEARCHING="구조 분석 중 — 판단할 관측을 모으는 중", STRUCTURE_DISCRIMINATIVE="구조로 구별되는 위치 — 구조 검증 대기, 표지판 정지 안 함",
          E7_STRUCTURE_VERIFY="구조 기반 E7 최종 검증 중", STRUCTURE_AMBIGUOUS="구조만으로 구별 안 됨 — 표지판 탐색", SEMANTIC_ACQUISITION_ENABLED="구조만으로 구별 안 됨 — 표지판 정지·촬영 사용",
          SEMANTIC_CANDIDATE="의미정보 기반 후보 있음 — 다음 표지판 또는 구조 근거 필요", E7_SEMANTIC_VERIFY="의미정보 기반 E7 최종 검증 중", GLOBAL_FIX="최종 위치 확정")


def phase_to_hw(manual_phase, settle_state=None):
    """E6 manual / settle records -> one hardware state. The names the page and the FIX presentation use."""
    if settle_state in ("WAITING_FOR_SETTLE_APPROVAL", "WAITING_FOR_STOP"):
        return "WAIT_BODY_SETTLE"
    m = dict(WAIT_BODY_SETTLE="WAIT_BODY_SETTLE", AIMING="PANTILT_AIMING", WAIT_AIM_SETTLE="PANTILT_AIMING", AIM_READY="PANTILT_AIMING",
             WAIT_MANUAL_FOCUS="WAIT_MANUAL_FOCUS", CAPTURE_REQUESTED="WAIT_MANUAL_FOCUS", CAPTURING="WAIT_MANUAL_FOCUS", CAPTURE_SAVED="CAPTURE_REVIEW", CAPTURE_REVIEW="CAPTURE_REVIEW", CAPTURE_FAILED="CAPTURE_REVIEW",
             REPOSITION_NEEDED="REPOSITION_NEEDED", RETURNING_HOME="HOME_RETURNING", HOME_CONFIRMED="MOVE_ALLOWED")
    return m.get(manual_phase, "E6_IDLE")


def assess(hyps, e2, e4_state=None, long_d=None):
    """One tick's evidence. Pure. hyps = E3 hypotheses (id, floor, score, n_kf, status). -> dict(raw, ...). raw: NO_EVIDENCE | AMB | DISC | MIDDLE."""
    e2 = e2 or {}
    total = int(e2.get("floors_total") or 9)
    coarse = len((e2.get("coarse") or {}).get("floors_done") or [])
    exact = len(e2.get("floors_done") or [])
    alive = [h for h in hyps or [] if h.get("status") in ("ACTIVE", "SUSPENDED") and h.get("score") is not None]
    ev = [h for h in alive if int(h.get("n_kf") or 0) >= KF_MIN]
    out = dict(floors_total=total, coarse_done=coarse, exact_done=exact, coarse_complete=coarse >= total or exact >= total, n_alive=len(alive), n_active=sum(h["status"] == "ACTIVE" for h in alive), n_evidence=len(ev))
    if not ev:
        return dict(out, raw="NO_EVIDENCE", leader_floor=None, rival_floor=None, gap=None, near_floors=[], floors=[])
    act = [h for h in ev if h["status"] == "ACTIVE"] or ev
    b = min(act, key=lambda h: h["score"])
    per = {}
    for h in ev:
        if h["floor"] not in per or h["score"] < per[h["floor"]]["score"]:
            per[h["floor"]] = h
    rivals = sorted((h for f, h in per.items() if f != b["floor"]), key=lambda h: h["score"])
    gap = None if not rivals else round(rivals[0]["score"] - b["score"], 4)
    near = sorted(f for f, h in per.items() if h["score"] - b["score"] < GAP_AMB)
    fl = [dict(floor=f, score=round(h["score"], 3), id=h["id"], status=h["status"], n_kf=h["n_kf"]) for f, h in sorted(per.items(), key=lambda kv: kv[1]["score"])][:9]
    out.update(leader=b["id"], leader_floor=b["floor"], leader_score=round(b["score"], 4), rival_floor=rivals[0]["floor"] if rivals else None, gap=gap, near_floors=near, floors=fl)
    out["pose_quality"] = pose_quality(b)
    L = long_d or {}
    if L.get("status") == "AMBIGUOUS_FLOORS" or (L.get("close_floors") and any(x.get("converged") and x.get("floor") in L["close_floors"] for x in L.get("floors") or [])):
        return dict(out, raw="AMB", why=f"E7 전 경로 비교: {L.get('leader_floor')}F와 {L.get('close_floors')}F가 구별되지 않음")
    if gap is not None and gap < GAP_AMB:
        return dict(out, raw="AMB", why=f"{b['floor']}F와 {rivals[0]['floor']}F 점수 차 {gap} m < {GAP_AMB} m")
    if len(near) >= 2:
        return dict(out, raw="AMB", why=f"{near}층이 같은 수준으로 LiDAR를 설명")
    # DISC needs a compared rival floor: a floor without LiDAR evidence was never compared, so 'alone' does not count before every floor had a search and some rival has evidence
    # R40: E3 scores rivals over different keyframes, and not all floors are searched yet: 'gap 1.04' (R39 replay) was 0.017 in the fair comparison; X20261103_003729 was 'discriminative' on 9F, then 7F, then 4F. So the leader must also be believable.
    if out["coarse_complete"] and gap is not None and gap >= GAP_DISC and not out["pose_quality"]["low"] and int(b.get("n_kf") or 0) >= DISC_KF and int(rivals[0].get("n_kf") or 0) >= DISC_KF:
        return dict(out, raw="DISC", why=f"{b['floor']}F가 다음 층 {rivals[0]['floor']}F보다 {gap} m 낮음 (후보 {len(ev)}개 비교)")
    return dict(out, raw="MIDDLE", why=f"{b['floor']}F 선두, 다음 층과 차이 {gap}")


class AutoFSM:
    def __init__(self, state_dir=None, clock=time.time):
        self.state = None if state_dir is None else Path(state_dir)
        self.clock = clock
        self.cls, self.since, self.escalated = "UNDETERMINED", None, False
        self.st, self.t_amb, self.t_disc, self.t_coarse, self.path_disc = "AUTO_INITIALIZING", None, None, None, 0.0
        self.t_disc_entered, self.t_started = None, None
        self.d = dict(kind="LIVE2_AUTO_STATE", state=self.st, structure_class=self.cls)
        self.hist, self.t_hist = [], 0.0
        self.n_transitions = 0

    def _class(self, a, now, path_m, long_d):
        """Hysteresis over assess(). -> (class, why)."""
        raw = a["raw"]
        L = long_d or {}
        self.t_amb = (now if self.t_amb is None else self.t_amb) if raw == "AMB" else None
        if raw == "DISC":
            if self.t_disc is None:
                self.t_disc, self.path_disc = now, path_m
        else:
            self.t_disc = None
        if a["coarse_complete"] and self.t_coarse is None:
            self.t_coarse = now
        if L.get("status") == "WOULD_CONFIRM":
            self.escalated = False
            return "DISCRIMINATIVE", f"E7 전 경로 비교: {L.get('leader_floor')}F가 공정한 경쟁 층보다 낮음"
        if self.cls == "DISCRIMINATIVE":
            if self.t_amb is not None and now - self.t_amb >= T_AMB_S:
                return "AMBIGUOUS", a.get("why")
            if self.t_disc_entered is not None and now - self.t_disc_entered >= T_VERIFY_S:
                self.escalated = True
                return "AMBIGUOUS", f"구조 우세가 {T_VERIFY_S:.0f} s 동안 E7 전 경로 비교로 확인되지 않음 — 의미정보 사용"
            return "DISCRIMINATIVE", None
        hold_amb = T_AMB_S if a["coarse_complete"] else T_AMB_EARLY_S
        if self.cls == "UNDETERMINED" and self.t_amb is not None and now - self.t_amb >= hold_amb:
            return "AMBIGUOUS", a.get("why")
        if not self.escalated and self.t_disc is not None and now - self.t_disc >= T_DISC_S and path_m - self.path_disc >= NEW_PATH_M:
            return "DISCRIMINATIVE", a.get("why")
        if self.cls == "UNDETERMINED" and self.t_coarse is not None and now - self.t_coarse >= T_DECIDE_S:
            return "AMBIGUOUS", "전 층 개략 탐색까지 구조 우세 없음 — 의미정보 사용"
        return self.cls, None

    def tick(self, e2=None, e3=None, e4=None, long_d=None, path_m=0.0, global_decision=None, pair_status=None, semantic_status=None, final_status=None, hw="E6_IDLE", e6_alive=True, vlm_busy=False, vlm_resident=False, vlm_pending=False, now=None):
        now = self.clock() if now is None else now
        self.t_started = self.t_started if self.t_started is not None else now
        a = assess((e3 or {}).get("hypotheses") or [], e2, (e4 or {}).get("state"), long_d)
        cls, why_c = self._class(a, now, path_m, long_d)
        if cls != self.cls:
            self._log(now, "CLASS", self.cls, cls, why_c, a, path_m)
            self.cls, self.since = cls, now
            self.t_disc_entered = now if cls == "DISCRIMINATIVE" else None
        fixed = global_decision == "GLOBAL_CONFIRMED"
        if fixed:
            st = "GLOBAL_FIX"
        elif final_status in ("FINAL_VERIFICATION_STOP_REQUESTED", "VERIFYING"):
            st = "E7_STRUCTURE_VERIFY"
        elif pair_status == "E7_VERIFYING":
            st = "E7_SEMANTIC_VERIFY"
        elif semantic_status and semantic_status not in ("NO_READING",) and cls != "DISCRIMINATIVE":
            st = "SEMANTIC_CANDIDATE"
        elif cls == "DISCRIMINATIVE":
            st = "STRUCTURE_DISCRIMINATIVE"
        elif cls == "AMBIGUOUS":
            st = "SEMANTIC_ACQUISITION_ENABLED" if e6_alive else "STRUCTURE_AMBIGUOUS"
        elif a["raw"] == "NO_EVIDENCE" and not a["n_alive"]:
            st = "AUTO_INITIALIZING"
        else:
            st = "STRUCTURE_SEARCHING"
        if st != self.st:
            self._log(now, "STATE", self.st, st, why_c or a.get("why"), a, path_m)
            self.st = st
        if now - self.t_hist >= 10.0:          # Top-K change for the page: how many candidates, who leads, by how much
            self.t_hist = now
            self.hist = (self.hist + [dict(t=round(now - self.t_started), n_active=a["n_active"], n_alive=a["n_alive"], floor=a.get("leader_floor"), gap=a.get("gap"))])[-30:]
        stop_ok = cls != "DISCRIMINATIVE" and not fixed     # R40: only shown structure withholds a sign. 'Not known yet' does not (R39 withheld 1, 6, 1 candidates of the 11-03 runs before +102 s)
        busy = hw in HW_BUSY or vlm_busy or pair_status in ("PAIR_READING", "E7_VERIFYING")
        self.d = dict(kind="LIVE2_AUTO_STATE", authority="NONE: chooses how to collect evidence; a FIX comes only from final_verify / e6e7_link", state=st, state_ko=KO[st], structure_class=cls, class_since=self.since, assessment=a,
                      semantic_stop_allowed=stop_ok, needs_pantilt_consent=bool(cls == "AMBIGUOUS" and not e6_alive and not fixed), hardware=hw, e2_max_busy=0 if (vlm_resident or vlm_pending) else 1 if pair_status == "E7_VERIFYING" else 2 if busy else 3 if cls == "AMBIGUOUS" else 4,
                      pose_quality=a.get("pose_quality"), escalated=self.escalated, path_m=round(path_m, 1), wall=now, topk_hist=self.hist, n_transitions=self.n_transitions,
                      thresholds=dict(GAP_AMB=GAP_AMB, GAP_DISC=GAP_DISC, KF_MIN=KF_MIN, T_AMB_EARLY_S=T_AMB_EARLY_S, T_AMB_S=T_AMB_S, T_DISC_S=T_DISC_S, T_DECIDE_S=T_DECIDE_S, T_VERIFY_S=T_VERIFY_S, NEW_PATH_M=NEW_PATH_M))
        if self.state is not None:
            self._atomic(self.state / "auto_state.json", self.d)
            self._atomic(self.state / "e2" / "e2_budget.json", dict(max_busy=self.d["e2_max_busy"], wall=now, why=cls))
        return self.d

    def _atomic(self, p, obj):
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, default=str))
        os.replace(tmp, p)

    def _log(self, now, kind, a, b, why, ass, path_m):
        self.n_transitions += 1
        if self.state is None:
            return
        with open(self.state / "auto_transitions.jsonl", "a") as f:
            f.write(json.dumps(dict(wall=now, kind=kind, frm=a, to=b, why=why, raw=ass.get("raw"), leader_floor=ass.get("leader_floor"), rival_floor=ass.get("rival_floor"), gap=ass.get("gap"), near=ass.get("near_floors"),
                                    coarse=ass.get("coarse_done"), exact=ass.get("exact_done"), path_m=round(path_m, 1)), ensure_ascii=False, default=str) + "\n")


def stop_gate(state_dir, now=None, max_age_s=10.0):
    """visual_stop: None = may ask for a stop, else the suppression reason. No or an old file = not an automatic run or e34 is down: fails towards asking (the older behaviour)."""
    now = time.time() if now is None else now
    try:
        d = json.loads((Path(state_dir) / "auto_state.json").read_text())
    except (OSError, ValueError, TypeError):
        return None
    if now - float(d.get("wall") or 0) > max_age_s or d.get("semantic_stop_allowed"):
        return None
    if d.get("state") == "GLOBAL_FIX":
        return "SUPPRESSED_AFTER_GLOBAL_FIX"
    return "SUPPRESSED_STRUCTURE_SUFFICIENT" if d.get("structure_class") == "DISCRIMINATIVE" else "SUPPRESSED_STRUCTURE_ASSESSING"
