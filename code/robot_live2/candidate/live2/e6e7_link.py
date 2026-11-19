"""e6e7_link.py — LIVE2 E6 semantic result -> E3 ranking -> E7 LiDAR re-registration -> global decision (file-only, in e34).

Input  : <state>/e6_precision_results.jsonl — one line per precision observation (written by e6_precision_bridge.py when
         a capture + OCR/VLM exists): observation_id, source (AUTO_E5 | OPERATOR_ASSISTED), ocr_lines [[text, conf]],
         vlm (or null), capture metadata, and the E6 target (odom_link xy) from e6_observations.
Flow   : e6_semantic.evaluate (student classify_text / fuse / semantic maps) -> Tracker.apply_semantic -> E7 job on the
         semantically CONSISTENT hypotheses plus the best rivals (independent geometry) in a background thread at the lowest
         CPU priority (never blocks the E3/E4 loop; one job at a time, newer requests coalesce).
Decision (<state>/e7_state.json, global_decision):
  GLOBAL_CONFIRMED            a decisive semantic reading is CONSISTENT on hypothesis H AND E7 finds H geometrically unique
                              and separated (E7 GEOMETRY_CONSISTENT_SEMANTIC_PENDING with best == H)
  AMBIGUOUS                   E7 cannot separate H from a rival
  REJECTED                    a rival fits better than the semantic hypothesis by >= GAP_M (R35; a smaller lead is AMBIGUOUS) or its own fit failed
  MORE_OBSERVATION_REQUIRED   anything else (no decisive reading, not converged, too few frames)
  PROVISIONAL_SINGLE          (2026-10-22 E8 policy) E7 would confirm, but the request came from ONE observation — a single
                              observation never gives a global fix
  (2026-11-16) GLOBAL_CONFIRMED with path FUSED_FLOOR: fused_floor.py, the main rule since 11-16 (user directive) — after this E7 look,
                              floor = argmin of (E7 score / 0.05 m) + plate terms (room number on that floor's map, plate within 3 m of its room,
                              else ln 20), candidates converged and not crossing walls along the window path; confirmed when the next floor is
                              >= ln 100 behind. Record: <state>/e7_fused_floor.jsonl. LIVE2_FUSED_FLOOR=0 turns it off.
  (2026-11-17) the E7 look (continuation + evaluate) runs in its own process, e7_proc.py (user directive); same code and inputs,
                              this thread waits. Record: <state>/e7_proc.jsonl. LIVE2_E7_PROC=0 computes in this thread as before.
2026-11-18 (user decisions B1 B2 B3 B4 B6 B7; each has its own switch, default on):
  B1 LIVE2_E8_PATH_RANK   E7 takes the E8 candidates without a wall crossing first; those crossing a wall come last (e8_pair), never removed.
  B2 LIVE2_SEM_NEW_HYP    a hypothesis born after the last decisive room reading gets that reading's verdict (same e6_semantic rule), so a new
                          other-floor hypothesis no longer jumps ahead as 'no reading' (123844: H080-H085 on 8F/9F after '417').
  B3 LIVE2_SINGLE_PLATE_E7 one decisive room reading whose floor likelihood is >= e8_pair.SINGLE_FLOOR_P: a cheap pre-screen (the E2 rival cost, one best
                          candidate per floor) first; the full E7 runs only when geometry can separate floors (the plate floor ahead of every other floor by
                          the 0.08 m one plate cannot cover, or another floor ahead by more than one plate weighs, 0.15 m). fused_floor decides with the same
                          ln 100; a misread suspected on MISREAD_LOOKS full looks sets the reading aside (ranking, pair rule). E7's own confirm stays
                          PROVISIONAL_SINGLE. Records <state>/e7_single_plate.jsonl, semantic_misread.jsonl.
  B7 LIVE2_TEXT_FLOOR     words of a plate without a room number (map_text: jamo trigrams of the guide map's own room texts, no hand table) point to a floor
                          when its margin >= map_text.PLATE_MIN and >= TEXT_KNOWN_MIN of its trigrams are on the map: hypotheses of that floor WITHOUT a
                          room-number verdict get TEXT_FLOOR_SUPPORT (rank between consistent and no reading). Support only: never a mismatch, never a FIX
                          input, replaced by the next room number. Record <state>/text_floor_evidence.jsonl.
Policy (env LIVE2_E8_POLICY, default E8_PAIR): E8_PAIR = every precision record goes to e8_pair.PairManager (store, read
both images after the second capture, semantic + odometry consistency, E7 only for a consistent pair / group);
SINGLE_DIAGNOSTIC = the per-observation path above (diagnostic; its best outcome is PROVISIONAL_SINGLE).
GT is never read.

PATH A — STRUCTURE_ONLY (2026-10-27 directive): when E4 says CONFIDENT_GEOMETRY (its own rule: EVERY floor searched, gap >= 0.15 m
to any distinct rival incl. other floors, the same leader for >= 15 keyframes, >= 8 m path, recent cost <= 0.35 — unchanged
E4 thresholds) E7 is asked WITHOUT a semantic reading on the leader + its best rivals (any floor). A structure-only FIX needs
STRUCT_PASSES independent E7 windows whose ends are STRUCT_MIN_GAP_S..STRUCT_MAX_GAP_S apart (the operator's proposed 10-20 s
observation budget, NOT tuned on ground truth), each with E7 'geometry unique' (GEOMETRY_CONSISTENT_SEMANTIC_PENDING, no
ambiguity) and best == the E4 leader, the same FAST-LIO epoch, no < 8 Hz FAST-LIO in the window, and E4 still CONFIDENT on that
leader. Any failure -> no fix; that leader is not retried until E4 changes; E5/E6 (PATH B) continue as E4 asks.
2026-10-30 (staging) PATH A safety: (1) OFF unless e7_policy.structure_policy() finds a verified enable file whose code hash
matches the loaded E7 code (no default-on; LIVE2_STRUCTURE_FIX=0 still forces off, =1 never enables); the decision is written
to <state>/e7_policy.json with this pid for the tablet. (2) rivals = E3 top-3 + the best E2-exact candidate of up to 3 other
floors (e2_rivals); no usable E2 list for this odometry session -> no FIX (fail closed). (3) verification runs while the robot
MOVES (no stop request): a STRUCT_WIN_S window (>= STRUCT_MIN_PATH_M m of motion inside, else not evaluated) every
>= STRUCT_STEP_S; FIX = an unbroken run of passing windows (every evaluated window passes, ends <= STRUCT_RUN_GAP_S apart)
whose first and last windows are DISJOINT (ends >= STRUCT_WIN_S apart) — i.e. >= 2 independent windows AND every window in
between; any failing window restarts the run; a unique window for another candidate ends the attempt; <= STRUCT_MAX_WINDOWS
windows / STRUCT_BUDGET_S per leader. The 10-28 rule (2 overlapping 90 s windows 10-30 s apart, after a stop) is replaced: windows
at rest overlap and are not independent. E7 thresholds (GAP_M, segment overlap, wall crossing) are unchanged.
"""
import importlib.util
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

import e2_region_ledger as RL  # noqa: E402
import e6_semantic as S
import e7_policy as P7
import e7_retro as R7

E7_PATH = Path(__file__).resolve().with_name("e7_rematch.py")


def _path(rows, t0, t1):
    xy = [(r[2], r[3]) for r in rows or [] if t0 <= r[1] <= (t1 if t1 is not None else t0)]
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(xy, xy[1:])))


def _max_step(rows, t0, t1):
    xy = [(r[2], r[3]) for r in rows or [] if t0 <= r[1] <= t1]
    return max((math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(xy, xy[1:])), default=0.0)


def e2_input_windows(e2_dir, e2):
    """[(t0, t1)] odometry intervals E2 has fitted (pass_start t_now - window_s .. t_now). E7 windows that overlap one of
    these are NOT used as independent evidence (the E2 candidate was derived from those scans)."""
    W = float((e2 or {}).get("window_s") or 60.0)
    out = []
    try:
        for ln in (Path(e2_dir) / "e2_passes.jsonl").read_text().splitlines():
            r = json.loads(ln)
            if r.get("kind") == "pass_start" and r.get("t_now"):
                out.append((float(r["t_now"]) - W, float(r["t_now"])))
    except (OSError, ValueError):
        pass
    if (e2 or {}).get("t_now"):
        out.append((float(e2["t_now"]) - W, float(e2["t_now"])))
    return out


def _load_e7():
    """The LIVE2 e7_rematch next to this file. e6_semantic puts the research src/.../e5_e7 (which has its own
    e7_rematch.py, no evaluate()) first on sys.path, so a plain `import e7_rematch` in e34 resolves to the wrong module."""
    m = sys.modules.get("e7_rematch")
    if m is not None and Path(m.__file__).resolve() == E7_PATH:
        return m
    spec = importlib.util.spec_from_file_location("live2_e7_rematch", E7_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E7 = _load_e7()

E7_RIVALS = 3
MIN_WINDOWS, POSE_AGREE_M, POSE_AGREE_DEG, E7_WIN_MIN_S = 2, 1.5, 6.0, 60.0   # R41: two disjoint E7 windows must agree; the distance is FAST-LIO drift over one retry plus E7's own spread on right places (X20261103_001810), not tuned on GT
E7_E2_RIVALS = 3     # 2026-10-30 (staging): + the best E2-exact candidate of up to 3 OTHER floors, pre-screened on the E7 window.
                     # Only ADDS rivals (a FIX can only get harder): E3's duplicate merge (1 m / 10 deg at the current pose, scale not
                     # compared) can replace a different-scale other-floor geometry by a coarse one, so the closest other-floor rival
                     # was missing from E7 in one loc1 replay (A_p3: pass gap 0.0552 -> 0.0039 with the 7F E2 twin restored).


POLICIES = ("E8_PAIR", "SINGLE_DIAGNOSTIC")
STRUCT_PASSES, STRUCT_MIN_GAP_S, STRUCT_MAX_GAP_S = 2, 10.0, 30.0
# 2026-10-30 (staging) structure path windows — chosen from E7 data sufficiency on the real loc1/loc2 records, NOT on GT:
STRUCT_WIN_S, STRUCT_MIN_PATH_M = 30.0, 5.0            # each E7 window: 30 s, >= 5 m of motion inside it
STRUCT_STEP_S, STRUCT_RUN_GAP_S = 10.0, 30.0            # a new window every >= 10 s; a run breaks if ends are > 30 s apart (data gap)
STRUCT_MAX_WINDOWS, STRUCT_BUDGET_S = 40, 420.0
# 2026-10-30 21:xx retro windows: at E4 CONFIDENT the stored windows of the last RETRO_HORIZON_S are re-checked (e7_retro);
RETRO_HORIZON_S, RETRO_MAX_AGE_S, RETRO_PROCS, RETRO_CORES = 180.0, 180.0, 2, (6, 7)
EARLY_WIN_S, EARLY_STEP_S, EARLY_MIN_PATH_M, EARLY_MIN_KF, EARLY_RIVALS = 30.0, 30.0, 5.0, 3, 8   # R34 early E7 SHADOW: 30 s windows, record only
EARLY_MAX_KF_PENDING = 20
RECOVERY_TIMEOUT_S = 300.0  # regional E2 (one floor, ~60-110 s live) + E3 intake
SEM_NEW_HYP = os.environ.get("LIVE2_SEM_NEW_HYP", "1") == "1"            # B2 (2026-11-18)
SINGLE_E7 = os.environ.get("LIVE2_SINGLE_PLATE_E7", "1") == "1"          # B3 (2026-11-18)
TEXT_FLOOR = os.environ.get("LIVE2_TEXT_FLOOR", "1") == "1"              # B7 (2026-11-18)
MISREAD_LOOKS = 2          # B3: full single-plate looks (disjoint windows, E7 retry spacing) with a misread suspicion before the reading is set aside
ANCHOR_WALLCHECK = os.environ.get("LIVE2_ANCHOR_WALLCHECK", "1") == "1"   # 2026-11-18 (user "경로까지 보고 만드는걸로"): anchored candidates must not put the path through walls
ANCHOR_PATH = os.environ.get("LIVE2_ANCHOR_PATH", "1") == "1"             # 2026-11-18: second plate on the same floor -> path anchor (earlier plate pinned, rotation from the path's LiDAR)
E7_PLATE_CANDS = max(1, int(os.environ.get("LIVE2_E7_PLATE_CANDS", "4")))   # 2026-11-18 사용자 결정 "4+1 로 바꿔줘": E8 쌍 판정 E7 이 맞추는 명판 일치 후보 수(경쟁 1개는 그대로). 2 = 이전(2+1).
# 근거: 로봇 실측(조용한 상태, 쌍 판정 22번) 2+1 9.7 s → 4+1 14.9 s(정합 한 번 약 1.4 s), 확정 17 → 18, 위치 개선 2번(5.76 → 1.24 m, 미확정 → 1.05 m), 나머지 같음.
TEXT_KNOWN_MIN = 0.30      # B7: share of a plate's trigrams found on the map (= tools/lexicon/text_floor.KNOWN_MIN); below it the words are 'not on the map'
JUMP_STEP_M = 1.5          # consecutive LiveStore samples (~1.5 s) further apart than this: position jump -> window unusable
STRUCT_STOP_WAIT_S, STRUCT_VERIFY_BUDGET_S = 45.0, 20.0     # operator stop window; verification budget after the stop


def e2_rivals(e2, store, t_end, leader_floor, light, k=E7_E2_RIVALS, window_s=90.0):
    """-> ([{candidate_id, floor, rec, prescreen}], seconds). Pre-screen every E2 exact candidate of another floor with the E7 cost
    (structural window, identity rotation/scale, best of a +-1 m xy grid) and keep the best one per floor, up to k floors.
    Stricter-only: these are extra rivals for the unchanged E7 decision. Nothing here is a FIX criterion."""
    t0 = time.perf_counter()
    if not e2 or k <= 0:
        return [], 0.0
    pf = e2.get("per_floor") or {}
    todo = [(i, c) for i, c in enumerate(e2.get("candidates") or e2.get("candidates_pass") or [])
            if c.get("floor") != leader_floor and (c.get("stage") or "exact") == "exact" and (pf.get(str(c["floor"])) or {}).get("light")]
    if not todo:                                                                # no other-floor exact candidate: nothing to add
        return [], round(time.perf_counter() - t0, 2)
    fr = E7.frames(store, t_end, window_s, structural=True)
    if len(fr) < 5:
        return [], round(time.perf_counter() - t0, 2)
    pool = []
    for i, c in todo:
        meta = pf.get(str(c["floor"])) or {}
        rec = dict(c, light=meta["light"], H=meta.get("H"))
        L = light(meta["light"])
        cost = min(E7.cost(L, rec, fr, (dx, dy, 0.0, 1.0))[0] for dx in (-1.0, 0.0, 1.0) for dy in (-1.0, 0.0, 1.0))
        pool.append((cost, c["floor"], i, rec))
    pool.sort(key=lambda r: r[0])
    out, seen = [], set()
    for cost, f, i, rec in pool:
        if f in seen:
            continue
        seen.add(f)
        out.append(dict(candidate_id=f"E2X_F{f}_{i}", floor=f, rec=rec, prescreen=round(cost, 4)))
        if len(out) >= k:
            break
    return out, round(time.perf_counter() - t0, 2)


class E6E7Link:
    def __init__(self, state_dir, tracker, sem_floors=None, policy=None, reader=None, vlm=None, vlm_admit=None):
        self.state = Path(state_dir)
        self.tr = tracker
        self.sf = sem_floors or S.SemanticFloors()
        self.off = 0
        self.job = None
        self.lock = threading.Lock()
        self.last = dict(kind="LIVE2_E7_STATE", global_decision="MORE_OBSERVATION_REQUIRED", why="no precision observation yet")
        self.policy = policy or os.environ.get("LIVE2_E8_POLICY", "E8_PAIR")
        if self.policy not in POLICIES:
            raise ValueError(f"policy {self.policy} not in {POLICIES}")
        self.pair = None
        if self.policy == "E8_PAIR":
            import e8_pair as EP
            if vlm is None and os.environ.get("SCI_E6_VLM") == "1":
                import e6_pipeline as P
                vlm, vlm_admit = P.vlm_endpoint(state_dir=self.state)
            self.pair = EP.PairManager(self.state, tracker, self.sf, self.request_e7, reader=reader, vlm=vlm, vlm_admit=vlm_admit,
                                       lio_series=self._lio_series)
        self.struct_policy = P7.structure_policy()
        try:
            P7.write_report(self.state, self.struct_policy)
        except OSError as ex:                                                   # report missing -> tablet shows "unconfirmed"
            print(f"[e6e7] e7_policy report not written: {type(ex).__name__}: {ex}", flush=True)

    def _lio_series(self, n=900):
        try:
            return [json.loads(x) for x in (self.state / "lio_series.jsonl").read_text().splitlines()[-n:]]
        except (OSError, ValueError):
            return []

    def _new_results(self):
        p = self.state / "e6_precision_results.jsonl"
        try:
            with open(p) as f:
                f.seek(self.off)
                lines = f.readlines()
                self.off = f.tell()
        except OSError:
            return []
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
        return out

    def tick(self, e6_recent=None, odom_now=None):
        if self.pair is not None:
            for r in self._new_results():
                snap = self.pair.on_precision(r)
                self.observation_outcome(r, "PAIR_" + snap["state"], snap["why"] or "")
            self.pair.on_e6_progress(e6_recent, odom_now)
            self.recovery_tick()
            for step in (self._new_hyp_tick, self._text_tick):     # B2, B7 (2026-11-18): ranking only; a failure never stops the tick
                try:
                    step()
                except Exception as ex:                             # noqa: BLE001
                    print(f"[e6e7] {step.__name__}: {type(ex).__name__}: {ex}", flush=True)
            self.pair.reevaluate_on_new_hypotheses()
            self.pair.retry_e7()
            try:
                self._single_e7_tick()                              # B3 (2026-11-18)
            except Exception as ex:                                 # noqa: BLE001
                print(f"[e6e7] single plate: {type(ex).__name__}: {ex}", flush=True)
            self.pair.snapshot()
            return self.snapshot()
        for r in self._new_results():
            tgt = r.get("target_odom_link_xy")
            hyps = [dict(id=h.id, floor=h.floor, rec=h.rec) for h in self.tr.alive()]
            t0 = time.time()
            usable, gate = evidence_gate(r)
            alts, info = S.label_alternatives([tuple(x) for x in r.get("ocr_lines") or []] if usable else [],
                                              r.get("vlm") if usable else None)
            res = S.evaluate(alts, hyps, self.sf, tgt) if tgt else dict(alternatives=alts, per_hypothesis={})
            res["fusion"] = info.get("fusion")
            res["evidence_gate"] = gate
            t1 = time.time()
            n_decisive = self.tr.apply_semantic(res, r.get("observation_id"), r.get("source"))
            t2 = time.time()
            with open(self.state / "e6_hypothesis_updates.jsonl", "a") as f:
                for row in hypothesis_updates(r, res, usable, gate, hyps):
                    f.write(json.dumps(dict(row, semantic_map_match_ms=round(1000 * (t1 - t0), 1),
                                            hypothesis_update_ms=round(1000 * (t2 - t1), 1)), default=str) + "\n")
            with open(self.state / "e6_semantic_results.jsonl", "a") as f:
                f.write(json.dumps(dict(res, observation_id=r.get("observation_id"), source=r.get("source"), wall=time.time()),
                                   default=str) + "\n")
            # sem_verdict is sticky (set by the last decisive reading): only THIS record's gated, decisive reading may start E7,
            # so a later rejected / non-field record never re-runs E7 under its own observation_id.
            cons = [h for h in self.tr.alive() if h.sem_verdict == "SEMANTIC_CONSISTENT"]
            if usable and n_decisive and cons:
                self.observation_outcome(r, "E7_RUNNING", f"semantic reading consistent with {cons[0].id} -> E7 LiDAR check running")
                self.request_e7(cons, r)
            elif usable and n_decisive:          # 2026-10-22 N2: a valid reading can exist while no live hypothesis is at that
                self.candidate_missing(r, res)   # room (E2 produced no near-correct seed) -> say so; never invent a hypothesis
            else:
                alts_ = res.get("alternatives") or []
                why = (f"NO_EVIDENCE: {gate}" if not usable else
                       f"NO_LABEL: OCR/VLM gave no room label (fusion {(res.get('fusion') or {}).get('status')})" if not alts_ else
                       f"NOT_DECISIVE: label {[a[0] for a in alts_][:3]} below the decisive probability")
                self.observation_outcome(r, "E7_NOT_RUN", why)
        return self.snapshot()

    # ------------------------------------------------------------------ semantic recovery (2026-10-30)
    def recovery_tick(self):
        """E8 CANDIDATE_MISSING (two independent plates fit ONE floor, but no live hypothesis is there) -> one regional E2
        request (e2/e2_region_request.json, student fit limited to the plate region) -> when its candidates are in E3, the SAME
        closed group is re-evaluated (e8_pair.recover_closed) and goes to E7 only via REGION_CANDIDATE_READY. Never a FIX by
        itself; every step and failure is written to <state>/semantic_recovery.json for the tablet."""
        P = self.pair
        self._single_obs_tick(P)
        rc = getattr(self, "recovery", None)
        h = P.history[-1] if P.history else None
        now = time.time()
        if h and h["status"] == "CANDIDATE_MISSING" and (rc is None or rc.get("group_id") != h["group_id"]) and not P.group:
            region, why = P.recovery_region(getattr(self.tr.store, "rows", None))
            rc = self.recovery = dict(kind="LIVE2_SEMANTIC_RECOVERY", group_id=h["group_id"], requested_wall=now, region=region,
                                      status="REGION_REQUESTED" if region else "SKIPPED", why=why or "regional E2 search requested")
            if region:
                req = dict(request_id=h["group_id"], wall=now, why=f"SEMANTIC_RECOVERY {h['group_id']}", region=region,
                           odom_session=getattr(self.tr, "session", None))
                RL.submit_request(self.tr.e2_dir, req)
            self._recovery_write()
        elif rc and rc["status"] == "REGION_REQUESTED":
            sig = tuple(sorted(x.id for x in self.tr.alive() if (getattr(x, "rec", {}) or {}).get("provenance") == "SEMANTIC_ANCHORED"))
            if sig and sig != rc.get("anchored_sig") and not P.group:     # R41: an anchored fit takes seconds, the regional E2 minutes: do not wait for it when the anchored one already places the plates
                rc["anchored_sig"] = sig
                ev_ = P.recover_closed()
                rc.update(verdict=(ev_ or {}).get("verdict"), why=(ev_ or {}).get("why"))
                if (ev_ or {}).get("verdict") != "CANDIDATE_MISSING":
                    rc.update(status="RE_EVALUATED", evaluated_wall=now, by="ANCHORED")
                self._recovery_write()
                return
            try:
                e2 = json.loads((Path(self.tr.e2_dir) / "e2_state.json").read_text())
            except (OSError, ValueError):
                e2 = {}
            mine = [x for x in e2.get("regional") or [] if x.get("request_id") == rc["group_id"]]
            if mine and mine[-1].get("status") == "RETRY_WAIT" and rc.get("e2_wait") != mine[-1].get("attempt"):
                rc.update(e2_wait=mine[-1].get("attempt"), why=f"regional E2 waits for motion: {mine[-1].get('why_failed')}")
                self._recovery_write()
            done = [x for x in mine if x.get("status") in ("DONE", "FAILED")]
            if done:
                d = done[-1]
                rc["e2"] = {k: d.get(k) for k in ("status", "n", "sec", "region_pass_id", "why_failed", "candidates")}
                if d.get("status") != "DONE" or not d.get("n"):
                    rc.update(status="RESEARCH_FAILED", why=f"regional E2: {d.get('status')} {d.get('why_failed') or 'no candidate'}")
                elif any(getattr(x, "rec", {}).get("regional_request") == rc["group_id"] or
                         any(pk[0] == d.get("region_pass_id") for pk in getattr(x, "pass_keys", [])) for x in self.tr.alive()):
                    ev = P.recover_closed()
                    rc.update(status="RE_EVALUATED", verdict=(ev or {}).get("verdict"), why=(ev or {}).get("why"), evaluated_wall=now)
                elif now - rc["requested_wall"] > RECOVERY_TIMEOUT_S:
                    rc.update(status="RESEARCH_FAILED", why="regional candidates did not enter E3 (dropped / merged)")
                self._recovery_write()
            elif now - rc["requested_wall"] > RECOVERY_TIMEOUT_S:
                rc.update(status="TIMEOUT", why=f"no regional E2 result within {RECOVERY_TIMEOUT_S:.0f} s")
                self._recovery_write()

    def _single_obs_tick(self, P):
        """2026-11-01 (field X20261031_231940: '402' read decisively, 4F only, but every 4F hypothesis was 15-27 m away and the
        second plate failed, so the pair rule never asked for the regional search): ONE decisive usable reading with no
        semantically consistent hypothesis -> one regional E2 request for that observation (candidate generation only).
        Status in <state>/semantic_recovery_single.json(l). The pair rule and E7 decide any fix exactly as before."""
        ev = P.last_eval or {}
        done = getattr(self, "_single_done", None)
        if done is None:
            done = self._single_done = {}
        now = time.time()
        for r in ev.get("per_observation") or []:
            oid = r.get("observation_id")
            if not oid or oid in done or not r.get("usable") or not r.get("decisive"):
                continue
            # R36 (11-02 X20261102_025326: '405' read, a 4F hypothesis happened to lie within tolerance -> NOT_NEEDED, no anchored fit; E3 stayed on 1F until the end): a reading always
            # gets its anchored fit and orders the live hypotheses at once. Provisional: no FIX path here.
            self._reading_rank(r, oid)
            self._anchor_start(P, r)
            if any((v or {}).get("verdict") == "SEMANTIC_CONSISTENT" for v in (r.get("per_hypothesis") or {}).values()):
                done[oid] = dict(observation_id=oid, status="NOT_NEEDED", why="a live hypothesis already fits this reading: no regional E2 (anchored fit started)", wall=now)
                self._single_write(done[oid])
                continue
            region, why = P.single_obs_region(r, getattr(self.tr.store, "rows", None))
            rec = dict(kind="LIVE2_SEMANTIC_RECOVERY_SINGLE", observation_id=oid, group_id=P.gid, wall=now, region=region,
                       status="REGION_REQUESTED" if region else "SKIPPED", why=why or "regional E2 search requested from one decisive reading")
            if region:
                req = dict(request_id=f"OBS_{oid}", wall=now, why=f"SEMANTIC_RECOVERY_SINGLE {oid}", region=region,
                           odom_session=getattr(self.tr, "session", None))
                RL.submit_request(self.tr.e2_dir, req)
            done[oid] = rec
            self._single_write(rec)

    def _reading_rank(self, r, oid):
        """R36: one decisive room reading orders E3 now (consistent first, floor mismatch last; e3 rank_key). Reversible: the next reading replaces it. Room readings only: a category exists on every floor."""
        try:
            if r.get("reading_kind") == "ROOM":
                self.tr.apply_semantic(dict(alternatives=r.get("labels"), floor_likelihood=r.get("floor_likelihood"), per_hypothesis=r.get("per_hypothesis") or {}), oid, "E8_SINGLE_READING")
                o = self._obs_of(oid)                    # B2/B3 (2026-11-18): the last decisive room reading, for hypotheses born later and the single-plate look
                self._last_reading = dict(observation_id=oid, labels=[tuple(x) for x in r.get("labels") or []], floor_likelihood=r.get("floor_likelihood") or {},
                                          target=(o or {}).get("target_odom_link_xy"), wall=time.time(), excluded=None)
        except Exception as ex:                         # noqa: BLE001
            print(f"[e6e7] reading rank: {type(ex).__name__}: {ex}", flush=True)

    def _obs_of(self, oid):
        P = self.pair
        if P is None:
            return None
        return next((g for g in list(P.group) + list((getattr(P, "closed", None) or {}).get("observations") or []) if g.get("observation_id") == oid), None)

    def _jsonl(self, name, rec):
        try:
            with open(self.state / name, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------ B2 (2026-11-18): verdict for hypotheses born after the reading
    def _new_hyp_tick(self):
        lr = getattr(self, "_last_reading", None)
        if not SEM_NEW_HYP or not lr or lr.get("excluded") or not lr.get("target") or not lr.get("labels"):
            return
        oid = lr["observation_id"]
        new = [h for h in self.tr.alive() if h.sem_verdict in (None, "NO_LABEL", "TEXT_FLOOR_SUPPORT") and not any(x.get("observation_id") == oid for x in h.sem)]
        if not new:
            return
        sem = S.evaluate(lr["labels"], [dict(id=h.id, floor=h.floor, rec=h.rec) for h in new], self.sf, list(lr["target"]))
        ph = sem.get("per_hypothesis") or {}
        self.tr.apply_semantic(dict(alternatives=lr["labels"], floor_likelihood=lr.get("floor_likelihood"), per_hypothesis=ph), oid, "E8_READING_NEW_HYPOTHESIS")
        for h in new:
            if h.id not in ph:                           # never judged again for this reading
                h.sem.append(dict(observation_id=oid, verdict=None, source="E8_READING_NEW_HYPOTHESIS", decisive=False, wall=time.time()))
        self._jsonl("semantic_new_hypotheses.jsonl", dict(kind="LIVE2_SEMANTIC_NEW_HYPOTHESES", wall=time.time(), observation_id=oid,
                                                          verdicts={h.id: (ph.get(h.id) or {}).get("verdict") for h in new}))

    # ------------------------------------------------------------------ B7 (2026-11-18): plate words as floor support
    def _text_tick(self):
        P = self.pair
        if not TEXT_FLOOR or P is None:
            return
        done = getattr(self, "_text_done", None)
        if done is None:
            done = self._text_done = set()
        for r in (P.last_eval or {}).get("per_observation") or []:
            oid = r.get("observation_id")
            if not oid or oid in done:
                continue
            done.add(oid)
            if (r.get("reading_kind") == "ROOM" and r.get("decisive")) or r.get("why_not") == "IMAGE_NOT_USABLE":
                continue                                 # a room number decides by itself (number first); an unusable image says nothing
            o = self._obs_of(oid)
            if o is None:
                continue
            import map_text as MT
            lex = MT.lexicon()
            text = MT.plate_text(o) if lex is not None else ""
            sc = lex.score(text) if (lex is not None and MT.norm(text)) else None
            known = (sc["known"] / sc["grams"]) if sc and sc["grams"] else 0.0
            ok = bool(sc) and known >= TEXT_KNOWN_MIN and sc["margin"] >= MT.PLATE_MIN
            rec = dict(kind="LIVE2_TEXT_FLOOR", wall=time.time(), observation_id=oid, text=text[:80], lexicon=lex is not None, ok=ok,
                       floor=sc["top"] if ok else None, top=(sc or {}).get("top"), second=(sc or {}).get("second"), margin=(sc or {}).get("margin"),
                       known_share=round(known, 2), plate_min=MT.PLATE_MIN, known_min=TEXT_KNOWN_MIN,
                       why=None if ok else ("no map lexicon" if lex is None else "no letters" if not sc else "words not on the map" if known < TEXT_KNOWN_MIN else "no floor-separating words"))
            self._jsonl("text_floor_evidence.jsonl", rec)
            if ok:
                self._text_floor = dict(floor=int(sc["top"]), observation_id=oid, margin=sc["margin"], wall=time.time())
        tf = getattr(self, "_text_floor", None)
        if not tf:
            return
        for h in self.tr.alive():                         # support only: hypotheses without a room-number verdict; others untouched
            if h.sem_verdict in (None, "NO_LABEL") and h.floor == tf["floor"]:
                h.sem_verdict = "TEXT_FLOOR_SUPPORT"
                h.sem.append(dict(verdict="TEXT_FLOOR_SUPPORT", observation_id=tf["observation_id"], source="E8_TEXT", decisive=False, wall=time.time()))
            elif h.sem_verdict == "TEXT_FLOOR_SUPPORT" and h.floor != tf["floor"]:
                h.sem_verdict = None

    # ------------------------------------------------------------------ B3 (2026-11-18): one decisive plate -> E7 only when geometry can separate floors
    def _single_state(self, oid):
        d = getattr(self, "_single_looks", None)
        if d is None:
            d = self._single_looks = {}
        return d.setdefault(oid, dict(n=0, wall=0.0, path=None, suspects=0, pre=0, skipped_why=None))

    def _single_e7_tick(self):
        P = self.pair
        lr = getattr(self, "_last_reading", None)
        if not SINGLE_E7 or P is None or not lr or lr.get("excluded") or not lr.get("target") or not lr.get("labels"):
            return
        if (self.last or {}).get("global_decision") == "GLOBAL_CONFIRMED" or (self.job and self.job.is_alive()):
            return
        if P.status in ("REGION_CANDIDATE_READY", "E7_VERIFYING", "PAIR_READING", "PAIR_CONSISTENCY_CHECK") or \
                ((P.last_eval or {}).get("verdict") == "REGION_CANDIDATE_READY" and P.group):
            return                                       # the pair rule has the plates: it goes first
        import e8_pair as EP
        oid = lr["observation_id"]
        st = self._single_state(oid)
        fl = {int(k): float(v) for k, v in (lr.get("floor_likelihood") or {}).items()}
        if not fl:
            return
        fp, pmax = max(fl.items(), key=lambda kv: kv[1])
        if pmax < EP.SINGLE_FLOOR_P:                    # the reading itself fits several floors: no single-plate look
            if st["skipped_why"] is None:
                st["skipped_why"] = f"floor likelihood {pmax:.2f} < {EP.SINGLE_FLOOR_P} (the reading fits other floors too)"
                self._jsonl("e7_single_plate.jsonl", dict(kind="LIVE2_SINGLE_PLATE", wall=time.time(), observation_id=oid, step="SKIPPED", why=st["skipped_why"]))
            return
        cons = sorted([h for h in self.tr.alive() if h.sem_verdict == "SEMANTIC_CONSISTENT" and h.floor == fp], key=lambda h: h.rank_key())
        if not cons:
            return
        rows = getattr(self.tr.store, "rows", None) or []
        path = sum(math.hypot(b[2] - a[2], b[3] - a[3]) for a, b in zip(rows, rows[1:]))
        now = time.time()
        if st["n"] >= EP.E7_RETRY_MAX or (st["n"] and (now - st["wall"] < EP.E7_RETRY_S or path - (st["path"] or 0.0) < EP.E7_RETRY_M)):
            return
        st.update(n=st["n"] + 1, wall=now, path=path)
        rivals = {}
        for h in sorted(self.tr.alive(), key=lambda h: h.rank_key()):    # one best E3 hypothesis per other floor; e2_rivals fills floors E3 has none on
            if h.floor != fp and h.floor not in rivals:
                rivals[h.floor] = h
        lab = lr["labels"][0]
        r = dict(observation_id=f"SINGLE_{oid}", source="E8_SINGLE", e7_fast=True, target_odom_link_xy=lr["target"],
                 single_plate=dict(observation_id=oid, label=str(lab[0]), conf=float(lab[1]), target=list(lr["target"]), floor=fp, suspects=st["suspects"], look=st["n"]))
        cands = [dict(candidate_id=h.id, floor=h.floor, rec=dict(h.rec, light=h.light_dir)) for h in cons[:2] + list(rivals.values())]
        r["consistent_ids"] = [h.id for h in cons[:2]]
        with self.lock:
            if self.job and self.job.is_alive():
                return
            self.job = threading.Thread(target=self._run, args=(cands, r, cons[0].id), daemon=True)
            self.job.start()

    def _single_go(self, cands, r, t_end):
        """B3 pre-screen (E2 rival cost: E7 cost at identity, best of a +-1 m grid, 90 s structural window), one best candidate per floor.
        -> True when the full E7 can matter: the plate floor ahead of EVERY other floor by >= (ln100 - ln20) x SIG_G, or another floor ahead by >= ln20 x SIG_G."""
        import fused_floor as FF
        sp = r["single_plate"]
        fp = int(sp["floor"])
        rec = dict(kind="LIVE2_SINGLE_PLATE", wall=time.time(), observation_id=sp["observation_id"], step="PRESCREEN", look=sp.get("look"), floor=fp)
        fr = E7.frames(self.tr.store, t_end, 90.0, structural=True) if t_end else []
        if len(fr) < 5:
            self._jsonl("e7_single_plate.jsonl", dict(rec, go=False, why=f"only {len(fr)} frames"))
            return False
        try:
            e2 = json.loads((Path(self.tr.e2_dir) / "e2_state.json").read_text())
            if e2.get("odom_session") != getattr(self.tr, "session", None):
                e2 = None
        except (OSError, ValueError, TypeError):
            e2 = None
        have = {int(c["floor"]) for c in cands}
        extra, _ = e2_rivals(e2, self.tr.store, t_end, fp, self.tr.light, k=8) if e2 else ([], 0.0)
        cands.extend({k: x[k] for k in ("candidate_id", "floor", "rec")} for x in extra if int(x["floor"]) not in have)
        pre = {}
        for c in cands:
            L = self.tr.light(c["rec"]["light"])
            v = min(E7.cost(L, c["rec"], fr, (dx, dy, 0.0, 1.0))[0] for dx in (-1.0, 0.0, 1.0) for dy in (-1.0, 0.0, 1.0))
            f = int(c["floor"])
            pre[f] = min(pre.get(f, float("inf")), float(v))
        floors = sorted(getattr(self.sf, "maps", {}) or range(1, 10))
        mine = pre.get(fp)
        oth = {f: v for f, v in pre.items() if f != fp}
        missing = [f for f in floors if f != fp and f not in pre]
        need, weigh = (FF.LAM - FF.MISS) * FF.SIG_G, FF.MISS * FF.SIG_G
        lead = None if (mine is None or not oth) else min(oth.values()) - mine
        contra = sorted(f for f, v in oth.items() if mine is not None and mine - v >= weigh)
        go = bool(contra) or (lead is not None and lead >= need and not missing)
        self._jsonl("e7_single_plate.jsonl", dict(rec, go=go, prescreen={str(k): round(v, 4) for k, v in sorted(pre.items())}, lead_m=None if lead is None else round(lead, 4),
                                                   need_m=round(need, 3), contra=contra, missing_floors=missing, n_cands=len(cands),
                                                   why=("another floor fits better by more than one plate weighs: full E7 checks a misread" if contra else
                                                        f"plate floor ahead of every floor by {lead:.3f} m: full E7" if go else
                                                        "no candidate on the plate floor" if mine is None else
                                                        f"floors {missing} have no candidate: one plate cannot be backed" if missing else
                                                        f"geometry does not separate floors (lead {lead:.3f} < {need:.2f} m): no full E7")))
        self._single_state(sp["observation_id"])["pre"] += 1
        return go

    def _single_after(self, r, out):
        sp = r.get("single_plate") or {}
        oid = sp.get("observation_id")
        doc = getattr(self, "_fused_doc", None) or {}
        sd = doc.get("single_plate") or {}
        st = self._single_state(oid)
        if doc.get("observation_id") == r.get("observation_id") and sd.get("misread_suspect"):
            st["suspects"] += 1
        self._jsonl("e7_single_plate.jsonl", dict(kind="LIVE2_SINGLE_PLATE", wall=time.time(), observation_id=oid, step="E7", look=sp.get("look"),
                                                   global_decision=out.get("global_decision"), why=out.get("why"), single=sd, suspects=st["suspects"]))
        if st["suspects"] >= MISREAD_LOOKS:
            self._exclude_reading(oid, f"single-plate E7: geometry beat the plate floor on {st['suspects']} looks ({sd.get('why')})")
        if out.get("global_decision") == "GLOBAL_CONFIRMED" and self.pair is not None and self.pair.group:
            self.pair._close("GLOBAL_FIX", out.get("why"))

    def _exclude_reading(self, oid, why):
        lr = getattr(self, "_last_reading", None)
        if lr and lr.get("observation_id") == oid:
            lr["excluded"] = why
        n = 0
        for h in self.tr.alive():
            last = next((x for x in reversed(h.sem) if x.get("decisive")), None)
            anch = ((h.rec.get("evidence") or {}) if isinstance(h.rec.get("evidence"), dict) else {}).get("observation_id") == oid
            if ((last and last.get("observation_id") == oid) or anch) and h.sem_verdict in ("SEMANTIC_CONSISTENT", "PLACE_MISMATCH", "FLOOR_MISMATCH"):
                h.sem_verdict = None
                h.sem.append(dict(verdict="EXCLUDED_AS_MISREAD", observation_id=oid, why=why, decisive=False, wall=time.time()))
                n += 1
        if self.pair is not None:
            self.pair.excluded = dict(getattr(self.pair, "excluded", None) or {}, **{oid: why})
        self._jsonl("semantic_misread.jsonl", dict(kind="LIVE2_SEMANTIC_MISREAD", wall=time.time(), observation_id=oid, why=why, n_hypotheses_reset=n))

    # ------------------------------------------------------------------ R34 SEMANTIC_ANCHORED channel (candidate generation only)
    def _anchor_start(self, P, r):
        """One decisive reading naming exactly one room on one floor -> a background anchored fit. The candidates are provisional: E8 + E7 alone can confirm them."""
        if os.environ.get("LIVE2_ANCHOR", "1") != "1":
            return
        region, why = P.single_obs_region(r, getattr(self.tr.store, "rows", None))
        oid = r.get("observation_id")
        o = next((g for g in list(P.group) + list((getattr(P, "closed", None) or {}).get("observations") or []) if g.get("observation_id") == oid), None)   # R41: the evaluation may have closed the group a moment ago (405 of X20261103_001810: 'no target position', no anchored fit)
        if not region or not o or not o.get("target_odom_link_xy"):
            self._anchor_write(dict(observation_id=oid, status="SKIPPED", why=why or "no target position"))
            return
        pend = getattr(self, "_anchor_pending", None)
        if pend is None:
            pend = self._anchor_pending = []
        pend.append(dict(o=o, region=region, r=r, wall=time.time()))
        seen = getattr(self, "_anchor_readings", None)
        if seen is None:
            seen = self._anchor_readings = []
        if ANCHOR_PATH:                                   # 2026-11-18: an earlier plate of another room on the same floor, same FAST-LIO session -> path anchor first
            prev = next((x for x in reversed(seen) if int(x["region"]["floor"]) == int(region["floor"]) and x["region"]["rooms"][0] != region["rooms"][0]
                         and x["o"].get("session") == o.get("session") and x["o"].get("target_odom_link_xy")), None)
            if prev is not None:
                pend.insert(0, dict(kind="PATH", o=o, region=region, r=r, prev=prev, wall=time.time()))
        seen.append(dict(o=o, region=region))
        self._anchor_kick()

    def _anchor_kick(self):
        pend = getattr(self, "_anchor_pending", None)
        th = getattr(self, "_anchor_thread", None)
        if pend and (th is None or not th.is_alive()):
            self._anchor_thread = threading.Thread(target=self._anchor_job, args=(pend.pop(0),), daemon=True)
            self._anchor_thread.start()

    def _anchor_job(self, job):
        import semantic_anchor as SA
        import e8_pair as EP
        try:
            os.sched_setaffinity(threading.get_native_id(), {6, 7})
        except (AttributeError, OSError):
            pass
        o, region = job["o"], job["region"]
        oid = o["observation_id"]
        t0 = time.time()
        try:
            floor, room = int(region["floor"]), region["rooms"][0]
            st = self.tr.store
            rows = getattr(st, "rows", None)
            e2 = json.loads((Path(self.tr.e2_dir) / "e2_state.json").read_text())
            pr = SA.priors(e2, floor, None)
            # R37 (field: WAIT_E2_PRIOR while 4F was being searched: its per_floor entry had no map yet): whichever record has the light map
            pf = next((x for x in ((e2.get("per_floor") or {}).get(str(floor)), ((e2.get("coarse") or {}).get("per_floor") or {}).get(str(floor))) if x and x.get("light")), {})
            if pr is None or not rows or not pf.get("light"):
                if time.time() - job["wall"] < 300.0:      # E2 has not reached this floor yet: try again, bounded
                    time.sleep(10.0)
                    self._anchor_pending.insert(0, job)
                    return self._anchor_write(dict(observation_id=oid, status="WAIT_E2_PRIOR", why="E2 has no beta / map for this floor yet"))
                return self._anchor_write(dict(observation_id=oid, status="SKIPPED", why="no E2 prior within 300 s"))
            t_end = min(rows[-1][1], float(o.get("capture_wall") or rows[-1][1]))
            path_job = job.get("kind") == "PATH"
            win_s = SA.WINDOW_S
            if path_job:                                  # window from the earlier plate's stop (rotation from the path between the plates), 90..240 s
                po = job["prev"]["o"]
                t_prev = float((po.get("odom_pose_at_stop") or {}).get("t") or po.get("capture_wall") or po.get("stopped_wall") or (t_end - SA.WINDOW_S))
                win_s = float(min(SA.WINDOW_S, max(SA.WIN90_S, t_end - t_prev + 20.0)))
            fr = E7.frames(st, t_end, win_s, structural=True)
            if len(fr) < 8:
                return self._anchor_write(dict(observation_id=oid, status="SKIPPED", why=f"only {len(fr)} LiDAR frames before the reading"))
            L = self.tr.light(pf["light"])
            m = self.sf.maps[floor]
            sp = next(s for s in m["spaces"] if s.get("id") == room)
            cor = [c["polygon_px"] for c in m.get("corridors") or [] if c.get("polygon_px")]
            res, _ = EP.floor_res(floor, m)
            ppm0 = 1.0 / res
            accept = None
            if ANCHOR_WALLCHECK or path_job:
                import e3_tracker as E3T
                last, win = SA.path_parts(rows, t_end)
                extra = None
                if path_job:                              # the newer plate must fall within PLATE2_TOL_M of its room under the pinned transform
                    sp_now = [s for s in m["spaces"] if s.get("id") == room]
                    tgt_now = o["target_odom_link_xy"]

                    def extra(rec_, sp_now=sp_now, tgt_now=tgt_now, cor=cor):
                        q = np.array(SA.project(rec_, tgt_now))
                        if S.HALF_CORRIDOR:                   # 2026-11-19: same place rule as the E8 pair check (half corridor across the wall, 3 m along)
                            ok, info = S.place_check(sp_now, cor, q, rec_["ppm"])
                            return ok, (None if ok else f"newer plate off its wall line: {info.get('wall_perp_m')} m across (half corridor {info.get('half_corridor_m')} m), "
                                                         f"{info.get('wall_seg_m', info.get('dist_m'))} m from the wall line")
                        d = min(S._poly_dist_px(s_["polygon_px"], q) for s_ in sp_now) / float(rec_["ppm"])
                        return (d <= SA.PLATE2_TOL_M), (None if d <= SA.PLATE2_TOL_M else f"newer plate {d:.1f} m from its room")
                accept = SA.wall_checker(L, last, win, E3T.wall_pass_any, extra=extra)
            if path_job:
                proom = job["prev"]["region"]["rooms"][0]
                psp = next(s for s in m["spaces"] if s.get("id") == proom)
                wp = SA.plate_wall_points(psp["polygon_px"], cor, ppm0)
                top, _n = SA.scan(E7.cost, fr, L, job["prev"]["o"]["target_odom_link_xy"], wp, ppm0, pr["beta"], pr["mirror"], floor, accept=accept)
            else:
                wp = SA.plate_wall_points(sp["polygon_px"], cor, ppm0)
                top, _n = SA.scan(E7.cost, fr, L, o["target_odom_link_xy"], wp, ppm0, pr["beta"], pr["mirror"], floor, accept=accept)
            stats = dict(SA.LAST_STATS)
            cands = [SA.candidate(t, room, floor, dict(o, label=room), pf.get("H"), pf["light"], pf.get("map_version"), getattr(self.tr, "session", None), pr["merit"], t_end, i + 1)
                     for i, t in enumerate(top)]
            if path_job:                                  # pinned on the earlier plate: its stored position is the transform's anchor; both plates recorded as evidence
                for c in cands:
                    c["pass_id"] = f"AP_{oid}"
                    c["evidence"].update(kind="PATH_ANCHORED", observation_ids=[job["prev"]["o"]["observation_id"], oid], pinned_observation_id=job["prev"]["o"]["observation_id"],
                                         pinned_room=job["prev"]["region"]["rooms"][0], window_s=round(win_s))
            doc = self._anchor_doc()
            doc["candidates"] = (doc.get("candidates") or []) + cands
            doc["runs"] = (doc.get("runs") or []) + [dict(observation_id=oid, kind="PATH" if path_job else "SINGLE", floor=floor, room=room, n_frames=len(fr), window_s=round(win_s),
                                                         sec=round(time.time() - t0, 2), prior=pr, wall_check=accept is not None, scan_stats=stats,
                                                         pinned=(job["prev"]["o"]["observation_id"] if path_job else None),
                                                         top=[c["evidence"] for c in cands], status="ANCHORED_PROVISIONAL", wall=time.time())]
            doc["status"] = "ANCHORED_PROVISIONAL"
            SA.write(self.tr.e2_dir, doc)
        except Exception as ex:                         # noqa: BLE001  a failed anchored fit must not touch the E8 / E7 path
            self._anchor_write(dict(observation_id=oid, status="ERROR", why=f"{type(ex).__name__}: {ex}"))
        finally:
            self._anchor_kick()

    def _anchor_doc(self):
        try:
            return json.loads((Path(self.tr.e2_dir) / "semantic_anchor.json").read_text())
        except (OSError, ValueError):
            return dict(kind="LIVE2_SEMANTIC_ANCHOR", model_status="UNVERIFIED_ANCHOR_MODEL", candidates=[], runs=[])

    def _anchor_write(self, run):
        import semantic_anchor as SA
        doc = self._anchor_doc()
        doc["runs"] = (doc.get("runs") or []) + [dict(run, wall=time.time())]
        doc["status"] = run.get("status")
        SA.write(self.tr.e2_dir, doc)

    # ------------------------------------------------------------------ R34 early E7 SHADOW (record only, never a FIX)
    def early_shadow_tick(self, now=None):
        """loc1 11-01: 379 s with E2 7/9, so E4 never reached CONFIDENT and E7 never ran, although the E3 leader was 1.18 m from the truth at the end. This runs E7 on 30 s windows
        with the current E3 leader, before E4 is confident. RECORD ONLY: it writes e7_shadow_early.json(l), never e7_state, and has no path to a FIX. A leader change clears the
        run; floors E2 has not searched are listed; FAST-LIO below the line holds it."""
        if os.environ.get("LIVE2_E7_EARLY_SHADOW", "0") != "1":   # R36: off by default. 30 s windows separated nothing in the 11-02 runs and took E3's core; e7_long compares the whole path in its own process
            return None
        pol = getattr(self, "struct_policy", None) or {}
        if pol.get("mode") == "OFF" or (self.last or {}).get("global_decision") == "GLOBAL_CONFIRMED":
            return None
        th = getattr(self, "_early_thread", None)
        if (th is not None and th.is_alive()) or (self.job and self.job.is_alive()):
            return None
        rows = getattr(self.tr.store, "rows", None)
        if not rows:
            return None
        t_end = rows[-1][1]
        es = getattr(self, "early", None)
        if es is None:
            es = self.early = dict(kind="LIVE2_E7_EARLY_SHADOW", status="WAITING", leader=None, floor=None, passes=[], n_pass=0, authority="NONE: record only, cannot produce a FIX",
                                   policy_mode=pol.get("mode") or "SHADOW")
        if es["passes"] and t_end - es["passes"][-1]["t_end"] < EARLY_STEP_S:
            return es
        alive = sorted(self.tr.alive(), key=lambda h: h.rank_key())
        leader = next((h for h in alive if h.n_kf >= EARLY_MIN_KF), None)
        if leader is None:
            return es
        if es["leader"] != leader.id:                      # a new leader starts over: earlier windows say nothing about it
            es.update(leader=leader.id, floor=leader.floor, passes=[], n_pass=0, status="RUNNING", leader_changed_wall=time.time(), provenance=leader.rec.get("provenance") or "E2")
        moved = _path(rows, t_end - EARLY_WIN_S, t_end)
        if moved < EARLY_MIN_PATH_M:
            es["status"] = "WAITING_MOTION"
            return es
        if self.fastlio_degraded(30.0)[0]:
            es["status"] = "HELD_FASTLIO"
            return es
        pend = getattr(self.tr, "pending", None)
        if callable(pend) and len(pend()) > EARLY_MAX_KF_PENDING:      # 11-01 loc1 backlog (kf_pending up to 109): E3 goes first
            es["status"] = "HELD_E3_BACKLOG"
            return es
        rivals, seen = [], {leader.floor}
        for h in alive:
            if h.floor not in seen or (h is not leader and len(rivals) < 3):
                rivals.append(h); seen.add(h.floor)
            if len(rivals) >= EARLY_RIVALS:
                break
        cands = [dict(candidate_id=h.id, floor=h.floor, rec=dict(h.rec, light=h.light_dir)) for h in [leader] + rivals]
        self._early_thread = threading.Thread(target=self._early_job, args=(es, cands, t_end, moved), daemon=True)
        self._early_thread.start()
        return es

    def _early_job(self, es, cands, t_end, moved):
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
        except (AttributeError, OSError):
            pass
        t0 = time.time()
        try:
            lights = {c["rec"]["light"]: self.tr.light(c["rec"]["light"]) for c in cands}
            res = E7.evaluate(self.tr.store, cands, lights, t_end, "STRUCTURE_ONLY", window_s=EARLY_WIN_S, structural=True, semantic_ok=None)
        except Exception as ex:                            # noqa: BLE001
            res = dict(final_decision="MORE_OBSERVATION_REQUIRED", why=f"E7 error {type(ex).__name__}: {ex}", results=[])
        unique = (res.get("final_decision") == "GEOMETRY_CONSISTENT_SEMANTIC_PENDING" and not res.get("ambiguity") and not res.get("structurally_indistinguishable"))
        ok = unique and res.get("best") == es["leader"]
        try:
            e2 = json.loads((Path(self.tr.e2_dir) / "e2_state.json").read_text())
        except (OSError, ValueError):
            e2 = {}
        done = e2.get("floors_done") or []
        rec = dict(t_end=t_end, window_s=EARLY_WIN_S, path_m=round(moved, 1), n_frames=res.get("n_frames"), best=res.get("best"), final_decision=res.get("final_decision"), why=res.get("why"),
                   gap_m=res.get("gap_m"), unique=unique, leader_best=ok, rivals=[c["candidate_id"] for c in cands[1:]], e2_floors_done=len(done), e2_floors_total=e2.get("floors_total"),
                   unsearched_floors=[f for f in e2.get("floors") or [] if f not in done], sec=round(time.time() - t0, 2), wall=time.time())
        es["passes"] = (es["passes"] + [rec])[-12:]
        es["n_pass"] = int(es.get("n_pass") or 0) + 1
        run = 0
        for p in reversed(es["passes"]):
            if not p["leader_best"]:
                break
            run += 1
        es.update(leader_best_run=run, status="SHADOW_WOULD_CONFIRM" if run >= 2 and not rec["unsearched_floors"] else "RUNNING", last=rec)
        tmp = self.state / f".e7_shadow_early.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(dict(es, written_wall=time.time()), default=str))
        os.replace(tmp, self.state / "e7_shadow_early.json")
        with open(self.state / "e7_shadow_early.jsonl", "a") as f:
            f.write(json.dumps(dict(rec, leader=es["leader"], floor=es["floor"], status=es["status"]), default=str) + "\n")

    def _single_write(self, rec):
        tmp = self.state / f".semantic_recovery_single.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(dict(rec, written_wall=time.time()), default=str))
        os.replace(tmp, self.state / "semantic_recovery_single.json")
        with open(self.state / "semantic_recovery_single.jsonl", "a") as f:
            f.write(json.dumps(dict(rec, written_wall=time.time()), default=str) + "\n")

    def _recovery_write(self):
        tmp = self.state / f".semantic_recovery.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(dict(self.recovery, written_wall=time.time()), default=str))
        os.replace(tmp, self.state / "semantic_recovery.json")
        with open(self.state / "semantic_recovery.jsonl", "a") as f:
            f.write(json.dumps(dict(self.recovery, written_wall=time.time()), default=str) + "\n")

    # ------------------------------------------------------------------ PATH A (structure only)
    def structure_tick(self, e4, now, stop=None):
        """Called by e34 every tick with the E4 result. 2026-10-30 21:xx (policy mode OFF / SHADOW / ENABLED, e7_policy):
        when E4 reaches CONFIDENT_GEOMETRY on a leader,
          1. RETRO: every stored STRUCT_WIN_S window of the last RETRO_HORIZON_S (ends on a fixed STRUCT_STEP_S grid back from
             the CONFIDENT time, >= STRUCT_MIN_PATH_M m of motion, no position jump, NOT overlapping an E2 input window) is
             re-checked with the CURRENT leader transform + E3 rivals + the complete E2 rival list (worker processes on the idle
             E2 cores). Nothing is cached: a new leader restarts everything.
          2. LIVE: new windows every >= STRUCT_STEP_S while the robot moves (a still robot consumes no window).
        FIX rule (unchanged E7 thresholds): an unbroken run of passing windows (best == leader, unique) whose first and last
        windows do not overlap, the run's end <= RETRO_MAX_AGE_S old, AND at least one LIVE window (after CONFIDENT) with
        best == leader (the current geometry still agrees), AND no window anywhere unique for another candidate, same
        odometry epoch, FAST-LIO >= 8 Hz. SHADOW records 'SHADOW_WOULD_CONFIRM' and never writes a FIX. `stop` is unused."""
        pol = getattr(self, "struct_policy", None) or {}
        mode = pol.get("mode") or ("ENABLED" if pol.get("enabled") else "OFF")
        if mode == "OFF" or (self.last or {}).get("global_decision") == "GLOBAL_CONFIRMED":
            return None
        m = (e4 or {}).get("metrics") or {}
        sa = getattr(self, "struct", None)
        if (e4 or {}).get("state") != "CONFIDENT_GEOMETRY":
            if sa and sa.get("status") == "RUNNING":
                if self.job and self.job.is_alive():
                    return None                 # 2026-11-01: keep the attempt; it is ended at the first tick after the job
                self._struct_end(sa, "NOT_CONFIRMED", f"E4 left CONFIDENT_GEOMETRY ({(e4 or {}).get('state')})")
            self.struct = None if not sa or sa.get("status") == "RUNNING" else sa
            return None
        leader = m.get("best_id")
        if mode == "ENABLED" and any(h.id == leader and h.rec.get("provenance") == "SEMANTIC_ANCHORED" for h in self.tr.alive()):
            mode = "SHADOW"                                    # R34: a leader that exists only through one reading never fixes on structure alone
        if sa and sa.get("leader") == leader and sa.get("status") != "RUNNING":
            return sa                                          # this leader already tried: no retry loop
        if self.job and self.job.is_alive():
            return sa
        rows = getattr(self.tr.store, "rows", None)
        t_end = rows[-1][1] if rows else None
        if not sa or sa.get("leader") != leader:
            if sa and sa.get("status") == "RUNNING":
                self._struct_end(sa, "NOT_CONFIRMED", f"E4 leader changed to {leader}: earlier windows invalidated")
            if t_end is None:
                return None
            sa = self.struct = dict(kind="LIVE2_STRUCTURE_PATH", leader=leader, floor=m.get("best_floor"), status="RUNNING", mode=mode,
                                    phase="RETRO_VERIFY", event_id=f"STRUCT_{leader}_{int(now)}", started_wall=now, t_confident=t_end,
                                    passes=[], epoch=self._epoch(), e4_gap=m.get("gap"), need_path_m=STRUCT_MIN_PATH_M,
                                    win_s=STRUCT_WIN_S, max_windows=STRUCT_MAX_WINDOWS)
            self._struct_write(sa)
        if sa.get("status") != "RUNNING" or t_end is None:
            return sa
        if now - sa["started_wall"] > STRUCT_BUDGET_S:
            self._struct_end(sa, "NOT_CONFIRMED", f"verification budget {STRUCT_BUDGET_S:.0f} s exceeded ({len(sa['passes'])} windows)")
            return sa
        if self._epoch() != sa["epoch"]:
            self._struct_end(sa, "NOT_CONFIRMED", "ODOM_SESSION_CHANGED during structure verification")
            return sa
        cons = [h for h in self.tr.alive() if h.id == leader]
        if not cons:
            self._struct_end(sa, "NOT_CONFIRMED", f"leader {leader} no longer alive")
            return sa
        rivals = [h for h in sorted(self.tr.alive(), key=lambda h: h.rank_key()) if h.id != leader][:E7_RIVALS]
        cands = [dict(candidate_id=h.id, floor=h.floor, rec=dict(h.rec, light=h.light_dir)) for h in cons + rivals]
        if sa["phase"] == "RETRO_VERIFY":
            self.job = threading.Thread(target=self._run_retro, args=(sa, cands), daemon=True)
            self.job.start()
            return sa
        # LIVE windows on a fixed grid after CONFIDENT (t_confident + k * STRUCT_STEP_S): windows that ended while the retro
        # job ran are caught up from the store; a window at rest is recorded as skipped (no E7, breaks a run)
        last_end = max((p["t_end"] for p in sa["passes"] if p.get("t_end") is not None), default=sa["t_confident"])
        te = max(last_end, sa["t_confident"]) + STRUCT_STEP_S
        n0 = len(sa["passes"])
        while te <= t_end:
            moved = _path(rows, te - STRUCT_WIN_S, te)
            if moved >= STRUCT_MIN_PATH_M:
                sa.pop("waiting_motion", None)
                self.job = threading.Thread(target=self._run_structure, args=(sa, cands, te, round(moved, 2)), daemon=True)
                self.job.start()
                return sa
            sa["passes"].append(dict(t_end=te, ok=False, skipped=f"AT_REST ({moved:.1f} m < {STRUCT_MIN_PATH_M:.0f} m)",
                                     path_m=round(moved, 2), window_s=STRUCT_WIN_S, live=True, wall=time.time()))
            sa["waiting_motion"] = round(moved, 2)
            te += STRUCT_STEP_S
        if len(sa["passes"]) != n0:
            self._struct_write(sa)
        return sa

    def _e2_for_rivals(self):
        """(e2_state dict, why_unusable): the E2 list must exist and belong to this odometry session (fail closed)."""
        try:
            e2 = json.loads((Path(self.tr.e2_dir) / "e2_state.json").read_text()) if getattr(self.tr, "e2_dir", None) else None
        except (OSError, ValueError) as ex:
            return None, f"E2 state unreadable: {type(ex).__name__}"
        if not e2:
            return None, "no E2 state"
        if e2.get("odom_session") != getattr(self.tr, "session", None):
            return None, f"E2 state from another odometry session ({e2.get('odom_session')})"
        return e2, None

    def _window_skip(self, rows, t_end, e2_wins):
        moved = _path(rows, t_end - STRUCT_WIN_S, t_end)
        if moved < STRUCT_MIN_PATH_M:
            return moved, f"AT_REST ({moved:.1f} m < {STRUCT_MIN_PATH_M:.0f} m)"
        if _max_step(rows, t_end - STRUCT_WIN_S, t_end) > JUMP_STEP_M:
            return moved, "POSITION_JUMP in the window"
        for a, b in e2_wins:
            if t_end - STRUCT_WIN_S < b and t_end > a:
                return moved, f"E2_INPUT_OVERLAP ({a:.1f}..{b:.1f})"
        return moved, None

    def _run_retro(self, sa, cands):
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
        except (AttributeError, OSError):
            pass
        t0 = time.time()
        e2, e2_why = self._e2_for_rivals()
        if e2_why:
            return self._struct_end(sa, "NOT_CONFIRMED", f"E2 rival list unavailable ({e2_why}) -> no structure FIX")
        rows = list(self.tr.store.rows)
        tc, first = sa["t_confident"], rows[0][1]
        e2_wins = e2_input_windows(self.tr.e2_dir, e2)
        ends, k = [], 0
        while True:
            te = tc - k * STRUCT_STEP_S
            if te - STRUCT_WIN_S < max(first, tc - RETRO_HORIZON_S) - 1e-6:
                break
            ends.append(te); k += 1
        ends.sort()
        jobs, skipped = [], []
        for te in ends:
            moved, why = self._window_skip(rows, te, e2_wins)
            if why:
                skipped.append(dict(t_end=te, ok=False, skipped=why, path_m=round(moved, 2), retro=True, window_s=STRUCT_WIN_S))
            else:
                jobs.append(dict(store=str(self.tr.store.root), e2_state=str(Path(self.tr.e2_dir) / "e2_state.json"), cands=cands,
                                 t_end=te, win=STRUCT_WIN_S, leader_floor=sa.get("floor"), path_m=round(moved, 2)))
        sa.update(retro=dict(n_windows=len(ends), n_eval=len(jobs), n_skipped=len(skipped), e2_input_windows=e2_wins,
                             horizon_s=RETRO_HORIZON_S, started_wall=time.time()))
        self._struct_write(sa)
        out = R7.evaluate_windows(jobs, procs=RETRO_PROCS, cores=RETRO_CORES)
        passes = list(skipped)
        for j, r in zip(jobs, out):
            res = r["res"]
            unique = (res.get("final_decision") == "GEOMETRY_CONSISTENT_SEMANTIC_PENDING" and not res.get("ambiguity")
                      and not res.get("structurally_indistinguishable"))
            passes.append(dict(t_end=j["t_end"], ok=unique and res.get("best") == sa["leader"], unique=unique, best=res.get("best"),
                               final_decision=res.get("final_decision"), why=res.get("why"), gap_m=res.get("gap_m"),
                               n_frames=res.get("n_frames"), e7_s=r["e7_s"], e2_rivals=r["e2_rivals"], prescreen_s=r["prescreen_s"],
                               window_s=STRUCT_WIN_S, path_m=j["path_m"], retro=True, wall=time.time(),
                               leader_correction=r.get("leader_correction")))
        sa["passes"] = sorted(passes, key=lambda p: p["t_end"])
        sa["retro"].update(took_s=round(time.time() - t0, 1), n_pass=sum(1 for p in passes if p.get("ok")))
        sa["phase"] = "VERIFY_WHILE_MOVING"
        return self._decide(sa, None)

    def _run_structure(self, sa, cands, t_end, moved):
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
        except (AttributeError, OSError):
            pass
        t0 = time.time()
        win = STRUCT_WIN_S
        e2, e2_why = self._e2_for_rivals()
        extra, pre_s = [], 0.0
        if not e2_why:
            try:
                extra, pre_s = e2_rivals(e2, self.tr.store, t_end, sa.get("floor"), self.tr.light, window_s=win)
            except (OSError, ValueError, KeyError, TypeError) as ex:
                e2_why = f"E2 rival pre-screen failed: {type(ex).__name__}: {ex}"
        if e2_why:                                                              # fail closed: never FIX against E3 rivals only
            sa["passes"].append(dict(t_end=t_end, ok=False, why=e2_why, wall=time.time(), window_s=win))
            return self._struct_end(sa, "NOT_CONFIRMED", f"E2 rival list unavailable ({e2_why}) -> no structure FIX")
        skip = None
        if _max_step(self.tr.store.rows, t_end - win, t_end) > JUMP_STEP_M:
            skip = "POSITION_JUMP in the window"
        if skip:
            sa["passes"].append(dict(t_end=t_end, ok=False, skipped=skip, path_m=moved, window_s=win, wall=time.time()))
            return self._decide(sa, None)
        cands = cands + [{k: x[k] for k in ("candidate_id", "floor", "rec")} for x in extra]
        lights = {c["rec"]["light"]: self.tr.light(c["rec"]["light"]) for c in cands}
        try:
            res = E7.evaluate(self.tr.store, cands, lights, t_end, "STRUCTURE_ONLY", window_s=win, structural=True, semantic_ok=None)
        except Exception as ex:                                                 # noqa: BLE001
            res = dict(final_decision="MORE_OBSERVATION_REQUIRED", why=f"E7 error {type(ex).__name__}: {ex}", results=[])
        unique = (res.get("final_decision") == "GEOMETRY_CONSISTENT_SEMANTIC_PENDING" and not res.get("ambiguity")
                  and not res.get("structurally_indistinguishable"))
        sa["passes"].append(dict(t_end=t_end, ok=unique and res.get("best") == sa["leader"], unique=unique, best=res.get("best"),
                                 final_decision=res.get("final_decision"), why=res.get("why"), gap_m=res.get("gap_m"),
                                 n_frames=res.get("n_frames"), e7_s=round(time.time() - t0, 1), wall=time.time(),
                                 e2_rivals=[(x["candidate_id"], x["floor"], x["prescreen"]) for x in extra], prescreen_s=pre_s,
                                 window_s=win, path_m=moved, live=True,
                                 leader_correction=next((r.get("correction") for r in res.get("results") or []
                                                         if r.get("candidate_id") == sa["leader"]), None)))
        sa["last_e7"] = res
        return self._decide(sa, res)

    def _decide(self, sa, res):
        P = sorted(sa["passes"], key=lambda p: p["t_end"])
        contra = [p for p in P if p.get("unique") and p.get("best") != sa["leader"]]
        if contra:
            c = contra[0]
            return self._struct_end(sa, "NOT_CONFIRMED", f"window ending {c['t_end'] - sa['t_confident']:+.0f} s (vs CONFIDENT): geometry "
                                                         f"unique for {c['best']}, not the E4 leader {sa['leader']} (contradiction)")
        runs, first, prev = [], None, None
        for p in P:
            if p.get("ok") and prev is not None and prev.get("ok") and first is not None and p["t_end"] - prev["t_end"] <= STRUCT_RUN_GAP_S:
                pass
            elif p.get("ok"):
                first = p["t_end"]
            else:
                first = None
            if p.get("ok"):
                runs.append((first, p["t_end"]))
            prev = p
        t_now = P[-1]["t_end"] if P else sa["t_confident"]
        good = [(a, b) for a, b in runs if b - a >= STRUCT_WIN_S and t_now - b <= RETRO_MAX_AGE_S]
        live_agree = [p for p in P if p.get("live") and p.get("best") == sa["leader"]]
        sa["evidence"] = dict(run=list(good[-1]) if good else None, live_agree=len(live_agree), n_eval=sum(1 for p in P if "best" in p))
        n_eval = sa["evidence"]["n_eval"]
        if not (good and live_agree):
            if n_eval >= STRUCT_MAX_WINDOWS:
                return self._struct_end(sa, "NOT_CONFIRMED", f"{n_eval} windows: " + ("no passing run over 2 disjoint windows" if not good
                                                                                      else "no live window agreeing with the leader"))
            sa["need"] = ("a passing run over 2 disjoint 30 s windows (structurally distinctive motion)" if not good
                          else "one more moving 30 s window after CONFIDENT whose best is the leader")
            return self._struct_write(sa)
        if self._epoch() != sa["epoch"]:
            return self._struct_end(sa, "NOT_CONFIRMED", "ODOM_SESSION_CHANGED between E7 windows")
        bad, why_lio = self.fastlio_degraded()
        if bad:
            return self._struct_end(sa, "NOT_CONFIRMED", why_lio)
        a_, b_ = good[-1]
        why = (f"passing run {a_ - sa['t_confident']:+.0f}..{b_ - sa['t_confident']:+.0f} s vs CONFIDENT (first and last 30 s windows "
               f"disjoint, every window in between passes, E2 input windows excluded) + {len(live_agree)} live window(s) agreeing; "
               f"no contradicting window; complete E2 rivals")
        if sa.get("mode") != "ENABLED":
            sa["would_confirm_wall"] = time.time()
            with open(self.state / "e7_shadow.jsonl", "a") as f:
                f.write(json.dumps(dict(kind="LIVE2_E7_SHADOW", leader=sa["leader"], floor=sa["floor"], why=why, wall=time.time(),
                                        t_confident=sa["t_confident"], evidence=sa["evidence"]), default=str) + "\n")
            return self._struct_end(sa, "SHADOW_WOULD_CONFIRM", "SHADOW (FIX output disabled by policy): " + why)
        self._struct_end(sa, "CONFIRMED", why)
        out = dict(kind="LIVE2_E7_STATE", observation_id=f"STRUCT_{sa['leader']}", source="STRUCTURE_ONLY", path="STRUCTURE_ONLY",
                   semantic_hypothesis=sa["leader"], e7=res or sa.get("last_e7"), global_decision="GLOBAL_CONFIRMED",
                   why="STRUCTURE_ONLY: E4 CONFIDENT_GEOMETRY + " + why,
                   e7_s=sum(x.get("e7_s") or 0 for x in P), written_wall=time.time(), used_observation_ids=[],
                   structure_path={k: sa.get(k) for k in ("leader", "floor", "passes", "started_wall", "e4_gap", "t_confident", "evidence")},
                   fix_blocked=None)
        self.last = out
        tmp = self.state / f".e7_state.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(out, default=str))
        os.replace(tmp, self.state / "e7_state.json")
        with open(self.state / "e7_history.jsonl", "a") as f:
            f.write(json.dumps(dict({k: out[k] for k in out if k != "e7"}), default=str) + "\n")
        return out

    def _struct_end(self, sa, status, why):
        sa.update(status=status, phase=status, why=why, ended_wall=time.time())
        return self._struct_write(sa)

    def _struct_write(self, sa):
        rec = {k: v for k, v in sa.items() if k != "last_e7"}
        tmp = self.state / f".structure_path.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(rec, default=str))
        os.replace(tmp, self.state / "structure_path.json")
        with open(self.state / "structure_path.jsonl", "a") as f:
            f.write(json.dumps(dict(rec, wall=time.time()), default=str) + "\n")
        return sa

    def observation_outcome(self, r, outcome, why):
        """2026-10-22: why THIS observation did (not) reach a FIX, for the tablet — e7_state.json keeps the last E7 run (a
        confirmed FIX stays visible); <state>/e7_observation.json always describes the newest processed observation."""
        out = dict(kind="LIVE2_E7_OBSERVATION", observation_id=r.get("observation_id"), source=r.get("source"),
                   e6_status=r.get("status"), outcome=outcome, why=why, written_wall=time.time())
        tmp = self.state / f".e7_observation.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(out, default=str))
        os.replace(tmp, self.state / "e7_observation.json")
        return out

    def candidate_missing(self, r, res):
        rooms = sorted({(v.get("map_room_id"), v.get("verdict")) for v in (res.get("per_hypothesis") or {}).values()}, key=str)[:8]
        out = dict(kind="LIVE2_E7_STATE", observation_id=r.get("observation_id"), source=r.get("source"), semantic_hypothesis=None,
                   e7=None, global_decision="MORE_OBSERVATION_REQUIRED",
                   why=f"CANDIDATE_MISSING: decisive reading {[a[0] for a in res.get('alternatives') or []][:3]} matches no live "
                       f"hypothesis (floor likelihood {res.get('floor_likelihood')}); E7 not run — re-search / more observation",
                   candidate_gap=dict(rooms_checked=rooms, n_alive=len(self.tr.alive())), written_wall=time.time())
        self.last = out
        tmp = self.state / f".e7_state.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(out, default=str))
        os.replace(tmp, self.state / "e7_state.json")
        with open(self.state / "e6_candidate_gaps.jsonl", "a") as f:
            f.write(json.dumps(out, default=str) + "\n")
        self.observation_outcome(r, "CANDIDATE_MISSING", out["why"])

    def fastlio_degraded(self, window_s=120.0, line_hz=8.0):
        """(bool, why): FAST-LIO below its rate line inside the E7 window. 2026-10-22 N1: at < 8 Hz FAST-LIO's state degraded
        (gravity estimate 15.5 -> 11.2 deg, height wandering); E7 registers the same odom_link samples, so no FIX on them."""
        try:
            now = time.time()
            rec = [s for s in (json.loads(x) for x in (self.state / "lio_series.jsonl").read_text().splitlines()[-400:]) if now - float(s.get("wall", 0)) <= window_s]
            low = [s for s in rec if (s.get("hz") or 99) < line_hz]
        except (OSError, ValueError):
            return False, None
        # R41: the 2026-10-22 failure was a sustained rate loss. One short dip (X20261103_001810 +583 s: 7.78 Hz, 3 samples of 120, E2 holds at once) must not void a place that two disjoint windows agree on. Sustained (>= 10 % of the window) or deep (< 6 Hz) still blocks.
        if len(low) >= max(5, 0.10 * len(rec)) or any((s.get("hz") or 99) < 6.0 for s in low):
            return True, f"FAST-LIO below {line_hz:.0f} Hz in {len(low)} samples of the last {window_s:.0f} s"
        return False, None

    def request_e7(self, cons, r):
        if (self.last or {}).get("global_decision") == "GLOBAL_CONFIRMED":      # R41: a FIX stands. Replay X20261103_001810: a look queued behind the deciding one ran after it and its AMBIGUOUS replaced the FIX record
            return
        cons = sorted(cons, key=lambda h: (bool(getattr(h, "e8_path_down", False)), h.rank_key()))    # R37: 11-02 tested 'the first' of the E8 candidates (H174, not converged) while its neighbour H177 fitted best. B1 (2026-11-18): a candidate whose path crosses a wall comes last
        r = dict(r, consistent_ids=[h.id for h in cons])
        rivals = [h for h in sorted(self.tr.alive(), key=lambda h: h.rank_key()) if h not in cons][:E7_RIVALS]
        if (getattr(self, "_e7_ledger", None) or {}).get("gid") == r.get("observation_id"):       # R41: a later window asks 'the same place again?'. It keeps rivals, fewer: one look took 50-160 s beside SigLIP and E2 (2.5 s per registration on a free core)
            cons, rivals = cons[:E7_PLATE_CANDS], rivals[:1]
        if r.get("source") == "E8_PAIR" and os.environ.get("LIVE2_E7_FAST", "1") == "1":
            # 2026-11-16 (user directive, with FUSED_FLOOR): every E8 pair look uses the plate-consistent candidates <= 2 and ONE rival, no E2 rival list.
            # The fused floor rule gives floors E7 did not register their best possible geometry, so rivals do not change its decision; they only cost time
            # (11-16 field: 56-118 s per look, 9 of 12 ready pairs ended before E7). LIVE2_E7_FAST=0 restores cons[:3] + rivals[:3] + E2 rivals.
            cons, rivals = cons[:E7_PLATE_CANDS], rivals[:1]                  # 2026-11-18: 2 -> E7_PLATE_CANDS (4)
            r = dict(r, e7_fast=True)
        cands = [dict(candidate_id=h.id, floor=h.floor, rec=dict(h.rec, light=h.light_dir)) for h in (cons if r.get("e7_fast") else cons[:3]) + rivals]
        with self.lock:
            if self.job and self.job.is_alive():
                self.pending = (cands, r, cons[0].id)
                return
            self.pending = None
        self.job = threading.Thread(target=self._run, args=(cands, r, cons[0].id), daemon=True)
        self.job.start()

    def _run(self, cands, r, sem_id):
        if (self.last or {}).get("global_decision") == "GLOBAL_CONFIRMED":
            return
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 5)      # this thread only. R37: nice 19 on E3's core took 124 s in the field. R41: 10 -> 5, ahead of the E2 children (nice 10) on these cores: replay 31-56 s per look behind 3 children, and the 60 s age gate stands behind it
            os.sched_setaffinity(threading.get_native_id(), {6, 7})
        except (AttributeError, OSError):
            pass
        t0 = time.time()
        sess0 = self._epoch()                                                   # odometry epoch the reading and geometry belong to
        self.tr.store.refresh()
        t_end = self.tr.store.rows[-1][1] if self.tr.store.rows else None
        # 2026-10-30 (staging): the same E2 rival completion as PATH A (a misread label must not confirm a floor whose closest
        # other-floor geometry E3 merged away); no usable E2 list -> this E7 can never confirm (fail closed)
        lead_floor = next((c["floor"] for c in cands if c["candidate_id"] == sem_id), None)
        extra, e2_why = [], None
        try:
            e2 = json.loads((Path(self.tr.e2_dir) / "e2_state.json").read_text()) if getattr(self.tr, "e2_dir", None) else None
            if not e2:
                e2_why = "no E2 state"
            elif e2.get("odom_session") != getattr(self.tr, "session", None):
                e2_why = f"E2 state from another odometry session ({e2.get('odom_session')})"
            else:
                later = (getattr(self, "_e7_ledger", None) or {}).get("gid") == r.get("observation_id")
                extra, _ = e2_rivals(e2, self.tr.store, t_end, lead_floor, self.tr.light, k=0 if r.get("e7_fast") else (2 if later else E7_E2_RIVALS))
        except (OSError, ValueError, KeyError, TypeError) as ex:
            e2_why = f"E2 rival pre-screen failed: {type(ex).__name__}: {ex}"
        cands = cands + [{k: x[k] for k in ("candidate_id", "floor", "rec")} for x in extra]
        single = r.get("source") == "E8_SINGLE"
        if single:                                                              # B3: the full look only when geometry can separate floors
            try:
                go = self._single_go(cands, r, t_end)
            except Exception as ex:                                             # noqa: BLE001
                go = False
                self._jsonl("e7_single_plate.jsonl", dict(kind="LIVE2_SINGLE_PLATE", wall=time.time(), observation_id=(r.get("single_plate") or {}).get("observation_id"), step="PRESCREEN", go=False, why=f"error {type(ex).__name__}: {ex}"))
            if not go:
                with self.lock:
                    nxt, self.pending = getattr(self, "pending", None), None
                if nxt:
                    self._run(*nxt)
                return
        win_s = 90.0                                                            # R41: after a first look, only frames that look has not used: the windows of one group are disjoint
        led = getattr(self, "_e7_ledger", None)
        if led and led.get("gid") == r.get("observation_id") and led.get("t_last") and t_end:
            win_s = max(E7_WIN_MIN_S, min(90.0, float(t_end) - float(led["t_last"])))
        res, proc_rec = None, None
        if os.environ.get("LIVE2_E7_PROC", "1") == "1":
            # 2026-11-17 (user directive "E7 을 e34 와 다른 프로세스로 떼어 내줘, 백업 꼭"): the continuation + evaluate below run in their own process
            # (e7_proc.py, taskset 0,1,6,7, nice 5 — like the e7_long worker). This thread waits without the GIL. 11-16 field: 56-118 s per look,
            # ~15 s per registration inside e34 vs 2.5 s on a free core. Same code, same inputs; no result from the process -> computed here as before.
            # LIVE2_E7_PROC=0 restores the in-thread computation.
            try:
                import e7_proc as EP7
                got, proc_rec = EP7.run(self.state, self.tr.store, cands, r, t_end, win_s, E7)
                if got is not None:
                    cands, res = got["cands"], got["res"]
            except Exception as ex:                                             # noqa: BLE001  the in-thread computation below runs as before
                proc_rec = dict(mode="THREAD_FALLBACK", error=f"{type(ex).__name__}: {ex}")
        if res is None:
            lights = {c["rec"]["light"]: self.tr.light(c["rec"]["light"]) for c in cands}
            try:                                                                    # R40: the candidates E8 accepted get the same continuation from their search bound as the fair floor comparison (e7_long), or a right place 1.3 m off reads 'not converged' (X20261103_001810)
                import e7_long as EL_
                fr_ = E7.frames(self.tr.store, t_end, 90.0, structural=True)
                for c in cands:
                    if c["candidate_id"] in (r.get("consistent_ids") or []) and len(fr_) >= 2:
                        for _ in range(2):
                            rg = E7.register(lights[c["rec"]["light"]], c["rec"], fr_)
                            if not rg.get("on_bound"):
                                break
                            c["rec"] = EL_.recentered(c["rec"], fr_, rg["correction"])
            except Exception:                                                       # noqa: BLE001  the evaluation below decides as before
                pass
            try:
                res = E7.evaluate(self.tr.store, cands, lights, t_end, "E6_OPERATOR_ASSISTED" if r.get("source") == "OPERATOR_ASSISTED"
                                  else "E6_SEMANTIC", window_s=win_s, structural=True, semantic_ok=True)
            except Exception as ex:                                                 # noqa: BLE001
                res = dict(final_decision="MORE_OBSERVATION_REQUIRED", why=f"E7 error {type(ex).__name__}: {ex}", results=[])
        if proc_rec is not None:
            res["e7_proc"] = {k: proc_rec.get(k) for k in ("mode", "wall_s", "worker_s", "n_register", "error")}
        if res.get("best") in (r.get("consistent_ids") or []):       # the best fit is one of the hypotheses E8 accepted: judge that one
            sem_id = res["best"]
        dec, why = global_decision(res, sem_id)
        e7_dec = dec                                                            # E7's own answer before the FIX gates
        s_cls, s_why = (structural_class(res, sem_id) if res.get("results") else (None, res.get("why")))
        res["structural_class"], res["structural_why"] = s_cls, s_why
        res["e2_rivals"] = [(x["candidate_id"], x["floor"], x["prescreen"]) for x in extra]
        if dec == "GLOBAL_CONFIRMED" and e2_why:
            dec, why = "MORE_OBSERVATION_REQUIRED", f"E7 geometry fits {sem_id}, but the E2 rival list is unavailable ({e2_why}): not a fix"
        if dec == "GLOBAL_CONFIRMED" and not r.get("pair_observation_ids"):
            dec, why = "PROVISIONAL_SINGLE", f"E7 geometry fits {sem_id}, but from ONE observation: E8 needs a consistent pair"
        if dec == "GLOBAL_CONFIRMED":
            bad, why_lio = self.fastlio_degraded()
            if bad:
                dec, why = "MORE_OBSERVATION_REQUIRED", f"E7 geometry would confirm, but {why_lio}: not a fix — observe again"
        if sess0 is not None and self._epoch() != sess0:
            dec, why = "MORE_OBSERVATION_REQUIRED", f"ODOM_SESSION_CHANGED during E7 ({sess0} -> new): stale geometry is never a fix"
        out = dict(kind="LIVE2_E7_STATE", observation_id=r.get("observation_id"), source=r.get("source"), semantic_hypothesis=sem_id,
                   e7=res, global_decision=dec, why=why, structural_class=s_cls, structural_why=s_why, e7_s=round(time.time() - t0, 1), written_wall=time.time(),
                   used_observation_ids=r.get("pair_observation_ids") or [r.get("observation_id")],
                   fix_blocked=None if dec == e7_dec else dict(at_wall=time.time(), e7_decision=e7_dec, why=why))
        rec = None if single else self._semantic_shadow(res, sem_id, dec)     # B3: a single-plate look never writes the pair's shadow record (retry_e7 reads it)
        out = self._semantic_fix(out, rec, r, e2_why, sess0, cands=cands, res=res)
        try:                                   # 2026-11-16 (user directive): floor confirmation by the fused geometry + plate score (fused_floor.py) is the main FIX rule.
            import fused_floor as FF           # path-based wall crossing, room-number floor evidence; LIVE2_FUSED_FLOOR=0 turns it off. A failure here keeps the decision above.
            fo = FF.apply(self, r, cands, res, out, E7, sess0)
            if fo:
                out = fo
        except Exception as ex:                # noqa: BLE001
            print(f"[e6e7] fused floor: {type(ex).__name__}: {ex}", flush=True)
            try:                               # e34 stdout is not archived: the failure goes to the same record file
                with open(self.state / "e7_fused_floor.jsonl", "a") as f:
                    f.write(json.dumps(dict(kind="LIVE2_FUSED_FLOOR", wall=time.time(), observation_id=r.get("observation_id"), confirmed=False, error=f"{type(ex).__name__}: {ex}")) + "\n")
            except OSError:
                pass
        if single:
            try:
                self._single_after(r, out)
            except Exception as ex:                                             # noqa: BLE001
                print(f"[e6e7] single plate after E7: {type(ex).__name__}: {ex}", flush=True)
        dec, why = out["global_decision"], out["why"]
        self.last = out
        tmp = self.state / f".e7_state.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(out, default=str))
        os.replace(tmp, self.state / "e7_state.json")
        self.observation_outcome(r, dec, why)
        if self.pair is not None:
            self.pair.on_e7(out)
            self.pair.snapshot()
        with open(self.state / "e7_history.jsonl", "a") as f:
            f.write(json.dumps(dict({k: out[k] for k in out if k != "e7"}, best=res.get("best"), gap=res.get("gap_m")), default=str) + "\n")
        with self.lock:
            nxt, self.pending = getattr(self, "pending", None), None
        if nxt:
            self._run(*nxt)

    def _semantic_shadow(self, res, sem_id, dec):
        """R35: record what the structure + semantic rule would decide. No path to e7_state or a FIX. A failure here never touches E7."""
        pair = getattr(self, "pair", None)
        if pair is None:
            return
        try:
            import e7_semantic_shadow as SH
            import e8_pair as EP
            maps = getattr(self.sf, "maps", None) or {}
            scale = {}
            for f, m in maps.items():
                r_, _ = EP.floor_res(f, m)
                if r_:
                    scale[f] = 1.0 / r_
            try:
                long_state = json.loads((self.state / "e7_long.json").read_text())
            except (OSError, ValueError):
                long_state = None
            rec = SH.evaluate(pair.last_eval, list(pair.group), res, sem_id, dec, E7.GAP_M, E7.XW_VETO, scale, maps=maps, long_state=long_state, experiment_id=getattr(self.tr, "experiment_id", None) or self._experiment_id())
            SH.write(self.state, rec)
            return rec
        except Exception as ex:                                                 # noqa: BLE001
            try:
                with open(self.state / "e7_semantic_shadow.jsonl", "a") as f:
                    f.write(json.dumps(dict(kind="LIVE2_E7_SEMANTIC_SHADOW", verdict="SHADOW_ERROR", error=f"{type(ex).__name__}: {ex}", wall=time.time())) + "\n")
            except OSError:
                pass

    def _epoch(self):
        sess = getattr(self.tr.store, "session", None)
        return (sess() or {}).get("odom_session_id") if callable(sess) else None

    max_odom_age_s = 60.0
    min_windows = MIN_WINDOWS

    def _experiment_id(self):
        try:
            return json.loads((self.state.parent / "ctl" / "experiment.json").read_text()).get("exp_id")
        except (OSError, ValueError):
            return None

    def write_structure_long_fix(self, r, why):
        """R37: final_verify passed under an ENABLED policy. The hypothesis on the winning floor nearest the winning pose must be live: the FIX is about something E3 tracks. -> (ok, why)"""
        best = r["floors"][0]
        cons = [h for h in sorted(self.tr.alive(), key=lambda h: h.rank_key()) if h.floor == best["floor"]]
        if not cons:
            return False, f"no live hypothesis on {best['floor']}F"
        h = next((x for x in cons if x.id == best.get("hyp")), cons[0])
        bad, why_lio = self.fastlio_degraded()
        if bad:
            return False, why_lio
        cand = dict(candidate_id=h.id, floor=h.floor, rec=dict(h.rec, light=h.light_dir))
        res = E7.evaluate(self.tr.store, [cand], {h.light_dir: self.tr.light(h.light_dir)}, r["t_end"], "STRUCTURE_ONLY", window_s=max(30.0, r["t_end"] - self.tr.store.rows[0][1]), structural=True, semantic_ok=None)
        res["long_structure"] = {k: r.get(k) for k in ("floors", "path_m", "n_frames", "floors_exact", "n_refresh")}
        if res.get("final_decision") != "GEOMETRY_CONSISTENT_SEMANTIC_PENDING":    # R42: the pose this FIX shows must come from a converged, locally unique registration without a wall contradiction
            return False, f"the FIX pose registration did not pass: {res.get('final_decision')} ({res.get('why')})"
        try:                                                                    # R42: the current pose after a structure FIX comes from THIS registration (R41 had none, or an earlier semantic group's stale transform)
            import e7_long as EL_
            s0 = next(q for q in res.get("results") or [] if q.get("candidate_id") == h.id)
            fr = E7.frames(self.tr.store, r["t_end"], 30.0, structural=True)
            rec_fix = EL_.recentered(dict(cand["rec"]), fr, s0["correction"]) if fr else dict(cand["rec"])
        except Exception as ex:                                                 # noqa: BLE001
            return False, f"no transform for the FIX pose: {type(ex).__name__}: {ex}"
        self._e7_ledger = dict(gid=f"STRUCT_LONG_{h.id}", passes=[], t_last=r["t_end"], history=[], rec_fix=dict(rec_fix, floor=h.floor))
        out = dict(kind="LIVE2_E7_STATE", observation_id=f"STRUCT_LONG_{h.id}", source="STRUCTURE_ONLY", path="STRUCTURE_LONG", semantic_hypothesis=h.id, e7=res, global_decision="GLOBAL_CONFIRMED", why="STRUCTURE_LONG: " + why,
                   e7_s=r.get("sec"), written_wall=time.time(), used_observation_ids=[], fix_blocked=None)
        self.last = out
        tmp = self.state / f".e7_state.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(out, default=str))
        os.replace(tmp, self.state / "e7_state.json")
        with open(self.state / "e7_history.jsonl", "a") as f:
            f.write(json.dumps(dict({k: out[k] for k in out if k != "e7"}), default=str) + "\n")
        return True, why

    def _window_evidence(self, r, cands, res, rec, sem_id):
        """R41: one E7 look at a group = one window. Earlier: any single pass was a FIX, and the same right plates flipped between pass and fail from window to window (X20261103_001810 +640 pass, +700 fail, +760 pass). A FIX now needs MIN_WINDOWS disjoint windows that passed AND put the robot in the same place and heading. Retrying cannot buy a lucky pass: a later window has to agree with the earlier one, on frames it has not seen. -> dict(n, need, why, entry)."""
        import e3_tracker as E3
        import e7_long as EL_
        gid, t_end = r.get("observation_id"), res.get("t_end")
        led = getattr(self, "_e7_ledger", None)
        if not led or led.get("gid") != gid:
            led = self._e7_ledger = dict(gid=gid, passes=[], t_last=None, history=[])
        led["t_last"] = t_end
        s = next((q for q in res.get("results") or [] if q.get("candidate_id") == sem_id), None)
        c = next((c for c in cands if c["candidate_id"] == sem_id), None)
        import e7_semantic_shadow as SH
        ok, why = SH.fix_allowed(rec)
        row = dict(t_end=t_end, window_s=res.get("window_s"), hyp=sem_id, passed=bool(ok), why=why, crossing=(s.get("wall_crossing") or {}).get("violation") if s else None, margin=(rec or {}).get("crossing_margin"), cost=s.get("alignment_score") if s else None, blocked=(rec or {}).get("blocked_by"), rejected=(rec or {}).get("rejected_by"))
        led["history"].append(row)
        hard = [b for b in ((rec or {}).get("blocked_by") or []) if str(b).startswith("CROSSING_ABS")] + list((rec or {}).get("rejected_by") or [])
        if hard:                                                                 # the place itself was contradicted: earlier passes do not stand
            led["passes"] = []
        if not ok or s is None or c is None:
            return dict(n=len(led["passes"]), need=self.min_windows, why=why, history=led["history"])
        w_s = float(res.get("window_s") or 0.0)
        prev = max((float(e["t_end"]) for e in led["passes"]), default=None)
        if prev is not None and t_end is not None and float(t_end) - w_s < prev - 0.5:      # R42: a look triggered early (new photo, new hypotheses) reuses frames of the last pass: neutral, never a second window
            row["overlap_s"] = round(prev - (float(t_end) - w_s), 1)
            return dict(n=len(led["passes"]), need=self.min_windows, why=f"{len(led['passes'])} of {self.min_windows}: this window shares {row['overlap_s']} s with the last passing one", history=led["history"])
        fr = E7.frames(self.tr.store, t_end, 30.0, structural=True)
        rec_now = EL_.recentered(dict(c["rec"]), fr, s["correction"]) if fr else dict(c["rec"])
        tn = fr[-1] if fr else None
        keep = []
        if tn is not None:
            p1 = E3.pose_in_map(dict(rec_now, light=None), tn[1], tn[2], tn[3])
            for e in led["passes"]:
                p0 = E3.pose_in_map(dict(e["rec"], light=None), tn[1], tn[2], tn[3])
                d = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / float(rec_now["ppm"])
                dh = abs(math.degrees((p1[2] - p0[2] + math.pi) % (2 * math.pi) - math.pi))
                row.setdefault("agree", []).append(dict(with_t_end=e["t_end"], dist_m=round(d, 2), dhead_deg=round(dh, 1)))
                if d <= POSE_AGREE_M and dh <= POSE_AGREE_DEG:
                    keep.append(e)
        led["passes"] = keep + [dict(t_end=t_end, rec=rec_now, hyp=sem_id)]
        led["rec_fix"] = rec_now
        n = len(led["passes"])
        return dict(n=n, need=self.min_windows, why=f"{n} of {self.min_windows} disjoint windows agree on the place", history=led["history"])

    def _semantic_fix(self, out, rec, r, e2_why, sess0, cands=None, res=None):
        """R37: the only new way to a semantic FIX, and only when the policy allows it (e7_policy.semantic_policy): the strict rule of e7_semantic_shadow.fix_allowed, plus the gates every FIX has. -> out, changed or not."""
        if getattr(self, "pair", None) is None or rec is None:      # SINGLE path / no shadow record: nothing to judge
            if out.get("global_decision") == "GLOBAL_CONFIRMED":    # R42: E7's own confirm needs the same rule record as every other FIX
                g = "no semantic rule record (E8 pair or shadow rule missing)"
                return dict(out, global_decision="MORE_OBSERVATION_REQUIRED", why=f"E7 geometry would confirm, but {g}", fix_blocked=dict(at_wall=time.time(), e7_decision="GLOBAL_CONFIRMED", why=g))
            return out
        import e7_semantic_shadow as SH
        pol = P7.semantic_policy()
        ok, why = SH.fix_allowed(rec)
        W = self._window_evidence(r, cands or [], res or {}, rec, out.get("semantic_hypothesis")) if res else dict(n=self.min_windows, need=self.min_windows, why="no window record")
        try:
            with open(self.state / "e7_windows.jsonl", "a") as f:
                f.write(json.dumps(dict(wall=time.time(), gid=r.get("observation_id"), **{k: W.get(k) for k in ("n", "need", "why")}, last=(W.get("history") or [None])[-1]), default=str) + "\n")
        except OSError:
            pass
        t_end = (out.get("e7") or {}).get("t_end")
        if ok and self.max_odom_age_s and t_end and time.time() - float(t_end) > self.max_odom_age_s:
            ok, why = False, f"LiDAR / odometry {time.time() - float(t_end):.0f} s old at the decision"
        if out.get("global_decision") == "GLOBAL_CONFIRMED":
            # R42: E7's own GLOBAL_CONFIRMED (geometry unique + a pair) passed none of the R38/R41 rules: no floor evidence beyond the room digits, one window,
            # no odometry age, and not even LIVE2_SEMANTIC_FIX=0. It now passes the same gates as SEMANTIC_2ROOM; geometry uniqueness is extra, never instead.
            g = None if ok else f"semantic rule: {why}"
            if g is None and pol["mode"] != "ENABLED":
                g = f"semantic FIX policy {pol['mode']}"
            if g is None and W["n"] < self.min_windows:
                g = f"E7 독립 구간 {W['n']}/{self.min_windows}"
            if g:
                return dict(out, global_decision="MORE_OBSERVATION_REQUIRED", why=f"E7 geometry would confirm, but {g}", fix_blocked=dict(at_wall=time.time(), e7_decision="GLOBAL_CONFIRMED", why=g),
                            semantic_rule=dict(allowed=False, why=g, mode=pol["mode"], windows=W))
            used = [x.get("observation_id") for x in rec.get("room_readings") or [] if x.get("observation_id")]
            return dict(out, semantic_rule=dict(allowed=True, why=why, mode=pol["mode"], windows=W), used_observation_ids=used or out.get("used_observation_ids"))
        if not ok:
            return dict(out, semantic_rule=dict(allowed=ok, why=why, mode=pol["mode"]))
        if e2_why:
            return dict(out, semantic_rule=dict(allowed=False, why=f"E2 rival list unavailable: {e2_why}", mode=pol["mode"]))
        bad, why_lio = self.fastlio_degraded()
        if bad or (sess0 is not None and self._epoch() != sess0):
            return dict(out, semantic_rule=dict(allowed=False, why=why_lio if bad else "ODOM_SESSION_CHANGED", mode=pol["mode"]))
        if pol["mode"] != "ENABLED":
            return dict(out, semantic_rule=dict(allowed=True, why="SHADOW: " + why, mode=pol["mode"]), why=out.get("why"))
        if W["n"] < self.min_windows:
            return dict(out, semantic_rule=dict(allowed=False, why=f"E7 독립 구간 {W['n']}/{self.min_windows} — 새 구간에서 같은 위치가 다시 확인되어야 합니다", mode=pol["mode"], windows=W))
        used = [x.get("observation_id") for x in rec.get("room_readings") or [] if x.get("observation_id")]    # R41: the plates this FIX rests on, not every photo of the group
        return dict(out, global_decision="GLOBAL_CONFIRMED", path="SEMANTIC_2ROOM", why="SEMANTIC_2ROOM: " + why + f"; {W['why']}", semantic_rule=dict(allowed=True, why=why, mode=pol["mode"], windows=W), used_observation_ids=used or out.get("used_observation_ids"))

    def current_pose(self):
        """R41: after a FIX the robot goes on. The verified transform (the last agreeing E7 window) + the newest FAST-LIO sample = where it is now. Written for the page; the FIX record itself stays as verified. -> dict or None."""
        if (self.last or {}).get("global_decision") != "GLOBAL_CONFIRMED":
            return None
        rec = (getattr(self, "_e7_ledger", None) or {}).get("rec_fix")
        rows = getattr(self.tr.store, "rows", None) or []
        if not rec or not rows:
            return None
        import e3_tracker as E3
        r = rows[-1]
        px, py, h = E3.pose_in_map(dict(rec, light=None), r[2], r[3], r[4])
        try:                                                    # R41: metres on the same scale as the popup's verified pose (final2: 65.31 m verified, 73.12 m current for a 3.7 m move, because the ledger transform carries its own ppm)
            m_ppm = float(json.loads((self.state / "fix_overlay.json").read_text())["ppm"])
        except (OSError, ValueError, KeyError, TypeError):
            m_ppm = float(rec["ppm"])
        return dict(kind="LIVE2_FIX_CURRENT", floor=rec.get("floor"), pose_px=[round(px, 1), round(py, 1)], pose_m=[round(px / m_ppm, 2), round(py / m_ppm, 2)], heading_deg=round(math.degrees(h) % 360, 1), ppm=round(rec["ppm"], 2), odom_t=r[1], source="verified transform + FAST-LIO", written_wall=time.time())

    def snapshot(self):
        return self.last


def hypothesis_updates(r, res, usable, gate, hyps):
    """One audit row per alive hypothesis for this observation (E6 -> E3 contract). Only a decisive reading
    (max p >= e3 SEM_DECISIVE_P) changes ranking; every row says why."""
    import e3_tracker as E3
    alts = res.get("alternatives") or []
    p_max = max((p for _, p in alts), default=0.0)
    decisive = bool(alts) and p_max >= E3.SEM_DECISIVE_P
    fusion = res.get("fusion") or {}
    fl = res.get("floor_likelihood") or {}
    labels = {lab for lab, _ in alts}
    ocr_conf = max((float(c) for t, c in (r.get("ocr_lines") or []) if str(t).replace(" ", "") in labels), default=None)
    rows = []
    for h in hyps:
        v = (res.get("per_hypothesis") or {}).get(h["id"]) or dict(verdict="NO_LABEL")
        if not usable:
            reason = f"NO_EVIDENCE: {gate}"
        elif not alts:
            reason = f"NO_LABEL: fusion {fusion.get('status')}"
        elif not decisive:
            reason = f"NOT_DECISIVE: max p {p_max:.2f} < {E3.SEM_DECISIVE_P}"
        else:
            reason = f"DECISIVE: {v.get('verdict')}"
        rows.append(dict(kind="LIVE2_E6_HYPOTHESIS_UPDATE", observation_id=r.get("observation_id"), hypothesis_id=h["id"],
                         floor=h["floor"], observed_room=v.get("observed_room_id") or sorted(labels) or None,
                         map_room=dict(id=v.get("map_room_id"), name=v.get("map_room_name")) if v.get("map_room_id") else None,
                         ocr_confidence=ocr_conf, label_p=p_max or None,
                         semantic_likelihood=dict(floor=fl.get(h["floor"]), place=v.get("semantic_score"), dist_m=v.get("dist_m")),
                         verdict=v.get("verdict"), decisive=decisive, applied_to_ranking=decisive and usable,
                         observation_timestamp=r.get("capture_timestamp") or r.get("wall"), source=r.get("source"),
                         supporting_evidence=dict(fusion_status=fusion.get("status"), ocr_lines=r.get("ocr_lines"),
                                                  vlm=r.get("vlm"), image_quality=r.get("image_quality"),
                                                  capture_file=r.get("capture_file"), evidence_gate=gate),
                         update_reason=reason, gt_used=False, wall=time.time()))
    return rows


def evidence_gate(r):
    """Only a capture that passed the image-quality gate (student-formula sharpness on the sign ROI >= 30) may carry
    OCR/VLM text into the semantic map. Blurry / missing / refused captures give NO evidence (never a label).
    -> (usable, reason)."""
    iq = r.get("image_quality")
    if r.get("status") != "CAPTURED":
        return False, f"status {r.get('status')}"
    if not isinstance(iq, dict) or iq.get("ok") is not True:
        return False, f"image_quality not ok ({iq})"
    if r.get("source") not in ("AUTO_E5", "OPERATOR_ASSISTED"):
        return False, f"unknown source {r.get('source')}"
    return True, f"sharp {iq.get('roi_laplacian_var')} >= {iq.get('min_sharp')} ({iq.get('roi_source', 'roi')})"


STRUCTURAL_CONTRADICTION, STRUCTURALLY_AMBIGUOUS, STRUCTURALLY_COMPATIBLE = "STRUCTURAL_CONTRADICTION", "STRUCTURALLY_AMBIGUOUS", "STRUCTURALLY_COMPATIBLE"
ABS_FAIL_NO_RIVAL_M = 0.15     # no rival to compare with: E7's own inlier distance as the mean fit that counts as a failed fit


def structural_class(res, sem_id):
    """R35: what the LiDAR says about the semantic candidate. -> (class, why). A label only: GLOBAL_CONFIRMED still comes from E7's unchanged final_decision.
    STRUCTURAL_CONTRADICTION  a rival fits better by >= GAP_M, or (no rival) the candidate's own fit failed
    STRUCTURALLY_AMBIGUOUS    nothing contradicts it, but the geometry does not single it out (rival within GAP_M, not converged, crossing veto, local ambiguity)
    STRUCTURALLY_COMPATIBLE   first by >= GAP_M, converged, no veto; E7 lacks only the semantic part
    GLOBAL_CONFIRMED          E7's own final decision"""
    rs = [r for r in res.get("results") or [] if isinstance(r, dict) and r.get("alignment_score") is not None]
    s = next((r for r in rs if r.get("candidate_id") == sem_id), None)
    if res.get("best") == sem_id and res.get("final_decision") == "GLOBAL_CONFIRMED":
        return "GLOBAL_CONFIRMED", "E7 geometry unique and separated, semantic match"
    if s is None:
        return STRUCTURALLY_AMBIGUOUS, f"no E7 result for {sem_id}"
    others = [r for r in rs if r is not s]
    margin = E7.GAP_M
    if others:
        b = min(others, key=lambda r: r["alignment_score"])
        lead = round(s["alignment_score"] - b["alignment_score"], 4)
        if lead >= margin:
            return STRUCTURAL_CONTRADICTION, f"{b.get('candidate_id')} fits better than {sem_id} by {lead} m >= {margin} m"
    elif s["alignment_score"] >= ABS_FAIL_NO_RIVAL_M:
        return STRUCTURAL_CONTRADICTION, f"{sem_id} fit {s['alignment_score']} m >= {ABS_FAIL_NO_RIVAL_M} m with no rival"
    if res.get("best") != sem_id:
        lead = round(s["alignment_score"] - min((r["alignment_score"] for r in others), default=s["alignment_score"]), 4)
        return STRUCTURALLY_AMBIGUOUS, (f"best {res.get('best')} leads {sem_id} by only {lead} m < {margin} m: LiDAR does not separate them "
                                        f"({'best on its search bound, ' if not next((r for r in rs if r.get('candidate_id') == res.get('best')), {}).get('convergence') else ''}no structural contradiction)")
    why = [w for w, bad in ((f"rival within {res.get('gap_m')} m < {margin} m", bool(res.get("ambiguity"))), ("not converged", not s.get("convergence")), ("locally ambiguous", (s.get("local_ambiguity") or {}).get("ambiguous")),
                            ("wall-crossing veto", res.get("crossing_veto")), ("per-segment costs overlap", res.get("structurally_indistinguishable"))) if bad]
    if why:
        return STRUCTURALLY_AMBIGUOUS, "; ".join(why)
    return STRUCTURALLY_COMPATIBLE, f"{sem_id} first by {res.get('gap_m')} m, converged, no veto"


def global_decision(res, sem_id):
    """R35: the only change from R34: a rival in first place by less than GAP_M is AMBIGUOUS, not REJECTED ('geometry disagrees' was false there). GLOBAL_CONFIRMED has exactly the old condition."""
    if not res.get("results"):
        return "MORE_OBSERVATION_REQUIRED", res.get("why") or "no E7 result"
    if res.get("best") == sem_id and res.get("final_decision") == "GLOBAL_CONFIRMED":
        return "GLOBAL_CONFIRMED", "decisive semantic reading consistent AND E7 geometry unique and separated"
    cls, why = structural_class(res, sem_id)
    if cls == STRUCTURAL_CONTRADICTION:
        return "REJECTED", f"E7: {why}"
    if res.get("best") != sem_id:
        return "AMBIGUOUS", f"E7: {why}"
    if res.get("ambiguity"):
        return "AMBIGUOUS", f"E7 gap {res.get('gap_m')} m to a rival"
    return "MORE_OBSERVATION_REQUIRED", res.get("why") or res.get("final_decision")
