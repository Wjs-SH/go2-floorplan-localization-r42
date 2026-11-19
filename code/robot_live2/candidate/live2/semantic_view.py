#!/usr/bin/env python3
"""semantic_view.py (R37): 'where do the plates read so far put the robot', for the map, from the first room reading on. DISPLAY ONLY: nothing here reaches E3, E7 or a FIX.

11-02 X20261102_052047: '405' read at +486 s, the anchored fit existed from about +510 s (start error 1.58 m), and nothing on the tablet said where that was. The plate position is not the robot position:
the pose comes from hypotheses fitted to the LiDAR (SEMANTIC_ANCHORED / regional E2 / E2) whose transform is consistent with the reading. The pose is re-evaluated at the newest odometry, so it moves with the robot, and its age is shown.
Statuses: NO_READING | WAITING_CANDIDATE (reading exists, no fitted hypothesis yet) | PROVISIONAL (one room) | MULTI_PLACE (2+ rooms consistent) | INCONSISTENT. Several separated poses are reported as ambiguity, not averaged."""
import json
import math
import os
import time
from pathlib import Path

EVERY_S, SAME_PLACE_M = 1.0, 3.0


def build(pair_snapshot, last_eval, hyps, odom_now, pose_in_map, floors_ppm=None, now=None):
    """pair_snapshot / last_eval: E8 records; hyps: [{id, floor, rec, score, n_kf, sem_verdict, provenance}], rank order; odom_now: (t, x, y, yaw) or None; pose_in_map: e3_tracker.pose_in_map."""
    now = time.time() if now is None else now
    ev = last_eval or {}
    rd = [r for r in ev.get("per_observation") or [] if r.get("usable") and r.get("reading_kind") == "ROOM"]
    d = dict(kind="LIVE2_SEMANTIC_ESTIMATE", authority="NONE: display only", written_wall=now)
    if not rd:
        return dict(d, status="NO_READING", readings=[])
    readings = []
    for r in rd:
        lab = max(r.get("labels") or [["", 0]], key=lambda a: a[1])
        fl = r.get("floor_likelihood") or {}
        f_, p = max(fl.items(), key=lambda kv: kv[1]) if fl else (None, None)
        readings.append(dict(observation_id=r.get("observation_id"), label=lab[0], p=lab[1], floor=None if f_ is None else int(f_), floor_p=p, fusion=r.get("fusion_status"), ocr=r.get("ocr_lines"), vlm=r.get("vlm")))
    floors = ev.get("floors_explaining") or [x["floor"] for x in readings if x["floor"] is not None][-1:]
    want = {int(x) for x in floors}
    cands = []
    for h in hyps:
        if int(h["floor"]) in want and h.get("sem_verdict") == "SEMANTIC_CONSISTENT":
            cands.append(h)
    d.update(readings=readings, floors=sorted(want), basis=["OCR/VLM", "의미지도", "LiDAR", "Odometry"], verdict=ev.get("verdict"), e8_why=ev.get("why"))
    if not cands:
        if ev.get("verdict") == "MORE_OBSERVATION_REQUIRED" and str(ev.get("why", "")).startswith("INCONSISTENT"):
            return dict(d, status="INCONSISTENT", why=ev.get("why"))
        return dict(d, status="WAITING_CANDIDATE", why="의미지도 대응 완료. LiDAR 정합 후보 계산 중")
    if odom_now is None:
        return dict(d, status="WAITING_CANDIDATE", why="odometry 없음")
    t, x, y, yaw = odom_now
    poses = []
    for h in cands:
        px, py, hd = pose_in_map(dict(h["rec"], light=None), x, y, yaw)
        ppm = float(h["rec"].get("ppm") or 25.0)
        poses.append(dict(id=h["id"], floor=int(h["floor"]), pose_px=[round(px, 1), round(py, 1)], heading_deg=round(math.degrees(hd), 1), ppm=round(ppm, 2), score=h.get("score"), n_kf=h.get("n_kf"), provenance=h.get("provenance")))
    best = poses[0]
    far = [p for p in poses[1:] if p["floor"] != best["floor"] or math.hypot(p["pose_px"][0] - best["pose_px"][0], p["pose_px"][1] - best["pose_px"][1]) / best["ppm"] > SAME_PLACE_M]
    return dict(d, status="MULTI_PLACE" if len({x["label"] for x in readings}) >= 2 else "PROVISIONAL", best=best, alternatives=far[:4], n_candidates=len(poses), odom_t=t, age_s=round(now - float(t), 1))


class View:
    def __init__(self, state, tracker, pair):
        self.state, self.tr, self.pair, self.t = Path(state), tracker, pair, 0.0

    def tick(self, now=None):
        now = time.time() if now is None else now
        if self.pair is None or now - self.t < EVERY_S:
            return
        self.t = now
        import e3_tracker as E3
        rows = getattr(self.tr.store, "rows", None)
        od = None if not rows else (rows[-1][1], rows[-1][2], rows[-1][3], rows[-1][4])
        hy = [dict(id=h.id, floor=h.floor, rec=h.rec, score=h.score(), n_kf=h.n_kf, sem_verdict=h.sem_verdict, provenance=h.rec.get("provenance") or h.stage) for h in sorted(self.tr.alive(), key=lambda h: h.rank_key())]
        d = build(None, self.pair.last_eval, hy, od, E3.pose_in_map, now=now)
        tmp = self.state / f".semantic_estimate.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(d, ensure_ascii=False, default=str))
        os.replace(tmp, self.state / "semantic_estimate.json")
