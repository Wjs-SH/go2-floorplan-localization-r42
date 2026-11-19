"""loc_overlay.py — LiDAR-on-map view of the current E3 leader (2026-10-31; display only, never an input to E2..E8).

Every OVERLAY_EVERY_S the e34 loop writes <state>/loc_overlay.json:
  * the leader's newest structural LiDAR points (LiveStore samples, the SAME frames / projection / wall-distance field E7 uses:
    e7_rematch.frames(structural=True) + project + sample_bilinear) in the floor-plan pixel frame of that floor's structure map
    (= the frame of pose_px and of /img/floor/<b>/<f>.png; no semantic-map scale is involved)
  * per point the wall distance in m at the leader's own px/m (on-wall <= ON_WALL_M), the median and the on-wall fraction
  * when an E7 result exists for that leader (structure path pass / semantic E7), the same points re-projected with E7's correction
    (dx, dy, dphi, scale) and the pose before / after the correction.
Downsampled (<= MAX_PTS); no extra LiDAR subscription; the map image is never re-sent (the page caches it).
"""
import json
import math
import os
import time
from pathlib import Path

import numpy as np

OVERLAY_EVERY_S, WINDOW_S, MAX_FRAMES, PTS_PER_FRAME, MAX_PTS, ON_WALL_M = 3.0, 20.0, 8, 70, 560, 0.30
ACCUM_EVERY_S, ACCUM_FRAMES, ACCUM_PTS_PER_FRAME, ACCUM_MAX_PTS = 15.0, 60, 25, 1200    # R39: the whole path so far on the structure map: same frames, same projection, thinner; no new sensor read


class Overlay:
    def __init__(self, state_dir, tracker, e7mod):
        self.state, self.tr, self.E7 = Path(state_dir), tracker, e7mod
        self.next, self.slow = 0.0, 0.0
        self.accum, self.t_accum = None, 0.0

    def _correction(self, leader):
        """E7 correction for this leader: newest structure-path pass that evaluated it, else the semantic E7 state."""
        for name in ("structure_path.json", "e7_state.json"):
            try:
                d = json.loads((self.state / name).read_text())
            except (OSError, ValueError):
                continue
            if name == "structure_path.json":
                if d.get("leader") != leader:
                    continue
                for p in reversed(d.get("passes") or []):
                    if p.get("leader_correction"):
                        return dict(p["leader_correction"], source="E7 structure window", t_end=p.get("t_end"), ok=p.get("ok"))
            else:
                for r in ((d.get("e7") or {}).get("results") or []):
                    if r.get("candidate_id") == leader and r.get("correction"):
                        return dict(r["correction"], source=f"E7 {d.get('source') or 'semantic'}", ok=d.get("global_decision") == "GLOBAL_CONFIRMED",
                                    alignment_score=r.get("alignment_score"))
        return None

    def fix_snapshot(self):
        """2026-11-01 (r33): ONCE per backend FIX (e7_state.json GLOBAL_CONFIRMED), <state>/fix_overlay.json for the final-result
        view: the LiDAR frames E7 itself evaluated (e7_rematch.frames(store, e7.t_end, e7.window_s, structural) = the call inside
        E7.evaluate), projected with the confirmed hypothesis before / after E7's own correction for that hypothesis. Nothing is
        synthesised: no E7 result / no hypothesis / frames gone -> the record says why and carries no points."""
        try:
            d = json.loads((self.state / "e7_state.json").read_text())
        except (OSError, ValueError):
            return None
        if d.get("global_decision") != "GLOBAL_CONFIRMED":
            return None
        tw = float(d.get("written_wall") or 0)
        if getattr(self, "_fix_done", None) == tw:
            return None
        self._fix_done = tw
        hid, e7 = d.get("semantic_hypothesis"), d.get("e7") or {}
        out = dict(kind="LIVE2_FIX_OVERLAY", e7_written_wall=tw, hypothesis=hid, path=d.get("path") or d.get("source"),
                   t_end=e7.get("t_end"), window_s=e7.get("window_s"), structural=e7.get("structural"),
                   frame="structure-map px of the fix floor (same as pose_px / floor image)", built_wall=time.time())
        try:
            sess = self.tr.store.session() if hasattr(self.tr.store, "session") else None
            out["odom_session"] = (sess or {}).get("odom_session_id") if isinstance(sess, dict) else sess
        except Exception:                                                     # noqa: BLE001
            pass
        h = next((x for x in self.tr.alive() if x.id == hid), None)
        r = next((x for x in (e7.get("results") or []) if x.get("candidate_id") == hid), None)
        if h is None or e7.get("t_end") is None:
            out["why"] = "확정 가설이 현재 추적기에 없음" if h is None else "E7 결과에 시간창 정보 없음"
        else:
            E7 = self.E7
            fr = E7.frames(self.tr.store, float(e7["t_end"]), float(e7.get("window_s") or 90.0), structural=bool(e7.get("structural")))
            if len(fr) < 2:
                out["why"] = "E7 시간창의 LiDAR 프레임이 저장소에 남아 있지 않음"
            else:
                rec = dict(h.rec, light=h.light_dir)
                light = self.tr.light(h.light_dir)
                c = (r or {}).get("correction") or {}
                dd = (float(c.get("dx_m", 0.0)), float(c.get("dy_m", 0.0)), math.radians(float(c.get("dphi_deg", 0.0))), float(c.get("scale", 1.0)))
                mp, ppm = E7.project(rec, fr)
                mp2, ppm2 = E7.project(rec, fr, dd)
                big = float(max(light.H, light.W))
                e1 = np.minimum(E7.sample_bilinear(light, mp, big) / ppm, 5.0)
                e2 = np.minimum(E7.sample_bilinear(light, mp2, big) / ppm2, 5.0)
                sel = np.linspace(0, len(mp) - 1, min(1200, len(mp))).astype(int)
                p0, p1 = mp_pose(rec, fr, (0.0, 0.0, 0.0, 1.0), E7), mp_pose(rec, fr, dd, E7)
                out.update(floor=h.floor, light=h.light_dir, n_frames=len(fr), frame_t=[fr[0][0], fr[-1][0]], ppm=round(float(ppm2), 3),
                           correction=dict(dx_m=c.get("dx_m"), dy_m=c.get("dy_m"), dphi_deg=c.get("dphi_deg"), scale=c.get("scale")) if c else None,
                           correction_why=None if c else "E7 결과에 이 가설의 보정량 없음 — 보정 전 자세만 표시",
                           pose_before_px=[round(p0[0], 1), round(p0[1], 1)], heading_before_deg=round(p0[2], 1),
                           pose_px=[round(p1[0], 1), round(p1[1], 1)], heading_deg=round(p1[2], 1),
                           pose_m=[round(p1[0] / float(ppm2), 2), round(p1[1] / float(ppm2), 2)],
                           shift_m=round(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / float(ppm2), 2),
                           pts_before=[[round(float(mp[i, 0]), 1), round(float(mp[i, 1]), 1)] for i in sel],
                           pts=[[round(float(mp2[i, 0]), 1), round(float(mp2[i, 1]), 1)] for i in sel],
                           dist_m=[round(float(e2[i]), 2) for i in sel], on_wall_m=ON_WALL_M,
                           on_wall_frac_before=round(float(np.mean(e1 <= ON_WALL_M)), 3), on_wall_frac=round(float(np.mean(e2 <= ON_WALL_M)), 3),
                           median_dist_m=round(float(np.median(e2)), 3),
                           e7_scores=dict(alignment_score=(r or {}).get("alignment_score"), residual_m=(r or {}).get("residual"),
                                          inlier_ratio=(r or {}).get("inlier_ratio"), n_points=(r or {}).get("n_points"),
                                          gap_m=e7.get("gap_m"), runner_up=e7.get("runner_up"), best=e7.get("best")),
                           rec_note="hypothesis transform at snapshot time (E3 re-anchoring keeps the odom->map mapping)")
        tmp = self.state / f".fix_overlay.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(out, default=float))
        os.replace(tmp, self.state / "fix_overlay.json")
        return out

    def tick(self, now=None):
        now = time.time() if now is None else now
        try:
            self.fix_snapshot()                                               # once per FIX, independent of the 3 s cadence
        except Exception as ex:                                                # noqa: BLE001  display add-on never breaks e34
            try:
                (self.state / "fix_overlay.json").write_text(json.dumps(dict(kind="LIVE2_FIX_OVERLAY", why=f"snapshot error {type(ex).__name__}: {ex}",
                                                                             e7_written_wall=getattr(self, "_fix_done", None))))
            except OSError:
                pass
        if now < self.next:
            return None
        t0 = time.perf_counter()
        try:
            out = self.build()
        except Exception as ex:                                                 # noqa: BLE001  display add-on never breaks e34
            out = dict(kind="LIVE2_LOC_OVERLAY", error=f"{type(ex).__name__}: {ex}")
        out["written_wall"] = time.time()
        out["build_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        tmp = self.state / f".loc_overlay.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(out, default=float))
        os.replace(tmp, self.state / "loc_overlay.json")
        slow = out["build_ms"] > 400.0                                          # back off when the Jetson is busy
        self.next = now + (OVERLAY_EVERY_S * 3 if slow else OVERLAY_EVERY_S)
        return out

    def _accum(self, h, rec, light, rows):
        """R39: all structural LiDAR points kept so far, placed by the leader's transform and FAST-LIO: what E7's whole-path comparison sees. Display only."""
        E7 = self.E7
        fr = E7.frames(self.tr.store, rows[-1][1], rows[-1][1] - rows[0][1], max_frames=ACCUM_FRAMES, pts_per_frame=ACCUM_PTS_PER_FRAME, structural=True)
        if len(fr) < 2:
            return None
        mp, ppm = E7.project(rec, fr)
        e = np.minimum(E7.sample_bilinear(light, mp, float(max(light.H, light.W))) / ppm, 5.0)
        sel = np.linspace(0, len(mp) - 1, min(ACCUM_MAX_PTS, len(mp))).astype(int)
        return dict(leader=h.id, floor=h.floor, pts=[[round(float(x), 1), round(float(y), 1)] for x, y in mp[sel]], dist_m=[round(float(v), 2) for v in e[sel]], on_wall_frac=round(float(np.mean(e <= ON_WALL_M)), 3), n_frames=len(fr), n_pts_total=len(mp), span_s=round(rows[-1][1] - rows[0][1], 1), on_wall_m=ON_WALL_M)

    def build(self):
        H = sorted(self.tr.alive(), key=lambda h: h.rank_key())
        if not H:
            return dict(kind="LIVE2_LOC_OVERLAY", leader=None, why="no hypothesis yet")
        h = H[0]
        rows = getattr(self.tr.store, "rows", None) or []
        if not rows:
            return dict(kind="LIVE2_LOC_OVERLAY", leader=h.id, why="no LiDAR sample yet")
        E7 = self.E7
        fr = E7.frames(self.tr.store, rows[-1][1], WINDOW_S, max_frames=MAX_FRAMES, pts_per_frame=PTS_PER_FRAME, structural=True)
        if len(fr) < 2:
            return dict(kind="LIVE2_LOC_OVERLAY", leader=h.id, why="too few structural LiDAR frames in the last 20 s")
        rec = dict(h.rec, light=h.light_dir)
        light = self.tr.light(h.light_dir)
        mp, ppm = E7.project(rec, fr)
        e = np.minimum(E7.sample_bilinear(light, mp, float(max(light.H, light.W))) / ppm, 5.0)
        if time.time() - self.t_accum >= ACCUM_EVERY_S or (self.accum or {}).get("leader") != h.id:
            self.t_accum = time.time()
            self.accum = self._accum(h, rec, light, rows)
        sel = np.linspace(0, len(mp) - 1, min(MAX_PTS, len(mp))).astype(int)
        pose = mp_pose(rec, fr, (0.0, 0.0, 0.0, 1.0), E7)
        out = dict(accum=self.accum if (self.accum or {}).get("leader") == h.id else None, kind="LIVE2_LOC_OVERLAY", leader=h.id, floor=h.floor, light=h.light_dir, ppm=round(float(ppm), 3),
                   t_end=fr[-1][0], n_frames=len(fr), window_s=WINDOW_S,
                   frame="structure-map px of this floor (same as pose_px / floor image); hypothesis transform, not verified",
                   pose_px=[round(pose[0], 1), round(pose[1], 1)], heading_deg=round(pose[2], 1),
                   pts=[[round(float(mp[i, 0]), 1), round(float(mp[i, 1]), 1)] for i in sel],
                   dist_m=[round(float(e[i]), 2) for i in sel], on_wall_m=ON_WALL_M,
                   median_dist_m=round(float(np.median(e)), 3), on_wall_frac=round(float(np.mean(e <= ON_WALL_M)), 3),
                   rivals=[dict(id=x.id, floor=x.floor) for x in H[1:4]])
        c = self._correction(h.id)
        if c:
            d = (float(c.get("dx_m", 0.0)), float(c.get("dy_m", 0.0)), math.radians(float(c.get("dphi_deg", 0.0))), float(c.get("scale", 1.0)))
            mp2, ppm2 = E7.project(rec, fr, d)
            e2 = np.minimum(E7.sample_bilinear(light, mp2, float(max(light.H, light.W))) / ppm2, 5.0)
            pose2 = mp_pose(rec, fr, d, E7)
            out["corrected"] = dict(correction=c, pose_px=[round(pose2[0], 1), round(pose2[1], 1)], heading_deg=round(pose2[2], 1),
                                    shift_m=round(math.hypot(pose2[0] - pose[0], pose2[1] - pose[1]) / float(ppm2), 2),
                                    pts=[[round(float(mp2[i, 0]), 1), round(float(mp2[i, 1]), 1)] for i in sel],
                                    dist_m=[round(float(e2[i]), 2) for i in sel], median_dist_m=round(float(np.median(e2)), 3),
                                    on_wall_frac=round(float(np.mean(e2 <= ON_WALL_M)), 3))
        return out


def mp_pose(rec, fr, d, E7=None):
    """(px, py, heading_deg) of the newest frame's robot pose with correction d, computed with e7_rematch.project itself on a
    synthetic 2-point scan (robot origin and 1 m ahead along the LiDAR x axis = robot forward, yaw + beta); heading in the page
    convention (x right, y up on screen: arrow to (x + cos a, y - sin a))."""
    if E7 is None:
        import importlib
        E7 = importlib.import_module("e7_rematch")
    tn, xn, yn, yawn, _ = fr[-1]
    q, _ = E7.project(rec, [(tn, xn, yn, yawn, np.array([[0.0, 0.0], [1.0, 0.0]]))], d)
    (x0, y0), (x1, y1) = q[0], q[1]
    return float(x0), float(y0), math.degrees(math.atan2(-(y1 - y0), x1 - x0)) % 360.0
