"""STAGING (2026-10-31): loc_overlay on the REAL loc1 record X20261030_223346 (body IMU, 10-30 22:33). The final E3 hypotheses are
rebuilt from the archived e3_state (global_pose at odom_t + phi / ppm / beta, as e7_scale_audit recon.py); the overlay is computed at
the end of the record. Checks: (1) frame — the overlay pose equals the E3 snapshot pose_px / heading of the same hypothesis;
(2) the near-truth 4F leader's points lie on walls far more than another floor's leader; (3) a known correction moves the pose
by exactly that many metres. GT is not used. Run: python3 -s -B test_loc_overlay.py OUT"""
import json
import math
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
STG = HERE.parent / "candidate/live2"
sys.path.insert(0, str(STG))
import e3_tracker as E3  # noqa: E402
import e7_rematch as E7  # noqa: E402
import loc_overlay as OV  # noqa: E402

OPR = Path.home() / "sci_ws/repairs/livefield2_20261018_151606"
X = OPR / "experiments/X20261030_223346"
LROOT = OPR / "runtime/e2_cache/light"


class Hyp:
    def __init__(self, h, rows, k):
        gp = h["global_pose"]; row = min(rows, key=lambda r: abs(r[1] - gp["odom_t"]))
        self.id, self.floor, self.k = h["id"], h["floor"], k
        self.light_dir = f"{LROOT}/kci_{h['floor']}F_{h['map_version']}"
        self.rec = dict(pos=[gp["px"], gp["py"]], anchor=[row[2], row[3]], phi=h["phi"], ppm=h["ppm"], beta=math.radians(h["beta_deg"]), mirror=1)
        self.snap = h

    def rank_key(self):
        return (self.k,)


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    st = E3.Store(str(X / "archive/live2_store")); st.refresh()
    S = json.loads((X / "state/e3_state.json").read_text())
    Hs = sorted([h for h in S["hypotheses"] if h.get("global_pose") and h.get("status") in ("ACTIVE", "SUSPENDED")], key=lambda h: h["score"])
    lights = {}
    res = []
    for want in (lambda h: h["id"] == "H080_F4", lambda h: h["floor"] != 4):
        h = next(x for x in Hs if want(x))
        hy = Hyp(h, st.rows, 0)
        tr = types.SimpleNamespace(store=st, alive=lambda hy=hy: [hy],
                                   light=lambda d: lights.setdefault(d, E3.LightMap(d)))
        ov = OV.Overlay(out, tr, E7).build()
        # frame check vs the E3 snapshot (pose at the hypothesis' own odom_t, which is the newest keyframe; the overlay uses the
        # newest LiveStore frame — same within the robot's motion between them)
        dt = st.rows[-1][1] - h["global_pose"]["odom_t"]
        dpx = math.hypot(ov["pose_px"][0] - h["pose_px"][0], ov["pose_px"][1] - h["pose_px"][1])
        dh = (ov["heading_deg"] - h["heading_deg"] + 180) % 360 - 180
        d = (0.5, 0.0, 0.0, 1.0)                                    # synthetic correction: +0.5 m map x
        pc = OV.mp_pose(hy.rec, E7.frames(st, st.rows[-1][1], OV.WINDOW_S, max_frames=OV.MAX_FRAMES, pts_per_frame=OV.PTS_PER_FRAME,
                                          structural=True), d, E7)
        shift = math.hypot(pc[0] - ov["pose_px"][0], pc[1] - ov["pose_px"][1]) / hy.rec["ppm"]
        r = dict(hyp=h["id"], floor=h["floor"], n_pts=len(ov["pts"]), median_dist_m=ov["median_dist_m"], on_wall_frac=ov["on_wall_frac"],
                 pose_vs_e3_px=round(dpx, 1), heading_vs_e3_deg=round(dh, 1), e3_pose_age_s=round(dt, 1), synthetic_shift_m=round(shift, 3),
                 build_ms=None)
        res.append(r); print(json.dumps(r), flush=True)
        (out / f"overlay_{h['id']}.json").write_text(json.dumps(ov, default=float))
    ok = (res[0]["on_wall_frac"] > res[1]["on_wall_frac"] and abs(res[0]["synthetic_shift_m"] - 0.5) < 0.01
          and res[0]["pose_vs_e3_px"] < 3 * 24.6 and abs(res[0]["heading_vs_e3_deg"]) < 15)
    (out / "result.json").write_text(json.dumps(dict(results=res, ok=ok), indent=1))
    print("OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
