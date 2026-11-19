"""STAGING (2026-11-01, r33): the FIX snapshot for the final-result view, on the REAL loc1 record (X20261030_102108 LiveStore, the
E3 transforms of the operational replay) with the REAL E6E7Link._run (E7 + all FIX gates) — the same cases as
test_e7_semantic_rivals: SEM_OK (+120 s, GLOBAL_CONFIRMED by the real gates), MISREAD (REJECTED), STRAIGHT (AMBIGUOUS),
NO_E2 (MORE_OBSERVATION_REQUIRED). The 'reading' is fixed by the harness (which hypothesis it points to); that is an OFFLINE
replay, not a field FIX. Checked: snapshot only for GLOBAL_CONFIRMED; it uses exactly E7's frames (count = E7's n_frames) and
E7's own correction for that hypothesis; once per FIX; no hypothesis -> reason, no points; SHADOW never snapshots.
Also writes the records the r33 UI test replays (data/). Run: python3 -s -B test_r33_fix_view.py OUT"""
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rp", HERE / "test_e7fix_replay.py")
RP = importlib.util.module_from_spec(spec); spec.loader.exec_module(RP)
RES = dict(cases=[], ok=True)


def check(name, ok, **kw):
    RES["cases"].append(dict(name=name, ok=bool(ok), **kw)); RES["ok"] &= bool(ok)
    print(("PASS " if ok else "FAIL ") + name, json.dumps(kw, default=str)[:300], flush=True)


def run_case(name, sem_id, alive_ids, t_rel, out, e2_mode="real"):
    L, E3 = RP.T.load(RP.STG)
    st, e2 = RP.record("loc1")
    D = json.loads(Path(RP.REC["loc1"]["recon"]).read_text())
    hyps = [RP.T.Hyp(c["candidate_id"], dict(c["rec"], floor=c["floor"]), i, light=c["rec"]["light"])
            for i, c in enumerate(D["cands"]) if c["candidate_id"] in alive_ids]
    sd = out / name; shutil.rmtree(sd, ignore_errors=True); sd.mkdir(parents=True)
    e2dir = sd / "e2"; e2dir.mkdir()
    if e2_mode == "real":
        (e2dir / "e2_state.json").write_text(json.dumps(e2, default=float))
    cs = RP.CutStore(st)
    tr = RP.Tracker(cs, hyps, e2dir, e2["odom_session"]); tr.E3 = E3
    cs.cut(st.rows[0][1] + t_rel)
    link = L.E6E7Link(sd, tr, policy="SINGLE_DIAGNOSTIC")
    cons = [h for h in hyps if h.id == sem_id]
    rivals = [h for h in sorted(hyps, key=lambda h: h.rank_key()) if h not in cons][:L.E7_RIVALS]
    cands = [dict(candidate_id=h.id, floor=h.floor, rec=dict(h.rec, light=h.light_dir)) for h in cons[:3] + rivals]
    link._run(cands, dict(observation_id=f"T_{name}", source="E8_PAIR", pair_observation_ids=["T_a", "T_b"]), sem_id)
    return sd, tr, link


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    data = out / "data"; data.mkdir(exist_ok=True)
    sys.path.insert(0, str(RP.STG))
    import loc_overlay as LO
    import e7_rematch as E7
    ALL = {"H025_F4", "H047_F7", "H003_F2", "H122_F3"}
    cases = [("SEM_OK", "H025_F4", ALL, 120.0, "real"), ("MISREAD", "H047_F7", ALL - {"H025_F4"}, 120.0, "real"),
             ("STRAIGHT", "H025_F4", ALL, 204.6, "real"), ("NO_E2", "H025_F4", ALL, 120.0, "missing")]
    for name, sem, alive, t, e2m in cases:
        sd, tr, link = run_case(name, sem, alive, t, out, e2m)
        e7s = json.loads((sd / "e7_state.json").read_text())
        shutil.copy(sd / "e7_state.json", data / f"e7_state_{name}.json")
        ov = LO.Overlay(sd, tr, E7)
        snap = ov.fix_snapshot()
        if e7s["global_decision"] != "GLOBAL_CONFIRMED":
            check(f"{name}: E7 {e7s['global_decision']} -> no FIX snapshot", snap is None and not (sd / "fix_overlay.json").exists())
            continue
        fo = json.loads((sd / "fix_overlay.json").read_text())
        e7 = e7s["e7"]
        r = next(x for x in e7["results"] if x["candidate_id"] == sem)
        check(f"{name}: GLOBAL_CONFIRMED from the real gates -> snapshot uses E7's OWN frames (count = E7 n_frames)",
              fo["n_frames"] == e7["n_frames"] and fo["window_s"] == e7["window_s"] and fo["t_end"] == e7["t_end"] and fo["structural"] is True,
              snap_frames=fo["n_frames"], e7_frames=e7["n_frames"], window_s=fo["window_s"])
        c = r["correction"]
        check(f"{name}: correction = E7's own correction for {sem}; pose moved by it", fo["correction"] == dict(dx_m=c["dx_m"], dy_m=c["dy_m"],
              dphi_deg=c["dphi_deg"], scale=c.get("scale")) and abs(fo["shift_m"] - math.hypot(c["dx_m"], c["dy_m"])) < 0.35,
              correction=fo["correction"], shift_m=fo["shift_m"], on_wall=[fo["on_wall_frac_before"], fo["on_wall_frac"]],
              pose_m=fo["pose_m"], heading=fo["heading_deg"])
        check(f"{name}: points = E7 frames projected (before {len(fo['pts_before'])} / after {len(fo['pts'])}), final pose in the "
              f"structure-map px of {fo['floor']}F", len(fo["pts"]) == len(fo["pts_before"]) > 100 and fo["floor"] == 4)
        b0 = fo["built_wall"]; time.sleep(0.05); ov.fix_snapshot()
        check(f"{name}: written ONCE per FIX", json.loads((sd / "fix_overlay.json").read_text())["built_wall"] == b0)
        shutil.copy(sd / "fix_overlay.json", data / f"fix_overlay_{name}.json")
        # same FIX record, hypothesis no longer tracked -> reason, never invented points
        tr.H[:] = [h for h in tr.H if h.id != sem]
        ov2 = LO.Overlay(sd, tr, E7)
        (sd / "fix_overlay.json").unlink()
        ov2.fix_snapshot()
        fo2 = json.loads((sd / "fix_overlay.json").read_text())
        check(f"{name}: confirmed hypothesis gone -> reason recorded, no points", fo2.get("why") and "pts" not in fo2, why=fo2.get("why"))
    # SHADOW: structure path would-confirm, e7_state never written -> no snapshot
    sd = out / "SHADOW"; shutil.rmtree(sd, ignore_errors=True); sd.mkdir()
    (sd / "structure_path.json").write_text(json.dumps(dict(status="SHADOW_WOULD_CONFIRM", leader="H025_F4", floor=4)))
    tr0 = RP.Tracker(RP.CutStore(RP.record("loc1")[0]), [], sd, "S")
    check("SHADOW_WOULD_CONFIRM (no e7_state) -> no snapshot", LO.Overlay(sd, tr0, E7).fix_snapshot() is None and not (sd / "fix_overlay.json").exists())
    (out / "result.json").write_text(json.dumps(RES, indent=1, default=str))
    print("ALL_OK" if RES["ok"] else "SOME_FAILED", flush=True)
    return RES["ok"]


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
