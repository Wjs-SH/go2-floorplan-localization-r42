"""R34 §11-8: pre-stop aim guidance on the REAL loc2 observations of X20261101_193139 (O001 pan 155.0, O004 136.8, O007 138.8 / 8.1 deg, all refused after the stop) and the real camera K.
The field depth was frozen, so its stale record gives AIM_TARGET_UNCERTAIN; the geometry is then checked with the same values marked fresh (OFFLINE_COUNTERFACTUAL). No hardware object exists in this test: command_sent is always False."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "candidate/live2"))
import e6_core as G  # noqa: E402
import e6_precision_bridge as B  # noqa: E402
import e6_tracker as TR  # noqa: E402

X = Path.home() / "sci_ws/repairs/livefield2_20261018_151606/experiments/X20261101_193139"
OBS = {}
for ln in open(X / "state/e6_observations.jsonl"):
    d = json.loads(ln)
    if d.get("state") == "STOPPED_TARGET_READY":
        OBS[d["observation_id"][-4:]] = d
K = json.loads((X / "archive/hub_index.json/index.json").read_text())["caminfo"]
ENV = dict(pan=[60.0, 135.0], tilt=[135.0, 167.189], source="operator envelope 2026-10-27")


def KM():
    return K.get("k") or K.get("K") or K


def run(oid, dt=0.05):
    o = OBS[oid]
    od = o["odom_pose_at_detection"]
    pose = G.raw_to_planar(od["x"], od["y"], od["yaw"])
    tu, tv = o["target_pose_in_odom"]["planar_uv"]
    return TR.pre_stop_aim(pose, tu, tv, o["target_pose_in_odom"]["up_m"], KM(), dt, float(o["depth"]), envelope=ENV)


def test_stale_depth_gives_no_number_at_all():
    for oid in OBS:
        r = run(oid, dt=(OBS[oid].get("uncertainty") or {}).get("depth_dt_s"))
        if r["status"] == "AIM_TARGET_UNCERTAIN":
            assert r["rows"] == [] and r["command_sent"] is False
    assert run("O004", dt=538.3)["status"] == "AIM_TARGET_UNCERTAIN" and run("O004", dt=None)["status"] == "AIM_TARGET_UNCERTAIN"


def test_the_three_refused_events_are_flagged_before_the_stop():
    res = {o: run(o) for o in ("O001", "O004", "O007")}
    for o, r in res.items():
        assert r["status"] in ("STOP_EARLY", "AIM_EXPECTED_REFUSED"), (o, r)
        assert r["command_sent"] is False
        assert not r["rows"][-1]["ok"]                        # at the longest measured overshoot it is refused, as in the field


def test_the_good_observations_are_not_all_flagged():
    res = {o: run(o) for o in OBS if o in ("O002", "O003", "O006")}
    assert any(r["status"] == "AIM_OK_AT_TYPICAL_STOP" or any(x["ok"] for x in r["rows"]) for r in res.values()), res


def test_limits_were_not_widened():
    env = B.load_envelope(Path.home() / "sci_ws/repairs/livefield2_20261018_151606/runtime/pantilt_envelope.json")
    assert list(env["pan"]) == [60.0, 135.0] and B.PRED_MAX_DEG == 8.0
    src = (HERE.parent / "candidate/live2/e6_pipeline.py").read_text()
    assert "FINE_AIM" in src
