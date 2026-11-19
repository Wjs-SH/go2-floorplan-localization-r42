"""R34: status_view with the REAL field records. Crash text from the REAL loc2 timeline + traceback; depth state from the REAL archived hub index (frozen depth)."""
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "candidate/live2"))
import status_view as SV  # noqa: E402
X = Path.home() / "sci_ws/repairs/livefield2_20261018_151606/experiments/X20261101_193139"
EXP = json.loads((X / "experiment.json").read_text())


def test_real_crash_timeline_and_traceback_are_shown(tmp_path):
    tl = [e for e in EXP["timeline"] if e["wall"] - EXP["created_wall"] < 630]      # just after the first crash (+619.7)
    assert any(e["kind"] == "module_ended_unexpectedly" for e in tl)
    (tmp_path / "e2").mkdir()
    (tmp_path / "e2" / "e2_state.json").write_text(json.dumps(dict(floors_total=9, floors_done=[1, 2, 3], status="RUNNING")))
    logs = tmp_path / "logs"; logs.mkdir()
    (logs / "e2_1.out").write_text((X / "e2_20261101_194850.out").read_text())
    c = SV.e2_chip(dict(EXP, timeline=tl), tmp_path, logs, EXP["created_wall"] + 625)
    assert c["code"] == "E2_CRASHED" and "CHILD_EXITED rc=1" in c["text"] and "E2 3/9" in c["text"] and "TypeError" in c["detail"] and "다시 시작" in c["next"]
    tl2 = [e for e in EXP["timeline"] if e["wall"] - EXP["created_wall"] < 680]     # restarted at +669.9
    assert SV.e2_chip(dict(EXP, timeline=tl2), tmp_path, logs, EXP["created_wall"] + 675)["code"] != "E2_CRASHED"


def test_real_frozen_hub_index_is_depth_stale():
    idx = json.loads((X / "archive/hub_index.json/index.json").read_text())
    idx["wall"] = 1000.0
    c = SV.depth_chip(idx, 1000.5)
    assert c["code"] in ("DEPTH_STALE", "DEPTH_MISSING", "DEPTH_OK")
    assert SV.depth_chip(None, 1.0)["code"] == "SENSOR_HUB_STALE"


def test_real_loc2_final_state_files_build_without_error(tmp_path):
    fs = SV.build(EXP, X / "state", X, None, now=EXP["created_wall"] + 1000)
    codes = [c["code"] for c in fs["chips"]]
    assert fs["gpu_e2"] == "NOT_AUTHORIZED" and "SENSOR_HUB_STALE" in codes and "GLOBAL_CONFIRMED" not in codes
