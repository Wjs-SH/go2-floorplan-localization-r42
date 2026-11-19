"""exp_archive (2026-10-21 field finding: EXP2 overwrote EXP1's /dev/shm/live2_store and FAST-LIO Log). FAKE files only."""
import json
import os
import sys
import time
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "candidate" / "live2"))
import exp_archive as EA  # noqa: E402


def mk_run(tmp, exp_id, t0, t1, store_t, log_t):
    L = tmp / "L"
    st, fl = tmp / "shm" / "live2_store", L / "runtime" / "fastlio" / "Log"
    (st / "samples" / "LIVE_000001").mkdir(parents=True, exist_ok=True)
    (st / "session.json").write_text(json.dumps(dict(started_wall=store_t, fastlio_cfg="/x.yaml", fastlio_cfg_sha256="ab")))
    (st / "odom_link.csv").write_text("sample_id,t_sec\nLIVE_000001,%f\n" % store_t)
    (st / "samples" / "LIVE_000001" / "s.npz").write_bytes(os.urandom(256))
    fl.mkdir(parents=True, exist_ok=True)
    (fl / "mat_pre.txt").write_text("0.1 0 0 0\n")
    os.utime(fl / "mat_pre.txt", (log_t, log_t))
    tag = time.strftime("%Y%m%d_%H%M%S", time.localtime(t0 + 10))
    (L / "runtime" / "field_eval" / tag).mkdir(parents=True, exist_ok=True)
    (L / "runtime" / "field_eval" / tag / "odom_trace.jsonl").write_text("{}\n")
    (L / "candidate" / "experiments" / f"live1_observe_{tag}").mkdir(parents=True, exist_ok=True)
    (L / "candidate" / "experiments" / f"live1_observe_{tag}" / "scenes.jsonl").write_text("{}\n")
    (L / "candidate" / "live2").mkdir(parents=True, exist_ok=True)
    (L / "candidate" / "live2" / "m.py").write_text("x=1\n")
    rd = L / "experiments" / exp_id
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "experiment.json").write_text(json.dumps(dict(exp_id=exp_id, state="ENDED", created_wall=t0, ended_wall=t1)))
    return L, rd, EA.live_sources(L, store=st, hub=tmp / "shm" / "hub")


def test_archive_copies_this_runs_sources_and_verifies(tmp_path):
    now = time.time()
    L, rd, src = mk_run(tmp_path, "X1", now - 300, now - 60, now - 290, now - 61)
    rec = EA.archive_run(rd, src)
    assert rec["verified"] and rec["n_files"] >= 5 and rec["copy_errors"] == 0
    m = json.loads((rd / "archive" / "MANIFEST.json").read_text())
    assert {"live2_store", "fastlio_Log", "field_eval", "observe_run"} <= set(m["sources"])
    assert m["not_recorded"]["ros_bag_raw_lidar"].startswith("RECORDER_NOT_AVAILABLE") and "hub_index.json" in m["missing"]
    assert (rd / "archive" / "live2_store" / "samples" / "LIVE_000001" / "s.npz").read_bytes() == \
        (tmp_path / "shm" / "live2_store" / "samples" / "LIVE_000001" / "s.npz").read_bytes()
    assert (tmp_path / "shm" / "live2_store" / "session.json").exists()          # source untouched (copy, not move)
    ok, _ = EA.verified(rd)
    assert ok
    (rd / "archive" / "SHA256SUMS").write_text("tampered\n")                     # later edit -> no longer verified
    assert not EA.verified(rd)[0]


def test_overwritten_source_is_listed_missing_not_copied(tmp_path):
    now = time.time()
    L, rd, src = mk_run(tmp_path, "X1", now - 300, now - 200, now - 50, now - 40)   # store + log belong to a LATER run
    rec = EA.archive_run(rd, src)
    m = json.loads((rd / "archive" / "MANIFEST.json").read_text())
    assert "live2_store" in m["missing"] and "OVERWRITTEN" in m["missing"]["live2_store"]
    assert "fastlio_Log" in m["missing"] and "live2_store" not in m["sources"] and rec["verified"]


def test_start_gate_archives_previous_run_then_refuses_on_copy_failure(tmp_path):
    now = time.time()
    L, rd, src = mk_run(tmp_path, "X20261021_222400", now - 300, now - 60, now - 290, now - 61)
    ok, why, rec = EA.start_gate(L / "experiments", src)
    assert ok and why.startswith("ARCHIVED_AT_START") and rec["verified"]
    ok, why, _ = EA.start_gate(L / "experiments", src)
    assert ok and why.startswith("ARCHIVE_VERIFIED")
    rd2 = L / "experiments" / "X20261021_230000"                                  # newer run, a source cannot be read
    rd2.mkdir()
    (rd2 / "experiment.json").write_text(json.dumps(dict(exp_id="X20261021_230000", state="ENDED", created_wall=now - 300, ended_wall=now - 60)))
    bad = tmp_path / "shm" / "live2_store" / "odom_link.csv"
    bad.chmod(0)
    try:
        if os.access(bad, os.R_OK):                                               # running as root: cannot simulate
            return
        ok, why, _ = EA.start_gate(L / "experiments", src)
        assert not ok and "ARCHIVE_NOT_VERIFIED" in why
    finally:
        bad.chmod(0o644)


def test_next_run_50s_later_is_not_taken_as_this_runs_store(tmp_path):
    """Real 2026-10-21 timing: EXP1 ended 22:23:10, EXP2 created 22:24:00, EXP2 store session 22:24:15 (a 120 s slack took it)."""
    t1_end = time.time() - 400
    L, rd, src = mk_run(tmp_path, "X1", t1_end - 255, t1_end, t1_end + 65, t1_end + 290)
    rec = EA.archive_run(rd, src, next_t0=t1_end + 50)
    m = json.loads((rd / "archive" / "MANIFEST.json").read_text())
    assert "live2_store" in m["missing"] and "fastlio_Log" in m["missing"] and rec["verified"]
    rec = EA.archive_run(rd, src)                                                 # even without next_t0 (10 s slack)
    assert "live2_store" in json.loads((rd / "archive" / "MANIFEST.json").read_text())["missing"]
