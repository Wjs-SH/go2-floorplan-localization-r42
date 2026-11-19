#!/usr/bin/env python3
"""R41: the keeper with a real child process and scripted temperature / memory. A memory pause resumes at 74 C (the board under E2); a thermal pause does not; the aborts are unchanged."""
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "candidate/live1"))
import observe_keeper as K  # noqa: E402


def run(tmp_path, script, monkeypatch):
    """script(elapsed_s) -> (temp, mem, stop?)."""
    hub = tmp_path / "hub"; hub.mkdir()
    state = tmp_path / "keeper.json"
    t0 = time.time()

    def beat():                     # the hub stays fresh for the whole run; the keeper does not ask for temperature while the hub is stale
        while True:
            (hub / "index.json").write_text("{}")
            if script(time.time() - t0)[2]:
                os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(0.2)

    (hub / "index.json").write_text("{}")
    threading.Thread(target=beat, daemon=True).start()
    monkeypatch.setattr(K.C, "max_temp", lambda: script(time.time() - t0)[0])
    monkeypatch.setattr(K.C, "mem_available_mb", lambda: script(time.time() - t0)[1])
    monkeypatch.setattr(K.C, "approval_status", lambda *a: dict(valid=True, expires_epoch=time.time() + 3600))
    monkeypatch.setattr(K.C, "find_pids", lambda *a: [])
    monkeypatch.setattr(sys, "argv", ["k", "--hub", str(hub), "--state", str(state), "--child-cmd", "exec sleep 60", "--log-prefix", str(tmp_path / "c")])
    old = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    try:
        K.main()
    finally:
        signal.signal(signal.SIGTERM, old[0]); signal.signal(signal.SIGINT, old[1])
    return json.loads(state.read_text())


def test_memory_pause_resumes_under_load(tmp_path, monkeypatch):
    def script(e):
        if e < 1: return 70.0, 6000.0, False       # cold start allowed
        if e < 3: return 74.0, 2400.0, False       # VLM load: abort
        if e < 24: return 74.0, 6000.0, False      # VLM gone, board still warm
        return 74.0, 6000.0, True
    d = run(tmp_path, script, monkeypatch)
    assert d["runs"][0]["end_reason"].startswith("PAUSE:MEM")
    assert len(d["runs"]) == 2, d


def test_memory_pause_waits_above_76(tmp_path, monkeypatch):
    def script(e):
        if e < 1: return 70.0, 6000.0, False
        if e < 3: return 74.0, 2400.0, False
        if e < 24: return 76.5, 6000.0, False
        return 76.5, 6000.0, True
    d = run(tmp_path, script, monkeypatch)
    assert len(d["runs"]) == 1, d


def test_thermal_pause_still_cools_to_72(tmp_path, monkeypatch):
    def script(e):
        if e < 1: return 70.0, 6000.0, False
        if e < 4: return 81.0, 6000.0, False
        if e < 24: return 74.0, 6000.0, False
        return 74.0, 6000.0, True
    d = run(tmp_path, script, monkeypatch)
    assert d["runs"][0]["end_reason"].startswith("PAUSE:TEMP")
    assert len(d["runs"]) == 1, d


def test_limits_unchanged():
    assert (K.START_T, K.ABORT_T, K.START_MB, K.ABORT_MB) == (72.0, 80.0, 4300.0, 2500.0)
