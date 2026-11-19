"""BEFORE reproduction of audit defects in the LEGACY reader (run with SCI_WS=baseline -> expected to FAIL).

Each test asserts the CORRECT behaviour through the legacy API. On the candidate tree the legacy API is replaced
(reader.R1_READER) and each test is skipped as superseded by the named AFTER test in test_r1_reader.py.
Evidence type: SIMULATED (fake OCR/VLM/admission; no model, no HTTP).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

WS = Path(os.environ["SCI_WS"])
for p in (WS / "src" / "sci_text_reading", WS / "src" / "sci_semantic_localization", WS / "external" / "sci_event" / "e5_onboard"):
    sys.path.insert(0, str(p))

import reader as R  # noqa: E402
from semantic import classify_text  # noqa: E402

LEGACY = not getattr(R, "R1_READER", False)


def superseded(after):
    if not LEGACY:
        pytest.skip(f"legacy API replaced in candidate; superseded by {after}")


class Stop(BaseException):
    pass


class FakeOcr:
    load_s = 0.0

    def __init__(self, lines):
        self.lines, self.calls = lines, 0

    def read(self, img, classify):
        self.calls += 1
        return dict(attempts=[], merged=list(self.lines), result=classify(self.lines), input_px=list(img.shape[:2]))


class FakeVlm:
    p, last_use = None, 0.0

    def healthy(self):
        return False

    def stop(self):
        pass


def legacy_reader(run, retry_s=0.0):
    r = R.Reader.__new__(R.Reader)
    r.a = argparse.Namespace(run=str(run), retry_s=retry_s, retry_window_s=600.0, once=False, vlm="on_demand",
                             vlm_idle_s=1e9, floor_hint=None, e6_floors=None, building="kci", easyocr=False)
    r.ocr, r.vlm = FakeOcr([]), FakeVlm()
    r.mon = argparse.Namespace(last=dict(mem_avail_mb=1100, temps={}, gpu_pct=0))
    return r


def make_trial(run, ev):
    trial = run / "trials" / f"ev{ev:03d}"
    (trial / "runner").mkdir(parents=True, exist_ok=True)
    (trial / "runner" / "events.jsonl").write_text(json.dumps(dict(t=1.0, stamp=1.0, picks=[[1, 1, 5, 5]], z=[2.0], pose=[0, 0, 0])) + "\n")
    img = trial / "cap.jpg"
    cv2.imwrite(str(img), (np.random.RandomState(ev).rand(720, 1280, 3) * 255).astype(np.uint8))
    return trial, img


def test_BEFORE_T13_three_deferred_jobs_are_retried_fairly(tmp_path, monkeypatch):
    superseded("test_T13_fair_deferred_and_all_terminal_within_deadline")
    run = tmp_path / "run"
    for d in ("pending", "done", "deferred"):
        (run / "jobs" / d).mkdir(parents=True)
    for ev in (1, 2, 3):
        trial, img = make_trial(run, ev)
        (run / "jobs" / "deferred" / f"ev{ev:03d}.json").write_text(json.dumps(
            dict(ev=ev, trial=str(trial), image=str(img), capture_source="IMX477_PANTILT", created=time.time(), stop={})))
    r = legacy_reader(run)
    calls = []

    def fake_process(j):
        calls.append(Path(j).name)
        if len(calls) >= 9:
            raise Stop()
        return dict(ev=Path(j).stem, status="NO_ROOM_LABEL", vlm_note="vlm_deferred:memory")
    r.process = fake_process
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    with pytest.raises(Stop):
        r.loop()
    tried = set(calls)
    assert tried == {"ev001.json", "ev002.json", "ev003.json"}, f"head-of-line: 9 retries went to {calls}"


def test_BEFORE_T14_retry_does_not_rerun_ocr_or_overwrite_first_read_time(tmp_path, monkeypatch):
    superseded("test_T14_admission_denials_do_not_rerun_ocr_or_reset_times")
    run = tmp_path / "run"
    trial, img = make_trial(run, 1)
    job = run / "jobs" / "pending" / "ev001.json"
    job.parent.mkdir(parents=True)
    job.write_text(json.dumps(dict(ev=1, trial=str(trial), image=str(img), capture_source="IMX477_PANTILT",
                                   created=time.time(), stop={})))
    r = legacy_reader(run)
    monkeypatch.setattr(R, "admit_gpu_job", lambda *a, **k: (False, "memory"))
    out1 = r.process(job)
    time.sleep(0.01)
    out2 = r.process(job)
    assert r.ocr.calls == 1, f"OCR re-run on every retry ({r.ocr.calls} runs)"
    assert out2["timestamps"]["t_read_start"] == out1["timestamps"]["t_read_start"], "first read time overwritten"


def test_BEFORE_T15_pantilt_capture_with_confident_ocr_does_not_force_vlm():
    superseded("test_T15_ocr_first_confident_pantilt_never_calls_vlm")
    import reader_onboard as RO
    rd = RO.OurReader.__new__(RO.OurReader)
    rd._src = "IMX477_PANTILT"
    need, why = rd.want_vlm(dict(result=classify_text([("418", 0.97)])))
    assert not need, f"confident OCR still sent to VLM: {why}"


def test_BEFORE_T18_no_text_and_parse_rejected_are_distinguished(tmp_path, monkeypatch):
    superseded("test_T18_no_text_vs_parse_rejected_vs_legacy")
    run = tmp_path / "run"
    outs = []
    for ev, lines in ((1, []), (2, [("화장실", 0.9)])):
        trial, img = make_trial(run, ev)
        job = run / "jobs" / "pending" / f"ev{ev:03d}.json"
        job.parent.mkdir(parents=True, exist_ok=True)
        job.write_text(json.dumps(dict(ev=ev, trial=str(trial), image=str(img), capture_source="REALSENSE",
                                       created=time.time(), stop={})))
        r = legacy_reader(run)
        r.ocr, r.vlm = FakeOcr(lines), None
        outs.append(r.process(job)["status"])
    assert outs[0] != outs[1], f"no-text and parser-reject both reported as {outs}"
