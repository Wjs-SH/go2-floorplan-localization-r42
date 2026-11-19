"""STAGING (2026-11-01) post-capture review / retake (field X20261031_231940: 2 of 3 photos unreadable, sent to OCR unseen).
Real e6_pipeline / e6_manual / e8_pair code with the fake aim console of test_e8_manual_focus_fake (no device, no camera).
Checked: retake stays in the SAME observation (one record, one accepted photo, older photos archived with their sha256 and never
read), HOME only after acceptance with hands clear, the head never re-aims for a retake, moving / cancel during the review
ends the observation without a stored record, attempt limit, commands out of phase are refused."""
import hashlib
import json
import time
from pathlib import Path

import cv2

from test_e8_manual_focus_fake import M, OBS, Clock, FakeConsole, P, build, first, phases, render  # noqa: F401
from test_e8_pair_fake import Hyp, OCR, floors, manager


def cmd(**k):
    return dict(k)


def new_photo(c, at):
    def f():
        c.jpg = cv2.imencode(".jpg", render(True, at=at))[1].tobytes()
        return None
    return f


def test_retake_keeps_one_observation_and_reads_only_the_accepted_photo(tmp_path):
    c = FakeConsole(render(False), sharp=6.0)
    sha1 = hashlib.sha256(c.jpg).hexdigest()
    script = first(cmd(action="capture_anyway")) + [None, cmd(action="retake"), new_photo(c, (900, 500)), cmd(action="capture"),
                                                    None, cmd(action="accept_capture")]
    p, ch, _ = build(tmp_path, c, script=script, review=None)
    rec = p.run(OBS, 1)
    k = c.kinds()
    caps = [i for i, x in enumerate(k) if x == "live2/manual_capture"]
    assert len(caps) == 2 and k.count("home") == 1 and k.index("home") > caps[1]          # HOME only after the accepted photo
    assert k.count("aim") == 1                                                                # no re-aim for a retake
    ph = phases(ch)
    assert ph.count("CAPTURE_REVIEW") == 2 and ph.index("CAPTURE_REVIEW") < ph.index("RETURNING_HOME")
    a = rec["manual"]["attempts"]
    assert [x["decision"] for x in a] == ["RETAKE", "ACCEPTED"] and rec["manual"]["n_attempts"] == 2
    old = Path(a[0]["file"])
    assert old.name == "O_m1_try1.jpg" and a[0]["sha256"] == a[0]["sha256_after_archive"] == sha1
    assert hashlib.sha256(old.read_bytes()).hexdigest() == sha1
    f = Path(rec["capture_file"])
    assert f.name == "O_m1.jpg" and rec["image_sha256"] == hashlib.sha256(f.read_bytes()).hexdigest() != sha1
    assert not rec["image_quality"]["operator_override_low_sharpness"]                       # quality of the ACCEPTED photo
    (tmp_path / "pair").mkdir(exist_ok=True)
    pm, _, _ = manager(tmp_path / "pair", [Hyp("H4", 4)], floors(4), OCR)
    pm.on_precision(rec)
    assert len(pm.group) == 1 and pm.group[0]["image_sha256"] == rec["image_sha256"]          # one observation, accepted photo


def test_accept_needs_hands_clear_and_capture_is_refused_during_review(tmp_path):
    c = FakeConsole(render())
    script = first(cmd(action="capture")) + [None, cmd(action="capture"), cmd(action="accept_capture", hands_clear=False),
                                             cmd(action="accept_capture")]
    p, ch, _ = build(tmp_path, c, script=script, review=None)
    rec = p.run(OBS, 1)
    ev = [json.loads(x) for x in (tmp_path / "e6_manual_events.jsonl").read_text().splitlines()]
    rep = [e for e in ev if e["kind"] == "COMMAND"]
    assert [r["accepted"] for r in rep] == [True, False, False, True]
    assert "REVIEW_PENDING" in rep[1]["why"] and "HANDS_CLEAR_REQUIRED" in rep[2]["why"]
    assert rec["status"] == "CAPTURE_STORED" and c.kinds().count("live2/manual_capture") == 1


def test_review_commands_outside_the_review_are_refused(tmp_path):
    c = FakeConsole(render())
    p, ch, _ = build(tmp_path, c, script=first(cmd(action="accept_capture")) + [cmd(action="retake"), cmd(action="capture")])
    rec = p.run(OBS, 1)
    ev = [json.loads(x) for x in (tmp_path / "e6_manual_events.jsonl").read_text().splitlines() if '"COMMAND"' in x]
    assert ev[0]["accepted"] is False and "NO_CAPTURE_TO_REVIEW" in ev[0]["why"] and "NO_CAPTURE_TO_REVIEW" in ev[1]["why"]
    assert rec["status"] == "CAPTURE_STORED"


def test_robot_moving_during_the_review_ends_without_a_record(tmp_path):
    clk = Clock()
    state = dict(moving=False)
    stop = lambda: dict(stopped=not state["moving"], why="MOVING speed 0.4 m/s" if state["moving"] else "STOPPED")  # noqa: E731

    def go():
        state["moving"] = True
        return None
    c = FakeConsole(render())
    p, ch, _ = build(tmp_path, c, script=first(cmd(action="capture")) + [go] + [None] * 60, stopped=stop, clock=clk, review=None)
    rec = p.run(OBS, 1)
    assert rec["status"] == "ROBOT_MOVED_DURING_FOCUS" and "CAPTURE_REVIEW" in phases(ch)
    a = rec["manual"]["attempts"]
    assert len(a) == 1 and a[0]["decision"] == "NOT_ACCEPTED_ROBOT_MOVED" and Path(a[0]["file"]).name == "O_m1_try1.jpg"
    assert not rec.get("capture_file") and "home" in c.kinds()
    (tmp_path / "pair").mkdir(exist_ok=True)
    pm, _, _ = manager(tmp_path / "pair", [Hyp("H4", 4)], floors(4), OCR)
    pm.on_precision(rec)
    assert pm.group == []


def test_cancel_during_review_is_not_counted(tmp_path):
    c = FakeConsole(render())
    p, ch, _ = build(tmp_path, c, script=first(cmd(action="capture")) + [None, cmd(action="cancel")], review=None)
    rec = p.run(OBS, 1)
    assert rec["status"] == "USER_CANCELLED" and rec["manual"]["attempts"][0]["decision"] == "NOT_ACCEPTED_USER_CANCELLED"
    assert not (tmp_path / "caps" / "O_m1.jpg").exists() and (tmp_path / "caps" / "O_m1_try1.jpg").exists()


def test_attempt_limit(tmp_path):
    c = FakeConsole(render())
    script = [None]
    for _ in range(M.MAX_ATTEMPTS):
        script += [cmd(action="capture"), None, cmd(action="retake")]
    script += [cmd(action="accept_capture")]
    p, ch, _ = build(tmp_path, c, script=script, review=None)
    rec = p.run(OBS, 1)
    a = rec["manual"]["attempts"]
    assert c.kinds().count("live2/manual_capture") == M.MAX_ATTEMPTS and len(a) == M.MAX_ATTEMPTS
    assert [x["decision"] for x in a] == ["RETAKE"] * (M.MAX_ATTEMPTS - 1) + ["ACCEPTED"]
    ev = [json.loads(x) for x in (tmp_path / "e6_manual_events.jsonl").read_text().splitlines() if '"COMMAND"' in x]
    assert any("MAX_ATTEMPTS" in (e.get("why") or "") for e in ev)
