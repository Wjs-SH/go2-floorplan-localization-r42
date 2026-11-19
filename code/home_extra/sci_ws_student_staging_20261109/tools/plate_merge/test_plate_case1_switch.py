"""test_plate_case1_switch.py — console.start_module 의 E6 갈래에 끼운 "명판 촬영(케이스 1) 방식" 스위치 (학생 병합, 2026-11-13).

운영폴더 runtime/plate_case1.on 이 있으면 팬틸트 사용에 동의한 실험에서도 E6 자동 촬영을 띄우지 않는다(OFF).
스위치가 없으면 원래 동작(동의 있음 -> E6 시작 절차로 진행, 동의 없음 -> OFF)과 같아야 한다. 프로세스는 하나도 띄우지 않는다."""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "candidate/live1"))
sys.path.insert(0, str(HERE.parent / "candidate/live2"))


class FakeExp:
    def __init__(self):
        self.d, self.events, self.mods = {}, [], {}

    def event(self, kind, **k):
        self.events.append(dict(k, kind=kind))

    def mod(self, m, **k):
        self.mods[m] = k


class Proceeded(Exception):
    """E6 갈래를 지나 원래의 시작 절차로 넘어갔다는 표시"""


def make(tmp_path, monkeypatch, approved, switch):
    import console as CO
    c = CO.Controller.__new__(CO.Controller)
    c.exp, c.steps = FakeExp(), []
    c.step = lambda job, **k: c.steps.append(k)
    (tmp_path / "runtime").mkdir(exist_ok=True)
    if switch:
        (tmp_path / "runtime" / "plate_case1.on").write_text("on")
    monkeypatch.setattr(CO, "L", tmp_path)
    monkeypatch.setattr(CO.C, "approval_status", lambda kind, mod: dict(valid=approved, why=None if approved else "NO_APPROVAL_RECORD"))

    def later(*a, **k):
        raise Proceeded()
    monkeypatch.setattr(CO, "approval", later)
    return c


def test_switch_on_with_consent_turns_e6_off(tmp_path, monkeypatch):
    c = make(tmp_path, monkeypatch, approved=True, switch=True)
    assert c.start_module({}, "e6", "kci") is True
    assert c.exp.mods["e6"]["state"] == "OFF" and "명판 촬영" in c.exp.mods["e6"]["why"][0]
    assert c.exp.events == [dict(kind="e6_plate", mode="CASE1")]          # 11-15: E6 는 꺼진 것이 아니라 명판 촬영 방식으로 바뀐 것 -> 사건 이름도 그렇게
    assert c.exp.mods["e6"].get("plate") is True and "꺼짐" not in c.exp.mods["e6"]["why"][0] and c.exp.mods["e6"]["why"][0].startswith("E6 = 명판 촬영 방식")
    assert c.steps == [dict(module="e6", ok=True, why=c.exp.mods["e6"]["why"][0])]


def test_switch_on_without_consent_is_the_original_off(tmp_path, monkeypatch):
    c = make(tmp_path, monkeypatch, approved=False, switch=True)
    assert c.start_module({}, "e6", "kci") is True
    assert c.exp.mods["e6"]["state"] == "OFF" and "승인 B 없음" in c.exp.mods["e6"]["why"][0]
    assert c.exp.events == [dict(kind="e6_off", why="NO_APPROVAL_RECORD")]


def test_switch_off_with_consent_goes_on_to_the_original_e6_start(tmp_path, monkeypatch):
    c = make(tmp_path, monkeypatch, approved=True, switch=False)
    with pytest.raises(Proceeded):
        c.start_module({}, "e6", "kci")
    assert "e6" not in c.exp.mods and not c.exp.events and not c.steps


def test_switch_off_without_consent_is_the_original_off(tmp_path, monkeypatch):
    c = make(tmp_path, monkeypatch, approved=False, switch=False)
    assert c.start_module({}, "e6", "kci") is True
    assert c.exp.mods["e6"]["state"] == "OFF" and c.exp.events == [dict(kind="e6_off", why="NO_APPROVAL_RECORD")]


def test_other_modules_are_not_touched_by_the_switch(tmp_path, monkeypatch):
    c = make(tmp_path, monkeypatch, approved=True, switch=True)
    with pytest.raises(Proceeded):
        c.start_module({}, "hub", "kci")
    assert not c.exp.mods and not c.exp.events


# ---------------- 실험을 시작할 때 학생 도구 시작 스크립트를 돌린다 (a_start 에 끼운 덩어리; 프로세스는 띄우지 않고 부르는지만 본다)
class FakeExpStart(FakeExp):
    def __init__(self):
        import threading
        super().__init__()
        self.lock, self.active = threading.Lock(), True


def controller(tmp_path, monkeypatch, calls):
    import subprocess
    import types
    import console as CO
    c = CO.Controller.__new__(CO.Controller)
    c.exp, c.guard, c.started = FakeExpStart(), types.SimpleNamespace(reset=lambda: None), []
    c.archive_state = lambda d: 0
    c.set_pause = lambda *a: None
    c.start_module = lambda job, m, b: c.started.append(m) or True
    (tmp_path / "runtime").mkdir(exist_ok=True)
    monkeypatch.setattr(CO, "L", tmp_path)
    monkeypatch.setattr(CO, "EXPERIMENTS", tmp_path / "experiments")
    monkeypatch.setattr(CO, "EVAL", tmp_path / "evaluation")
    monkeypatch.setattr(CO, "PENDING_START", tmp_path / "none.json")
    monkeypatch.setattr(CO, "approval", lambda: dict())
    monkeypatch.setattr(CO, "write_pantilt_consent", lambda *a: None)
    monkeypatch.setattr(CO, "sid_hash", lambda s: "h")
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **k: calls.append((cmd, k)))
    return c, CO


def test_start_hook_runs_the_script_when_the_file_names_one(tmp_path, monkeypatch):
    calls = []
    c, CO = controller(tmp_path, monkeypatch, calls)
    script = tmp_path / "hook.sh"; script.write_text("#!/bin/bash\n")
    (tmp_path / "runtime" / "student_start_hook").write_text(str(script) + "\n")
    r = c.a_start(None, "kci", True, "r", "s", "body", "AUTO", None, CO.AUTO_MODE)
    assert r["ok"] and calls and calls[0][0] == ["/bin/bash", str(script)] and calls[0][1].get("start_new_session") is True
    assert dict(kind="student_start_hook", script=str(script)) in c.exp.events
    assert c.started and c.started[0] == "fastlio"                    # 모듈 시작은 그대로 간다


def test_start_hook_does_nothing_without_the_file_or_with_a_missing_script(tmp_path, monkeypatch):
    calls = []
    c, CO = controller(tmp_path, monkeypatch, calls)
    r = c.a_start(None, "kci", True, "r", "s", "body", "AUTO", None, CO.AUTO_MODE)
    assert r["ok"] and not calls and not [e for e in c.exp.events if e["kind"] == "student_start_hook"]
    (tmp_path / "runtime" / "student_start_hook").write_text(str(tmp_path / "no_such_script.sh"))
    r = c.a_start(None, "kci", True, "r2", "s", "body", "AUTO", None, CO.AUTO_MODE)
    assert r["ok"] and not calls


def test_start_hook_failure_does_not_stop_the_experiment(tmp_path, monkeypatch):
    import subprocess
    calls = []
    c, CO = controller(tmp_path, monkeypatch, calls)
    script = tmp_path / "hook.sh"; script.write_text("#!/bin/bash\n")
    (tmp_path / "runtime" / "student_start_hook").write_text(str(script))

    def boom(cmd, **k):
        raise OSError("cannot fork")
    monkeypatch.setattr(subprocess, "Popen", boom)
    r = c.a_start(None, "kci", True, "r", "s", "body", "AUTO", None, CO.AUTO_MODE)
    assert r["ok"] and c.started

