"""R1 W-B BEFORE reproduction (STATIC): asserts the CORRECT console behaviour on the tree in $SCI_WS.

Baseline aim_console.py cannot be imported under tools/r1_guard.py (calib_console/autofocus try to import pyserial and
aruco_realsense_probe tries rclpy), so the defects are shown from source (STATIC evidence, not runtime).
Expected: FAIL on baseline (defect present), PASS on candidate.
"""
import ast
import os
from pathlib import Path

SRC_PATH = Path(os.environ["SCI_WS"]) / "external" / "sci_event" / "code" / "aim_console.py"
SRC = SRC_PATH.read_text(encoding="utf-8")
T = ast.parse(SRC)


def seg(cls, name):
    for n in ast.walk(T):
        if isinstance(n, ast.ClassDef) and n.name == cls:
            for f in n.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    return ast.get_source_segment(SRC, f)
    return ""


def test_before_focus_ok_is_not_half_of_best():
    """10-15: ok=true at sharpness 2-5 because ok = final >= best*0.5."""
    assert "final >= best * 0.5" not in SRC


def test_before_completion_is_not_a_global_job_counter():
    """10-15: E5 took 'job counter advanced' as its own result; manual /focus also advanced it (finish -> job+1)."""
    assert "self.job += 1" not in SRC
    assert '"job": a.job' not in seg("Handler", "do_GET")


def test_before_state_advertises_r1_protocol():
    assert "R1O.route" in seg("Handler", "do_GET") and "OWNER[\"o\"].state()" in seg("Handler", "do_GET")


def test_before_aim_reply_is_request_scoped():
    """Baseline /aim replies plain text '시작' / '이미 작업 중입니다' without a request id."""
    assert "이미 작업 중입니다" not in SRC


def test_before_focus_budget_is_planned_bidirectionally():
    """Baseline: blurry start -> +dir 24 big steps then -dir 48, cut by a 15 s budget -> the -dir is never reached."""
    assert "(direction, 24), (-direction, 48)" not in seg("Focuser", "autofocus") or "R1F.FocusSearch" in SRC
    assert "R1F.FocusSearch" in SRC
