"""R34: the staged RGB-only proposal. SYNTHETIC. It must stay off, invent no distance and send nothing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "candidate/live2"))
import e6_rgb_only as R  # noqa: E402
K = [606.8, 0, 310.5, 0, 606.7, 240.5, 0, 0, 1]
ENV = dict(pan=[60.0, 135.0], tilt=[135.0, 167.189])


def test_off_by_default():
    r = R.propose(300, 200, K, True, ENV)
    assert not r["ok"] and r["status"] == "STAGED_NOT_ENABLED" and r["command_sent"] is False


def test_enabled_gives_a_ray_only_and_no_distance():
    r = R.propose(300, 200, K, True, ENV, enabled=True)
    assert r["ok"] and r["distance_m"] is None and r["target_3d"] is None and r["usable_for_e8"] is False and r["command_sent"] is False


def test_refused_while_moving_or_outside_the_range():
    assert R.propose(300, 200, K, False, ENV, enabled=True)["status"] == "REFUSED_MOVING"
    assert R.propose(-2000, 200, K, True, ENV, enabled=True)["status"] == "OUTSIDE_VERIFIED_ENVELOPE"
