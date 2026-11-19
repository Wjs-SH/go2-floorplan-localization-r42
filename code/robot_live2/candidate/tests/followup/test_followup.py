"""SEM1 follow-up tests (ROBOT_CPU_FAKE / LOCAL_CPU_FUNCTION; no ROS, no model, no GPU, no device).
Uses the real follow-up review folder ($RT_ROOT/review) built from robot files and the TEST_ONLY export produced by the
real UI JavaScript (headless Firefox round trip)."""
import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest

WS = Path(os.environ["SCI_WS"])
RT = Path(os.environ["RT_ROOT"])
REVIEW = RT / "review"
EXPORT = REVIEW / "human_exports_TEST_ONLY"
for p in (WS / "src" / "sci_visual_observation", WS / "src" / "sci_sensor_bridge", WS / "tools", WS / "external" / "sci_event" / "e5_onboard"):
    sys.path.insert(0, str(p))
import followup_c1c2 as FC  # noqa: E402
from e5rt import budget as BG  # noqa: E402
from e5rt.data_audit import verify_encoder_input, split_leaks  # noqa: E402


def _csv_rows(text):
    return list(csv.DictReader(io.StringIO(text, newline="")))


# ---------------------------------------------------------------- provenance (R-E)
def test_v2_scores_with_v3_labels_and_prompt_mismatch_are_refused(tmp_path):
    import e5rt2_accuracy as A
    cfg = dict(A.DATASETS["test_S6F_v3"], run=Path("/home/unitree/sci_ws/experiments/test_S6F_v2"))
    with pytest.raises(SystemExit, match="LABEL_RUN_MISMATCH"):
        A.check_label_run(cfg)
    run = tmp_path / "experiments" / "fake_run"
    run.mkdir(parents=True)
    (run / "scenes.jsonl").write_text("")
    (run / "siglip_prompts.json").write_text(json.dumps(dict(env="wall")))
    lab = tmp_path / "labels.json"
    lab.write_text(json.dumps(dict(run="experiments/fake_run", labels={})))
    with pytest.raises(SystemExit, match="PROMPT_MISMATCH"):
        A.check_label_run(dict(run=run, labels=lab, expect_prompts="roomstd"))
    (run / "siglip_prompts.json").write_text(json.dumps(dict(env="roomstd")))
    lab.write_text(json.dumps(dict(run="experiments/fake_run", labels={}, run_fingerprint=dict(scenes_sha256="0" * 64))))
    with pytest.raises(SystemExit, match="RUN_FINGERPRINT_MISMATCH"):
        A.check_label_run(dict(run=run, labels=lab, expect_prompts="roomstd"))


def test_human_verified_is_explicit_not_inferred_from_labeller_text():
    src = (WS / "tools" / "e5rt2_accuracy.py").read_text()
    assert 'lab_doc.get("human_verified") is True' in src and 'startswith("human")' not in src


# ---------------------------------------------------------------- UI export -> loader
def test_ui_real_export_csv_and_json_load_identically_with_quoting_and_korean():
    acc_c, rej_c = FC.load_human(EXPORT / "TEST_ONLY_HUMAN_BBOX.csv", REVIEW)
    acc_j, rej_j = FC.load_human(EXPORT / "TEST_ONLY_sem1_labels_state.json", REVIEW)
    assert not rej_c and not rej_j and set(acc_c) == set(acc_j) and len(acc_c) == 2
    for k in acc_c:
        assert acc_c[k]["status"] == "TEST_ONLY" and acc_c[k]["human_verified"] is False
        assert acc_c[k]["box"] == acc_j[k]["box"]
    notes = [v["note"] for v in acc_c.values()]
    assert "쉼표, \"따옴표\"\n줄바꿈 한글 테스트" in notes                         # comma, quote, newline, Korean survive
    raw = (EXPORT / "TEST_ONLY_HUMAN_BBOX.csv").read_text(encoding="utf-8")
    hdr = raw.splitlines()[0].split(",")
    for c in ("dataset_id", "run_id", "item", "physical_target_id", "visit_id", "frame_path", "frame_id", "frame_sha256", "width", "height",
              "x0", "y0", "x1", "y1", "visual_type", "localization_role", "text_readability", "reviewer", "reviewed_at", "human_verified", "label_revision"):
        assert c in hdr
    # the UI forced UNSURE where no exact encoder input exists (the script asked for 'Y')
    assert all(v["target_in_model_crop"] == "UNSURE" for v in acc_c.values())


def test_default_selection_excludes_test_only_and_unverified():
    acc, _ = FC.load_human(EXPORT / "TEST_ONLY_HUMAN_BBOX.csv", REVIEW)
    assert [v for v in acc.values() if v["status"] == "ACCEPTED"] == []


def _mutated_csv(tmp_path, fn):
    rows = _csv_rows((EXPORT / "TEST_ONLY_HUMAN_BBOX.csv").read_text(encoding="utf-8-sig"))
    rows = fn(rows)
    p = tmp_path / "x.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return p


def test_box_on_other_frame_or_wrong_size_or_bad_box_is_refused(tmp_path):
    items = json.loads((REVIEW / "ui_items.json").read_text())["items"]
    pos = next(i for i in items if i["item"] == "S-1F|spot39")
    other = pos["frames"][1]

    def wrong_sha(rows):
        rows[0]["frame_sha256"] = other["frame_sha256"]                               # sha of another frame of the same spot
        return rows
    acc, rej = FC.load_human(_mutated_csv(tmp_path, wrong_sha), REVIEW)
    assert any("HUMAN_BOX_FRAME_MISMATCH:sha256" in r["reject"] for r in rej)

    def bad(rows):
        r0, r1 = dict(rows[0]), dict(rows[0])
        r0.update(x1="700"); r1.update(x0="100", x1="103", box_id="tiny")
        r2 = dict(rows[0], width="1280")
        return [r0, r1, r2]
    acc, rej = FC.load_human(_mutated_csv(tmp_path, bad), REVIEW)
    reasons = sorted(w for r in rej for w in r["reject"])
    assert "BOX_OUT_OF_RANGE" in reasons and "BOX_TOO_SMALL" in reasons and "HUMAN_BOX_FRAME_MISMATCH:size" in reasons


def test_duplicate_revision_conflict_and_newest_revision_wins(tmp_path):
    def dup(rows):
        a = dict(rows[0]); b = dict(rows[0], x1=str(float(rows[0]["x1"]) + 5))          # same revision, different box
        c = dict(rows[0], box_revision=str(int(rows[0]["box_revision"]) + 1), x1=str(float(rows[0]["x1"]) + 9))
        return [a, b, c]
    acc, rej = FC.load_human(_mutated_csv(tmp_path, dup), REVIEW)
    assert any("DUPLICATE_REVISION_CONFLICT" in r["reject"] for r in rej)
    (only,) = acc.values()
    assert only["box_revision"] == int(_csv_rows((EXPORT / "TEST_ONLY_HUMAN_BBOX.csv").read_text(encoding="utf-8-sig"))[0]["box_revision"]) + 1


# ---------------------------------------------------------------- C0/C1/C2 on the SAME frame
def test_c1_uses_the_frame_the_human_drew_on_not_the_best_frame():
    import cv2
    acc, _ = FC.load_human(EXPORT / "TEST_ONLY_HUMAN_BBOX.csv", REVIEW)
    rec = next(v for v in acc.values() if v["item"] == "S-1F|spot39")
    items = json.loads((REVIEW / "ui_items.json").read_text())["items"]
    it = next(i for i in items if i["item"] == "S-1F|spot39")
    f = next(x for x in it["frames"] if x["frame_sha256"] == rec["frame_sha256"])
    assert f["clip_role"] != "BEST"                                                      # drawn on the FIRST frame
    sc = FC.FakeScorer()
    rows = FC.run_arms({rec["key"]: rec}, REVIEW, ["C1_whole"], sc)
    direct = FC.square_crop(cv2.imread(f["source_dataset_path"]), rec["box"])           # original dataset file
    assert rows[0]["crop_sha256"] == hashlib.sha256(direct.tobytes()).hexdigest()
    assert hashlib.sha256(Path(f["source_dataset_path"]).read_bytes()).hexdigest() == rec["frame_sha256"]


def test_selected_arms_only_no_ocr_engine_and_c2_really_computed():
    for m in [k for k in sys.modules if k.startswith("rapidocr")]:
        del sys.modules[m]
    acc, _ = FC.load_human(EXPORT / "TEST_ONLY_HUMAN_BBOX.csv", REVIEW)
    sc = FC.FakeScorer()
    arms = FC.expand_arms("C0,C1,C2", [0.25, 0.5])
    assert arms == ["C0", "C1_tile", "C1_whole", "C2_pad0.25", "C2_pad0.5", "C2_rect"]
    rows = FC.run_arms(acc, REVIEW, arms, sc)
    assert not any(k.startswith("rapidocr") for k in sys.modules) and "torch" not in sys.modules
    by = {(r["item"], r["arm"]): r for r in rows}
    for item in ("S-1F|spot39", "S-1F|spot51"):
        for a in ("C1_tile", "C1_whole", "C2_pad0.25", "C2_pad0.5"):
            assert by[(item, a)]["result"] == "SCORED" and isinstance(by[(item, a)]["score"], float)
        assert by[(item, "C2_pad0.5")]["roi"] != by[(item, "C1_whole")]["roi"]
    assert by[("S-1F|spot39", "C2_rect")]["result"] == "NOT_RUN" and "NO_HUMAN_POLYGON" in by[("S-1F|spot39", "C2_rect")]["why"]
    assert by[("S-1F|spot51", "C2_rect")]["result"] == "SCORED"                         # the poster record has a polygon
    with pytest.raises(SystemExit):
        sys.argv = ["x", "--human", str(EXPORT / "TEST_ONLY_HUMAN_BBOX.csv"), "--review-root", str(REVIEW), "--arms", "C3", "--out", "/dev/null"]
        FC.main()


def test_preview_is_never_presented_as_encoder_input():
    items = json.loads((REVIEW / "ui_items.json").read_text())["items"]
    n_exact = 0
    for it in items:
        for f in it["frames"]:
            e = f["encoder_input"]
            if f.get("preview"):
                ok, why = verify_encoder_input(REVIEW / f["preview"], {})
                assert not ok
            if e["status"].startswith("EXACT"):
                n_exact += 1
                for t in e["tiles"]:
                    assert verify_encoder_input(REVIEW / t["png"], t)[0]
            else:
                assert e["status"] == "NOT_AVAILABLE" and not e.get("tiles")
    assert n_exact == 60
    html = (REVIEW / "LABEL_AND_BOX_UI.html").read_text()
    assert "모델 입력 아님" in html and "UNSURE (실제 모델 입력 없음)" in html


# ---------------------------------------------------------------- budget parity (R-C)
def _runtime_select(spot_info, thr, budget, regs):
    return BG.value_plan(regs, budget, spot_info, thr)[0]


def _sim_select(regs, state, thr, budget, now=10.0):
    import sem1_budget_sim as SIM
    cost = [BG.n_tiles(g["box"]) + 1 for g in regs]
    return BG.select_value(SIM.candidates(regs, state, now, cost), budget, thr)[0]


def test_budget_counterexample_now_identical_between_simulator_and_runtime():
    regs = [dict(box=[0, 0, 20, 20], spot=0), dict(box=[40, 0, 60, 20], spot=1), dict(box=[80, 0, 100, 20], spot=2)]
    state = {0: dict(sims=[.09], last_fresh=9.0, semantic_confirmed=False)}
    sim = _sim_select(regs, state, .092, 4)
    rt = _runtime_select([dict(new=False, sims=[.09], waited_s=1.0, semantic_confirmed=False, observed=False), dict(new=True), dict(new=True)],
                         .092, 4, [g["box"] for g in regs])
    assert sim == rt == [0, 1]                       # existing unconfirmed near threshold + one exploration slot


def test_budget_parity_random_causal_states():
    rng = random.Random(7)
    for _ in range(300):
        n = rng.randint(2, 7)
        regs = [dict(box=[0, 0, rng.choice([20, 120, 220]), rng.choice([20, 120])], spot=i) for i in range(n)]
        thr = rng.choice([0.085, 0.092, math.inf])
        state, info = {}, []
        for i in range(n):
            if rng.random() < 0.4:
                info.append(dict(new=True))
                continue
            sims = [round(rng.uniform(0.05, 0.13), 3) for _ in range(rng.randint(1, 4))]
            conf = rng.random() < 0.3
            waited = round(rng.uniform(0, 5), 2)
            state[i] = dict(sims=sims, last_fresh=10.0 - waited, semantic_confirmed=conf)
            info.append(dict(new=False, sims=sims, waited_s=waited, semantic_confirmed=conf, observed=False))
        budget = rng.choice([4, 6, 9, 12])
        if sum(BG.n_tiles(g["box"]) + 1 for g in regs) <= budget:
            continue
        assert _sim_select(regs, state, thr, budget) == _runtime_select(info, thr, budget, [g["box"] for g in regs])


def test_semantic_confirmed_separate_from_fired_and_cached_does_not_refresh_last_fresh():
    class Inner:
        def score(self, img, rr):
            n = len(rr)
            self.last_n_tiles = sum(BG.n_tiles(b) for b in rr)
            self.last_score_status = ["CACHED"] * n
            self.last_adv, self.last_board, self.last_ceil = [0.0] * n, [-1.0] * n, [-1.0] * n
            self.last_tiles, self.last_emb, self.last_ctx_plate, self.last_neg, self.last_ctx = [list(r) for r in rr], [b"e"] * n, [0.1] * n, {}, {}
            return np.full(n, 0.1), 0.0
    L = types.SimpleNamespace(spots=[dict(sims=[0.2, 0.21], ts=[1.0, 2.0]), dict(sims=[0.05], ts=[1.0])], seen=[0.05] * 60 + [0.2] * 6, fired={1})
    b = BG.BudgetedSiglip(Inner(), legacy_events=L, max_crops=4, policy="value")
    regs = [[0, 0, 20, 20], [40, 0, 60, 20], [80, 0, 100, 20], [120, 0, 140, 20]]
    b.set_context(region_world=lambda bx, d, vp, pose: ([float(bx[0]), 0, 1], 1.0), dimg=None, vp=None, pose=None,
                  match=lambda spots, xyz, r, v, t: {0.0: 0, 40.0: 1}.get(xyz[0]), P=dict(radius=0.7, quant=0.9, warm=50))
    info, thr = b._spot_info(regs)
    assert info[0]["semantic_confirmed"] and not info[0]["observed"]                   # confirmed, no STOP
    assert info[1]["observed"] and not info[1]["semantic_confirmed"]                   # fired but not semantically confirmed
    b.score(np.zeros((40, 200, 3), np.uint8), regs)
    assert b.spot_last_fresh == {}                                                       # CACHED results: no fresh update


# ---------------------------------------------------------------- ros_readonly geometry contract (R-D)
def _src(contract=None, wall=lambda: 1000.0):
    import ros_readonly as RR
    s = RR.RosReadonlySource.__new__(RR.RosReadonlySource)
    s._init_state(contract=contract, wall=wall)
    s.visual_only = False
    return s, RR


def _odom(st, x=0.0, y=0.0, q=(0, 0, 0, 1), frame="camera_init", child="body"):
    N = types.SimpleNamespace
    return N(header=N(stamp=N(sec=int(st), nanosec=int(round((st % 1) * 1e9))), frame_id=frame), child_frame_id=child,
             pose=N(pose=N(position=N(x=x, y=y, z=0.0), orientation=N(x=q[0], y=q[1], z=q[2], w=q[3]))))


def test_odom_callback_rejects_bad_samples_and_resets_sessions():
    s, _ = _src()
    s._on_odom(_odom(100.0, x=float("nan")))
    s._on_odom(_odom(100.1, q=(0, 0, 0, 2.0)))
    s._on_odom(_odom(2000.0))                                                            # far future vs wall 1000
    s._on_odom(_odom(999.0)); s._on_odom(_odom(999.0)); s._on_odom(_odom(999.1, x=5.0))  # duplicate, jump
    s._on_odom(_odom(999.2, x=0.05))
    s._on_odom(_odom(990.0))                                                             # backwards -> reset
    s._on_odom(_odom(990.1, frame="odom"))                                               # producer frame change
    r = s.health()["odom"]["rejected"]
    assert r["NON_FINITE"] == 1 and r["QUATERNION_NOT_UNIT"] == 1 and r["FUTURE_STAMP"] == 1 and r["DUPLICATE_OR_REORDERED"] == 1
    assert r["JUMP"] == 1 and r["TIME_BACKWARDS_RESET_NEW_SESSION"] == 1 and r["FRAME_CHANGED_NEW_SESSION"] == 1
    assert s.odom_session == 2 and [o[0] for o in s.odom_raw_hist()] == [990.1]         # fresh without poll()
    assert s.odom_hist == []                     # LIVE1: no verified contract -> no body-frame history for geometry


def test_pose_requires_verified_frame_contract_and_never_double_transforms():
    s, RR = _src(contract=None)
    s._on_odom(_odom(999.0, x=1.0))
    assert s.geometry_status == "UNVERIFIED_FRAME_CONTRACT" and s.pose(999.0) == (None, None)
    s.visual_identity_pose = True                # LIVE1: flagged identity for semantic search only, geometry stays off
    assert s.pose(999.0) == ((0.0, 0.0, 0.0), 999.0) and s.pose_mode() == "IDENTITY_FLAGGED_CONTRACT_MISSING" and s.odom_hist == []
    ident = dict(frame_id="camera_init", child_frame_id="body", body_transform=dict(kind="identity"), verified_by="tester", evidence="unit")
    s2, _ = _src(contract=ident)
    s2._on_odom(_odom(999.0, x=1.0))
    assert s2.pose(999.05) == ((1.0, 0.0, 0.0), 999.0)                                   # already body: no extra transform
    assert s2.pose(999.5) == (None, None)                                                # > max_gap from the image stamp
    s2._on_odom(_odom(999.1, x=1.0, frame="map"))
    assert s2.health()["odom"]["rejected"].get("FRAME_CONTRACT_MISMATCH") == 1
    bad = dict(ident, body_transform=dict(kind="fastlio_mirror_lever_yaw"))              # no delta / lever given
    s3, _ = _src(contract=bad)
    assert s3.geometry_status == "UNVERIFIED_FRAME_CONTRACT"


def test_fastlio_contract_matches_existing_odom_base_formula_with_explicit_values():
    from odom_base import fastlio_to_base_2d
    c = dict(frame_id="camera_init", child_frame_id="body", verified_by="tester", evidence="unit",
             body_transform=dict(kind="fastlio_mirror_lever_yaw", delta_deg=33.0, lever_m=[0.026, 0.0]))
    s, _ = _src(contract=c)
    for raw in [(1.0, 2.0, 0.3), (-3.0, 0.5, -1.2)]:
        assert np.allclose(s.body_pose(raw), fastlio_to_base_2d(*raw, delta_deg=33.0, lever=(0.026, 0.0)))
    src = (WS / "src" / "sci_sensor_bridge" / "ros_readonly.py").read_text()
    assert "45.3" not in src.replace("(e.g. 45.3 or 228 deg)", "") and "228" not in src.replace("(e.g. 45.3 or 228 deg)", "")


def test_image_decoding_honours_step_endianness_and_rejects_unsupported():
    s, RR = _src()
    N = types.SimpleNamespace
    rgb = (np.arange(4 * 3 * 3) % 255).astype(np.uint8).reshape(4, 3, 3)
    padded = np.zeros((4, 12), np.uint8)
    padded[:, :9] = rgb.reshape(4, 9)
    out = RR.RosReadonlySource._img(N(encoding="rgb8", width=3, height=4, step=12, is_bigendian=0, data=padded.tobytes()))
    assert np.array_equal(out, rgb[:, :, ::-1])
    d = np.array([[1, 513], [258, 4096]], np.uint16)
    out = RR.RosReadonlySource._img(N(encoding="16UC1", width=2, height=2, step=4, is_bigendian=1, data=d.astype(">u2").tobytes()))
    assert np.array_equal(out, d)
    with pytest.raises(ValueError, match="UNSUPPORTED_ENCODING"):
        RR.RosReadonlySource._img(N(encoding="yuv422", width=2, height=2, step=4, is_bigendian=0, data=b"\0" * 8))
    s.color.push((1.0, 1, 0, N(encoding="yuv422", width=2, height=2, step=4, is_bigendian=0, data=b"\0" * 8)), 8)
    assert s.poll() == [] and s.health()["decode_rejected"] == {"UNSUPPORTED_ENCODING": 1}


# ---------------------------------------------------------------- denominators / leakage
def test_budget_rows_keep_full_denominators_and_split_leaks_detected():
    rows = json.loads((RT / "reports" / "budget_parity_sim.json").read_text())
    for r in rows:
        assert r["tp"] + r["fn"] == r["n_pos"] and r["positive_visits"] >= r["n_pos"] - 1
    assert split_leaks([dict(physical_target_id="T1", split="DEV"), dict(physical_target_id="T1", split="FINAL")]) == {"T1": ["DEV", "FINAL"]}
